import torch
import torch.nn as nn

from blocks import LayerNorm, Encoder, Decoder
from transformers import AutoModel, AutoTokenizer
from typing import Optional, Dict, Any, Tuple
from pathlib import Path


#############
### Model ###
#############


class DPG(nn.Module):
    """
    Encoder: DNABERT-2 Weights converted to Bigbird Spare Attention
    Decoder:
        FF MoE
        No position embedding
    """
    
    def __init__(
        
        self,
        
        # Encoder config (from DNABERT-2)
        encoder: nn.Module,
        encoder_embed_dim: int,
        
        # Decoder config
        decoder_num_layers:         int = 6,
        decoder_num_heads:          int = 8,
        decoder_ff_dim:             int = 2048,
        decoder_dropout:            float = 0.1,
        decoder_attn_dropout:       float = 0.1,
        decoder_use_alibi:          bool = True,
        decoder_use_moe:            bool = False,
        decoder_num_experts:        int = 8,
        decoder_moe_top_k:          int = 2,
        decoder_moe_load_balance:   float = 0.01,
        
        # Output config
        vocab_size: int = 4096,
        
        # Freeze config
        freeze_encoder: bool = True
    ):
        super().__init__()
        
        self.encoder_embed_dim  = encoder_embed_dim
        self.freeze_encoder     = freeze_encoder
        
        # Encoder DNABERT-2
        self.encoder = encoder
        
        # Decoder
        self.decoder = Decoder(
            num_layers      =decoder_num_layers,
            embed_dim       =encoder_embed_dim,
            num_heads       =decoder_num_heads,
            ff_dim          =decoder_ff_dim,
            dropout         =decoder_dropout,
            attn_dropout    =decoder_attn_dropout,
            use_alibi       =decoder_use_alibi,
            use_moe         =decoder_use_moe,
            num_experts     =decoder_num_experts,
            moe_top_k       =decoder_moe_top_k,
            moe_load_balance=decoder_moe_load_balance
        )
        
        # Embeddings for decoder input
        self.decoder_embed          = nn.Embedding(vocab_size, encoder_embed_dim)
        self.decoder_embed_dropout  = nn.Dropout(decoder_dropout)
        
        # Output projection (language model head)
        self.lm_head = nn.Linear(encoder_embed_dim, vocab_size, bias=False)
        
        # Initialize decoder weights
        self._init_decoder_weights()
        
        # Apply freeze setting
        if freeze_encoder:
            self.freeze_encoder_weights()
    
    def _init_decoder_weights(self):
        """ Initialize decoder weights with Xavier/Glorot uniform """
        
        def _init_weights(module):
            
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
                    
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        
        self.decoder.apply(_init_weights)
        self.decoder_embed.apply(_init_weights)
        self.lm_head.apply(_init_weights)
        
        print("Decoder weights randomly initialized")
    
    def freeze_encoder_weights(self):
        """Freeze all encoder parameters"""
        
        for param in self.encoder.parameters():
            param.requires_grad = False
        self.freeze_encoder = True
        print("Encoder weights frozen")
    
    def unfreeze_encoder_weights(self):
        """Unfreeze all encoder parameters for fine-tuning"""
        
        for param in self.encoder.parameters():
            param.requires_grad = True
        self.freeze_encoder = False
        print("✓ Encoder weights unfrozen")
    
    def freeze_encoder_layers(self, num_layers_to_freeze):
        """Freeze only the first N encoder layers (gradual unfreezing)"""
        
        for i, layer in enumerate(self.encoder.layers):
            for param in layer.parameters():
                param.requires_grad = (i >= num_layers_to_freeze)
        
        frozen = min(num_layers_to_freeze, len(self.encoder.layers))
        trainable = len(self.encoder.layers) - frozen
        print(f"✓ Frozen {frozen} encoder layers, {trainable} layers trainable")
    
    def get_trainable_params(self):
        """Get count of trainable vs frozen parameters"""
        
        encoder_trainable   = sum(p.numel() for p in self.encoder.parameters() if p.requires_grad)
        encoder_frozen      = sum(p.numel() for p in self.encoder.parameters() if not p.requires_grad)
        decoder_trainable   = sum(p.numel() for p in self.decoder.parameters() if p.requires_grad)
        embed_trainable     = sum(p.numel() for p in self.decoder_embed.parameters() if p.requires_grad)
        head_trainable      = sum(p.numel() for p in self.lm_head.parameters() if p.requires_grad)
        
        total_trainable = encoder_trainable + decoder_trainable + embed_trainable + head_trainable
        total_frozen    = encoder_frozen
        
        return {
            "encoder_trainable": encoder_trainable,
            "encoder_frozen": encoder_frozen,
            "decoder_trainable": decoder_trainable,
            "embed_trainable": embed_trainable,
            "head_trainable": head_trainable,
            "total_trainable": total_trainable,
            "total_frozen": total_frozen,
            "total_params": total_trainable + total_frozen
        }
    
    def print_trainable_params(self):
        """Print parameter statistics"""
        
        stats = self.get_trainable_params()
        print("\n" + "="*50)
        print("Parameter Statistics:")
        print("="*50)
        print(f"Encoder trainable:  {stats['encoder_trainable']:,}")
        print(f"Encoder frozen:     {stats['encoder_frozen']:,}")
        print(f"Decoder trainable:  {stats['decoder_trainable']:,}")
        print(f"Embeddings:         {stats['embed_trainable']:,}")
        print(f"LM Head:            {stats['head_trainable']:,}")
        print("-"*50)
        print(f"Total trainable:    {stats['total_trainable']:,}")
        print(f"Total frozen:       {stats['total_frozen']:,}")
        print(f"Total parameters:   {stats['total_params']:,}")
        print("="*50 + "\n")
    
    def forward(
        self,
        encoder_input_ids: torch.Tensor,
        decoder_input_ids: torch.Tensor,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        decoder_attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,  # For caching
        return_encoder_hidden_states: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through encoder-decoder
        
        Args:
            encoder_input_ids: (B, src_len) - Source sequence token IDs
            decoder_input_ids: (B, tgt_len) - Target sequence token IDs
            encoder_attention_mask: (B, src_len) - Mask for encoder (1=attend, 0=ignore)
            decoder_attention_mask: (B, tgt_len) - Mask for decoder
            encoder_hidden_states: Pre-computed encoder outputs (for inference caching)
            return_encoder_hidden_states: Whether to return encoder outputs
        
        Returns:
            Dictionary with logits, optional encoder hidden states, optional moe_loss
        """
        # Encode (or use cached)
        if encoder_hidden_states is None:
            # Need embeddings - assuming encoder expects embedded input
            # If your encoder handles token IDs directly, adjust this
            encoder_hidden_states = self.encoder(
                encoder_input_ids,  # Your encoder's forward signature
                attention_mask=encoder_attention_mask
            )
        
        # Embed decoder inputs
        decoder_embeds = self.decoder_embed(decoder_input_ids)
        decoder_embeds = self.decoder_embed_dropout(decoder_embeds)
        
        # Decode
        decoder_output, moe_loss = self.decoder(
            hidden_states=decoder_embeds,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=decoder_attention_mask,
            encoder_attention_mask=encoder_attention_mask
        )
        
        # Project to vocabulary
        logits = self.lm_head(decoder_output)
        
        outputs = {"logits": logits}
        
        if moe_loss is not None:
            outputs["moe_loss"] = moe_loss
        
        if return_encoder_hidden_states:
            outputs["encoder_hidden_states"] = encoder_hidden_states
        
        return outputs
    
    def save_weights(
        self,
        save_dir: str,
        save_encoder: bool = True,
        save_decoder: bool = True,
        save_optimizer: Optional[torch.optim.Optimizer] = None,
        save_scheduler: Optional[Any] = None,
        epoch: Optional[int] = None,
        step: Optional[int] = None,
        additional_info: Optional[Dict] = None
    ):
        """
        Save model weights and training state
        
        Args:
            save_dir: Directory to save weights
            save_encoder: Whether to save encoder weights
            save_decoder: Whether to save decoder weights
            save_optimizer: Optimizer state to save
            save_scheduler: Scheduler state to save
            epoch: Current epoch number
            step: Current step number
            additional_info: Any additional metadata to save
        """
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            "config": {
                "encoder_embed_dim": self.encoder_embed_dim,
                "freeze_encoder": self.freeze_encoder,
            }
        }
        
        if save_encoder:
            checkpoint["encoder_state_dict"] = self.encoder.state_dict()
            print(f"  ✓ Encoder weights included")
        
        if save_decoder:
            checkpoint["decoder_state_dict"] = self.decoder.state_dict()
            checkpoint["decoder_embed_state_dict"] = self.decoder_embed.state_dict()
            checkpoint["lm_head_state_dict"] = self.lm_head.state_dict()
            print(f"  ✓ Decoder weights included")
        
        if save_optimizer is not None:
            checkpoint["optimizer_state_dict"] = save_optimizer.state_dict()
            print(f"  ✓ Optimizer state included")
        
        if save_scheduler is not None:
            checkpoint["scheduler_state_dict"] = save_scheduler.state_dict()
            print(f"  ✓ Scheduler state included")
        
        if epoch is not None:
            checkpoint["epoch"] = epoch
        
        if step is not None:
            checkpoint["step"] = step
        
        if additional_info is not None:
            checkpoint["additional_info"] = additional_info
        
        # Create filename
        filename = "checkpoint"
        if epoch is not None:
            filename += f"_epoch{epoch}"
        if step is not None:
            filename += f"_step{step}"
        filename += ".pt"
        
        save_file = save_path / filename
        torch.save(checkpoint, save_file)
        print(f"✓ Checkpoint saved to: {save_file}")
        
        # Also save a 'latest' symlink/copy for easy loading
        latest_file = save_path / "checkpoint_latest.pt"
        torch.save(checkpoint, latest_file)
        
        return str(save_file)
    
    def load_weights(
        self,
        checkpoint_path: str,
        load_encoder: bool = True,
        load_decoder: bool = True,
        strict: bool = True
    ) -> Dict[str, Any]:
        """
        Load model weights from checkpoint
        
        Args:
            checkpoint_path: Path to checkpoint file
            load_encoder: Whether to load encoder weights
            load_decoder: Whether to load decoder weights
            strict: Whether to strictly enforce state dict matching
        
        Returns:
            Dictionary with loaded checkpoint info (epoch, step, etc.)
        """
        print(f"Loading checkpoint from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        
        if load_encoder and "encoder_state_dict" in checkpoint:
            self.encoder.load_state_dict(checkpoint["encoder_state_dict"], strict=strict)
            print("  ✓ Encoder weights loaded")
        
        if load_decoder:
            if "decoder_state_dict" in checkpoint:
                self.decoder.load_state_dict(checkpoint["decoder_state_dict"], strict=strict)
                print("  ✓ Decoder weights loaded")
            
            if "decoder_embed_state_dict" in checkpoint:
                self.decoder_embed.load_state_dict(checkpoint["decoder_embed_state_dict"], strict=strict)
                print("  ✓ Decoder embeddings loaded")
            
            if "lm_head_state_dict" in checkpoint:
                self.lm_head.load_state_dict(checkpoint["lm_head_state_dict"], strict=strict)
                print("  ✓ LM head weights loaded")
        
        # Return training state info
        info = {}
        if "epoch" in checkpoint:
            info["epoch"] = checkpoint["epoch"]
        if "step" in checkpoint:
            info["step"] = checkpoint["step"]
        if "optimizer_state_dict" in checkpoint:
            info["optimizer_state_dict"] = checkpoint["optimizer_state_dict"]
        if "scheduler_state_dict" in checkpoint:
            info["scheduler_state_dict"] = checkpoint["scheduler_state_dict"]
        if "additional_info" in checkpoint:
            info["additional_info"] = checkpoint["additional_info"]
        
        print("✓ Checkpoint loaded successfully")
        return info
    
    def get_optimizer_param_groups(
        self,
        encoder_lr: float = 1e-5,
        decoder_lr: float = 1e-4,
        weight_decay: float = 0.01,
        no_decay_keywords: Tuple[str, ...] = ("bias", "LayerNorm", "layer_norm")
    ) -> list:
        """
        Get parameter groups with different learning rates for encoder vs decoder
        
        Args:
            encoder_lr: Learning rate for encoder (typically smaller)
            decoder_lr: Learning rate for decoder (typically larger)
            weight_decay: Weight decay coefficient
            no_decay_keywords: Parameter name keywords that shouldn't have weight decay
        
        Returns:
            List of parameter groups for optimizer
        """
        def should_decay(name):
            return not any(nd in name.lower() for nd in no_decay_keywords)
        
        param_groups = []
        
        # Encoder parameters (if trainable)
        encoder_decay = []
        encoder_no_decay = []
        for name, param in self.encoder.named_parameters():
            if param.requires_grad:
                if should_decay(name):
                    encoder_decay.append(param)
                else:
                    encoder_no_decay.append(param)
        
        if encoder_decay:
            param_groups.append({
                "params": encoder_decay,
                "lr": encoder_lr,
                "weight_decay": weight_decay,
                "name": "encoder_decay"
            })
        
        if encoder_no_decay:
            param_groups.append({
                "params": encoder_no_decay,
                "lr": encoder_lr,
                "weight_decay": 0.0,
                "name": "encoder_no_decay"
            })
        
        # Decoder parameters
        decoder_decay = []
        decoder_no_decay = []
        for name, param in list(self.decoder.named_parameters()) + \
                          list(self.decoder_embed.named_parameters()) + \
                          list(self.lm_head.named_parameters()):
            if param.requires_grad:
                if should_decay(name):
                    decoder_decay.append(param)
                else:
                    decoder_no_decay.append(param)
        
        if decoder_decay:
            param_groups.append({
                "params": decoder_decay,
                "lr": decoder_lr,
                "weight_decay": weight_decay,
                "name": "decoder_decay"
            })
        
        if decoder_no_decay:
            param_groups.append({
                "params": decoder_no_decay,
                "lr": decoder_lr,
                "weight_decay": 0.0,
                "name": "decoder_no_decay"
            })
        
        return param_groups


# =============================================================================
# Factory function to create the full model
# =============================================================================

def create_dnabert2_encoder_decoder(
    dnabert2_model_name: str = "zhihan1996/DNABERT-2-117M",
    block_size: int = 64,
    num_rand_blocks: int = 3,
    decoder_num_layers: int = 6,
    decoder_num_heads: int = 8,
    decoder_ff_dim: int = 2048,
    decoder_dropout: float = 0.1,
    decoder_use_alibi: bool = True,
    decoder_use_moe: bool = False,
    vocab_size: int = 4096,
    freeze_encoder: bool = True,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> Tuple[DNABERT2EncoderDecoder, AutoTokenizer]:
    """
    Factory function to create a DNABERT-2 encoder + BigBird sparse attention
    connected to a randomly initialized decoder
    
    Returns:
        model: DNABERT2EncoderDecoder instance
        tokenizer: DNABERT-2 tokenizer
    """
    from _blocks import Encoder, LayerNorm
    
    print("="*60)
    print("Creating DNABERT-2 Encoder-Decoder Model")
    print("="*60)
    
    # Load DNABERT-2
    print(f"\n1. Loading DNABERT-2 from {dnabert2_model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(dnabert2_model_name, trust_remote_code=True)
    original_model = AutoModel.from_pretrained(dnabert2_model_name, trust_remote_code=True)
    
    config = original_model.config
    
    print(f"\n2. DNABERT-2 Configuration:")
    print(f"   Hidden size: {config.hidden_size}")
    print(f"   Num layers: {config.num_hidden_layers}")
    print(f"   Num heads: {config.num_attention_heads}")
    print(f"   Intermediate size: {config.intermediate_size}")
    
    # Create BigBird sparse encoder
    print(f"\n3. Creating BigBird sparse encoder (block_size={block_size})...")
    converted_encoder = Encoder(
        num_layers=config.num_hidden_layers,
        embed_dim=config.hidden_size,
        num_heads=config.num_attention_heads,
        ff_dim=config.intermediate_size,
        dropout=config.hidden_dropout_prob,
        attn_dropout=config.attention_probs_dropout_prob,
        use_alibi=True,
        use_bigbird_sparse=True,
        block_size=block_size,
        num_rand_blocks=num_rand_blocks
    )
    
    # Transfer weights
    print("\n4. Transferring DNABERT-2 weights to BigBird encoder...")
    original_layers = original_model.bert.encoder.layer
    
    for layer_idx, (orig_layer, new_layer) in enumerate(zip(original_layers, converted_encoder.layers)):
        # Attention weights
        orig_attn = orig_layer.attention.self
        new_attn = new_layer.self_attn
        
        new_attn.q.weight.data.copy_(orig_attn.query.weight.data)
        new_attn.k.weight.data.copy_(orig_attn.key.weight.data)
        new_attn.v.weight.data.copy_(orig_attn.value.weight.data)
        new_attn.o.weight.data.copy_(orig_layer.attention.output.dense.weight.data)
        
        # LayerNorm
        new_layer.norm1.weight.data.copy_(orig_layer.attention.output.LayerNorm.weight.data)
        
        # FFN
        orig_ff = orig_layer.intermediate
        orig_output = orig_layer.output
        
        new_layer.ff.wi_0.weight.data.copy_(orig_ff.dense.weight.data)
        new_layer.ff.wi_1.weight.data.copy_(orig_ff.dense.weight.data)
        new_layer.ff.wo.weight.data.copy_(orig_output.dense.weight.data)
        new_layer.norm2.weight.data.copy_(orig_output.LayerNorm.weight.data)
    
    # Final norm
    converted_encoder.final_norm.weight.data.copy_(
        original_model.bert.encoder.LayerNorm.weight.data
    )
    
    print("   ✓ All encoder weights transferred")
    
    # Create full encoder-decoder model
    print(f"\n5. Creating encoder-decoder model...")
    model = DNABERT2EncoderDecoder(
        encoder=converted_encoder,
        encoder_embed_dim=config.hidden_size,
        decoder_num_layers=decoder_num_layers,
        decoder_num_heads=decoder_num_heads,
        decoder_ff_dim=decoder_ff_dim,
        decoder_dropout=decoder_dropout,
        decoder_use_alibi=decoder_use_alibi,
        decoder_use_moe=decoder_use_moe,
        vocab_size=vocab_size,
        freeze_encoder=freeze_encoder
    )
    
    model = model.to(device)
    
    print(f"\n6. Model created on device: {device}")
    model.print_trainable_params()
    
    # Clean up original model
    del original_model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    return model, tokenizer


# =============================================================================
# Training utilities
# =============================================================================

class Trainer:
    """Simple trainer class for the encoder-decoder model"""
    
    def __init__(
        self,
        model: DNABERT2EncoderDecoder,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[Any] = None,
        device: str = "cuda",
        gradient_accumulation_steps: int = 1,
        max_grad_norm: float = 1.0,
        log_interval: int = 100
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_grad_norm = max_grad_norm
        self.log_interval = log_interval
        
        self.global_step = 0
        self.epoch = 0
    
    def train_step(
        self,
        encoder_input_ids: torch.Tensor,
        decoder_input_ids: torch.Tensor,
        labels: torch.Tensor,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        decoder_attention_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, float]:
        """Single training step"""
        self.model.train()
        
        # Move to device
        encoder_input_ids = encoder_input_ids.to(self.device)
        decoder_input_ids = decoder_input_ids.to(self.device)
        labels = labels.to(self.device)
        
        if encoder_attention_mask is not None:
            encoder_attention_mask = encoder_attention_mask.to(self.device)
        if decoder_attention_mask is not None:
            decoder_attention_mask = decoder_attention_mask.to(self.device)
        
        # Forward
        outputs = self.model(
            encoder_input_ids=encoder_input_ids,
            decoder_input_ids=decoder_input_ids,
            encoder_attention_mask=encoder_attention_mask,
            decoder_attention_mask=decoder_attention_mask
        )
        
        logits = outputs["logits"]
        
        # Compute loss
        loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
        loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
        
        # Add MoE loss if present
        if "moe_loss" in outputs and outputs["moe_loss"] is not None:
            loss = loss + outputs["moe_loss"]
        
        # Scale loss for gradient accumulation
        loss = loss / self.gradient_accumulation_steps
        
        # Backward
        loss.backward()
        
        metrics = {"loss": loss.item() * self.gradient_accumulation_steps}
        
        # Optimizer step (with gradient accumulation)
        if (self.global_step + 1) % self.gradient_accumulation_steps == 0:
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.max_grad_norm
            )
            
            self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step()
            self.optimizer.zero_grad()
        
        self.global_step += 1
        
        return metrics
    
    def save_checkpoint(self, save_dir: str, **kwargs):
        """Save training checkpoint"""
        return self.model.save_weights(
            save_dir=save_dir,
            save_optimizer=self.optimizer,
            save_scheduler=self.scheduler,
            epoch=self.epoch,
            step=self.global_step,
            **kwargs
        )
    
    def load_checkpoint(self, checkpoint_path: str, load_optimizer: bool = True):
        """Load training checkpoint"""
        info = self.model.load_weights(checkpoint_path)
        
        if load_optimizer and "optimizer_state_dict" in info:
            self.optimizer.load_state_dict(info["optimizer_state_dict"])
            print("  ✓ Optimizer state loaded")
        
        if "scheduler_state_dict" in info and self.scheduler is not None:
            self.scheduler.load_state_dict(info["scheduler_state_dict"])
            print("  ✓ Scheduler state loaded")
        
        if "epoch" in info:
            self.epoch = info["epoch"]
        if "step" in info:
            self.global_step = info["step"]
        
        return info


# =============================================================================
# Example usage
# =============================================================================

if __name__ == "__main__":
    
    # -------------------------------------------------------------------------
    # 1. Create model
    # -------------------------------------------------------------------------
    print("\n" + "="*70)
    print(" EXAMPLE: Creating and Training DNABERT-2 Encoder-Decoder")
    print("="*70)
    
    # This would be the actual creation (requires _blocks module and HF model)
    """
    model, tokenizer = create_dnabert2_encoder_decoder(
        dnabert2_model_name="zhihan1996/DNABERT-2-117M",
        block_size=64,
        num_rand_blocks=3,
        decoder_num_layers=6,
        decoder_num_heads=8,
        decoder_ff_dim=2048,
        freeze_encoder=True,  # Start with frozen encoder
        device="cuda"
    )
    """
    
    # -------------------------------------------------------------------------
    # 2. Setup optimizer with different learning rates
    # -------------------------------------------------------------------------
    print("\n--- Setting up optimizer ---")
    print("""
    # Get parameter groups with different LRs
    param_groups = model.get_optimizer_param_groups(
        encoder_lr=1e-5,   # Small LR for pretrained encoder
        decoder_lr=1e-4,   # Larger LR for randomly initialized decoder
        weight_decay=0.01
    )
    
    optimizer = torch.optim.AdamW(param_groups)
    
    # Optional: Learning rate scheduler
    from transformers import get_linear_schedule_with_warmup
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=1000,
        num_training_steps=100000
    )
    """)
    
    # -------------------------------------------------------------------------
    # 3. Training loop example
    # -------------------------------------------------------------------------
    print("\n--- Training loop example ---")
    print("""
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        device="cuda",
        gradient_accumulation_steps=4,
        max_grad_norm=1.0
    )
    
    for epoch in range(num_epochs):
        trainer.epoch = epoch
        
        for batch in dataloader:
            metrics = trainer.train_step(
                encoder_input_ids=batch["encoder_ids"],
                decoder_input_ids=batch["decoder_ids"],
                labels=batch["labels"],
                encoder_attention_mask=batch["encoder_mask"],
                decoder_attention_mask=batch["decoder_mask"]
            )
            
            if trainer.global_step % 100 == 0:
                print(f"Step {trainer.global_step}: Loss = {metrics['loss']:.4f}")
        
        # Save checkpoint each epoch
        trainer.save_checkpoint(f"checkpoints/epoch_{epoch}")
    """)
    
    # -------------------------------------------------------------------------
    # 4. Freezing/unfreezing strategies
    # -------------------------------------------------------------------------
    print("\n--- Freezing strategies ---")
    print("""
    # Option 1: Train decoder only (encoder frozen)
    model.freeze_encoder_weights()
    
    # Option 2: Train everything
    model.unfreeze_encoder_weights()
    
    # Option 3: Gradual unfreezing (freeze first N layers)
    model.freeze_encoder_layers(num_layers_to_freeze=8)  # Freeze first 8 layers
    
    # Check what's trainable
    model.print_trainable_params()
    """)
    
    # -------------------------------------------------------------------------
    # 5. Saving and loading
    # -------------------------------------------------------------------------
    print("\n--- Save/Load examples ---")
    print("""
    # Save full checkpoint
    model.save_weights(
        save_dir="checkpoints",
        save_encoder=True,
        save_decoder=True,
        save_optimizer=optimizer,
        save_scheduler=scheduler,
        epoch=5,
        step=10000
    )
    
    # Save decoder only (encoder weights from DNABERT-2 can be reloaded)
    model.save_weights(
        save_dir="checkpoints/decoder_only",
        save_encoder=False,
        save_decoder=True
    )
    
    # Load checkpoint
    info = model.load_weights(
        "checkpoints/checkpoint_epoch5_step10000.pt",
        load_encoder=True,
        load_decoder=True
    )
    
    # Resume training from checkpoint
    trainer.load_checkpoint("checkpoints/checkpoint_latest.pt")
    """)
    
    # -------------------------------------------------------------------------
    # 6. Inference example
    # -------------------------------------------------------------------------
    print("\n--- Inference example ---")
    print("""
    model.eval()
    
    with torch.no_grad():
        # Encode once
        encoder_ids = tokenizer(dna_sequence, return_tensors="pt")["input_ids"]
        outputs = model(
            encoder_input_ids=encoder_ids.to(device),
            decoder_input_ids=start_token.to(device),
            return_encoder_hidden_states=True
        )
        
        # Use cached encoder states for autoregressive decoding
        encoder_hidden = outputs["encoder_hidden_states"]
        
        for _ in range(max_length):
            outputs = model(
                encoder_input_ids=encoder_ids.to(device),
                decoder_input_ids=current_tokens.to(device),
                encoder_hidden_states=encoder_hidden  # Cached!
            )
            next_token = outputs["logits"][:, -1, :].argmax(-1)
            current_tokens = torch.cat([current_tokens, next_token.unsqueeze(1)], dim=1)
    """)
    
    print("\n" + "="*70)
    print(" Setup complete! Adjust paths and run with your _blocks module.")
    print("="*70)