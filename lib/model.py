
import torch
import torch.nn as nn
import torch.nn.functional as F
import json

from pathlib import Path
from typing import Optional, Dict, List, Tuple

from lib.blocks import Encoder, Decoder


class GeneT5(nn.Module):
    """
    GeneT5: DNA-to-Protein Encoder-Decoder
    
    Encoder: BigBird sparse attention
    Decoder: Causal attention + cross-attention, optional MoE
    """
    
    def __init__(
        self,
        embed_dim          = 768,
        encoder_num_layers = 12,
        encoder_num_heads  = 12,
        encoder_ff_dim     = 3072,
        decoder_num_layers = 12,
        decoder_num_heads  = 12,
        decoder_ff_dim     = 3072,
        decoder_dropout    = 0.1,
        decoder_use_alibi  = True,
        decoder_use_moe    = False,
        decoder_num_experts= 8,
        decoder_moe_top_k  = 2,
        vocab_size         = 4096,
        tie_weights        = True,
        block_size         = 64,
        num_rand_blocks    = 3
    ):
        super().__init__()
        
        self.embed_dim  = embed_dim
        self.vocab_size = vocab_size
        
        # Encoder
        self.encoder = Encoder(
            num_layers         = encoder_num_layers,
            embed_dim          = embed_dim,
            num_heads          = encoder_num_heads,
            ff_dim             = encoder_ff_dim,
            dropout            = decoder_dropout,
            attn_dropout       = decoder_dropout,
            use_alibi          = True,
            use_bigbird_sparse = True,
            block_size         = block_size,
            num_random_blocks  = num_rand_blocks
        )
        
        # Decoder
        self.decoder = Decoder(
            num_layers       = decoder_num_layers,
            embed_dim        = embed_dim,
            num_heads        = decoder_num_heads,
            ff_dim           = decoder_ff_dim,
            dropout          = decoder_dropout,
            attn_dropout     = decoder_dropout,
            use_alibi        = decoder_use_alibi,
            use_moe          = decoder_use_moe,
            num_experts      = decoder_num_experts,
            moe_top_k        = decoder_moe_top_k
        )
        
        # Embeddings
        self.decoder_embed         = nn.Embedding(vocab_size, embed_dim)
        self.decoder_embed_dropout = nn.Dropout(decoder_dropout)
        
        # LM Head
        self.lm_head = nn.Linear(embed_dim, vocab_size, bias=False)
        
        # Weight tying
        if tie_weights:
            self.lm_head.weight = self.decoder_embed.weight
    
    def forward(
        self,
        encoder_input_ids,
        decoder_input_ids,
        labels                 = None,
        encoder_attention_mask = None,
        decoder_attention_mask = None,
        encoder_hidden_states  = None,
        past_key_values        = None,
        use_cache              = False
    ):
        """
        Forward pass
        
        Args:
            encoder_input_ids:      (B, src_len)
            decoder_input_ids:      (B, tgt_len)
            labels:                 (B, tgt_len) for loss computation
            encoder_attention_mask: (B, src_len)
            decoder_attention_mask: (B, tgt_len)
            encoder_hidden_states:  pre-computed encoder output
            past_key_values:        KV cache for inference
            use_cache:              whether to return updated cache
        
        Returns:
            dict with logits, loss, moe_loss, past_key_values
        """
        # Encode (or use cached)
        if encoder_hidden_states is None:
            encoder_hidden_states = self.encoder(
                encoder_input_ids,
                attention_mask = encoder_attention_mask
            )
        
        # Embed decoder inputs
        decoder_embeds = self.decoder_embed(decoder_input_ids)
        decoder_embeds = self.decoder_embed_dropout(decoder_embeds)
        
        # Decode
        decoder_output, moe_loss, new_cache = self.decoder(
            hidden_states          = decoder_embeds,
            encoder_hidden_states  = encoder_hidden_states,
            attention_mask         = decoder_attention_mask,
            encoder_attention_mask = encoder_attention_mask,
            past_key_values        = past_key_values,
            use_cache              = use_cache
        )
        
        # Project to vocab
        logits = self.lm_head(decoder_output)
        
        # Compute loss
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss     = loss_fct(logits.view(-1, self.vocab_size), labels.view(-1))
            if moe_loss is not None:
                loss = loss + moe_loss
        
        outputs = {
            "logits":           logits,
            "loss":             loss,
            "moe_loss":         moe_loss,
            "encoder_hidden":   encoder_hidden_states
        }
        
        if use_cache:
            outputs["past_key_values"] = new_cache
        
        return outputs
    
    @torch.no_grad()
    def generate(
        self,
        encoder_input_ids,
        encoder_attention_mask = None,
        max_length             = 100,
        temperature            = 1.0,
        top_k                  = 50,
        top_p                  = 0.9,
        bos_token_id           = 1,
        eos_token_id           = 2,
        pad_token_id           = 0
    ):
        """
        Autoregressive generation with KV-cache
        
        Args:
            encoder_input_ids:      (B, src_len)
            encoder_attention_mask: (B, src_len)
            max_length:             max tokens to generate
            temperature:            sampling temperature
            top_k:                  top-k sampling
            top_p:                  nucleus sampling
            bos_token_id:           start token
            eos_token_id:           end token
            pad_token_id:           pad token
        
        Returns:
            generated_ids: (B, generated_len)
        """
        self.eval()
        device = encoder_input_ids.device
        batch  = encoder_input_ids.size(0)
        
        # Encode once
        encoder_hidden = self.encoder(
            encoder_input_ids,
            attention_mask = encoder_attention_mask
        )
        
        # Start with BOS
        generated = torch.full((batch, 1), bos_token_id, dtype=torch.long, device=device)
        finished  = torch.zeros(batch, dtype=torch.bool, device=device)
        cache     = None
        
        for _ in range(max_length):
            # Only feed last token if using cache
            if cache is not None:
                decoder_input = generated[:, -1:]
            else:
                decoder_input = generated
            
            outputs = self.forward(
                encoder_input_ids      = None,
                decoder_input_ids      = decoder_input,
                encoder_hidden_states  = encoder_hidden,
                encoder_attention_mask = encoder_attention_mask,
                past_key_values        = cache,
                use_cache              = True
            )
            
            logits = outputs["logits"][:, -1, :]
            cache  = outputs["past_key_values"]
            
            # Temperature
            if temperature != 1.0:
                logits = logits / temperature
            
            # Top-K
            if top_k > 0:
                indices_to_remove        = logits < torch.topk(logits, top_k)[0][..., -1, None]
                logits[indices_to_remove] = float('-inf')
            
            # Top-P (nucleus)
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs              = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                sorted_indices_to_remove                     = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:]            = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0]             = 0
                
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = float('-inf')
            
            # Sample
            probs     = F.softmax(logits, dim=-1)
            next_token= torch.multinomial(probs, num_samples=1)
            
            # Handle finished sequences
            next_token[finished] = pad_token_id
            
            generated = torch.cat([generated, next_token], dim=1)
            
            # Check EOS
            finished = finished | (next_token.squeeze(-1) == eos_token_id)
            if finished.all():
                break
        
        return generated
    
    def get_param_stats(self):
        enc_train  = sum(p.numel() for p in self.encoder.parameters() if p.requires_grad)
        enc_frozen = sum(p.numel() for p in self.encoder.parameters() if not p.requires_grad)
        dec_train  = sum(p.numel() for p in self.decoder.parameters() if p.requires_grad)
        emb_train  = sum(p.numel() for p in self.decoder_embed.parameters() if p.requires_grad)
        head_train = sum(p.numel() for p in self.lm_head.parameters() if p.requires_grad)
        
        return {
            "encoder_trainable": enc_train,
            "encoder_frozen":    enc_frozen,
            "decoder_trainable": dec_train,
            "embed_trainable":   emb_train,
            "head_trainable":    head_train,
            "total_trainable":   enc_train + dec_train + emb_train + head_train,
            "total_frozen":      enc_frozen
        }
    
    def freeze_encoder(self):
        for param in self.encoder.parameters():
            param.requires_grad = False
    
    def unfreeze_encoder(self):
        for param in self.encoder.parameters():
            param.requires_grad = True
    
    def save(self, save_path):
        """Save model state"""
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        torch.save({
            "encoder":       self.encoder.state_dict(),
            "decoder":       self.decoder.state_dict(),
            "decoder_embed": self.decoder_embed.state_dict(),
            "lm_head":       self.lm_head.state_dict()
        }, path)
        print(f"Saved to {path}")
    
    def load(self, checkpoint_path, strict=False):
        """Load model state"""
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        
        self.encoder.load_state_dict(ckpt["encoder"], strict=strict)
        self.decoder.load_state_dict(ckpt["decoder"], strict=strict)
        self.decoder_embed.load_state_dict(ckpt["decoder_embed"], strict=strict)
        self.lm_head.load_state_dict(ckpt["lm_head"], strict=strict)
        
        print(f"Loaded from {checkpoint_path}")
    
    @classmethod
    def from_pretrained(cls, checkpoint_dir, device="cpu"):
        """Load model from build_model.py output"""
        path = Path(checkpoint_dir)
        
        # Load config
        with open(path / "config.json", "r") as f:
            config = json.load(f)
        
        # Create model
        model = cls(
            embed_dim           = config["embed_dim"],
            encoder_num_layers  = config["encoder_num_layers"],
            encoder_num_heads   = config["encoder_num_heads"],
            encoder_ff_dim      = config["encoder_ff_dim"],
            decoder_num_layers  = config["decoder_num_layers"],
            decoder_num_heads   = config["decoder_num_heads"],
            decoder_ff_dim      = config["decoder_ff_dim"],
            decoder_dropout     = config["decoder_dropout"],
            decoder_use_alibi   = config["decoder_use_alibi"],
            decoder_use_moe     = config["decoder_use_moe"],
            decoder_num_experts = config.get("decoder_num_experts", 8),
            decoder_moe_top_k   = config.get("decoder_moe_top_k", 2),
            vocab_size          = config["vocab_size"],
            tie_weights         = config["tie_weights"],
            block_size          = config.get("block_size", 64),
            num_rand_blocks     = config.get("num_rand_blocks", 3)
        )
        
        # Load weights
        model.load(path / "pytorch_model.bin")
        model = model.to(device)
        
        print(f"Loaded GeneT5 from {path}")
        stats = model.get_param_stats()
        print(f"  Total params: {stats['total_trainable'] + stats['total_frozen']:,}")
        
        return model