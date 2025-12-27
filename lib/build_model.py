
import torch
import torch.nn as nn
import json

from transformers import AutoModel, AutoTokenizer
from pathlib import Path

from lib.blocks import Encoder, Decoder


def build_gt5(
    dnabert_model_name  = "zhihan1996/DNABERT-2-117M",
    save_dir            = "./checkpoints/genet5_init",
    block_size          = 64,
    num_rand_blocks     = 3,
    decoder_num_layers  = None,
    decoder_num_heads   = None,
    decoder_ff_dim      = None,
    decoder_dropout     = 0.1,
    decoder_use_alibi   = True,
    decoder_use_moe     = False,
    decoder_num_experts = 8,
    decoder_moe_top_k   = 2,
    vocab_size          = 4096,
    tie_weights         = True
):
    """
    Build GeneT5 from DNABERT-2 and save clean checkpoint.
    
    Weight Transfer:
        Encoder Self-Attention  -> COPY from DNABERT-2
        Decoder Self-Attention  -> COPY from Encoder
        Cross-Attention         -> RANDOM INIT
        Layer Norms             -> COPY
        Embeddings              -> RANDOM INIT
        Output Head             -> TIED or RANDOM INIT
    """
    print("=" * 60)
    print("Building GeneT5 from DNABERT-2")
    print("=" * 60)
    
    # Load Pre-trained Model
    print(f"\n[1] Loading DNABERT-2: {dnabert_model_name}")
    tokenizer      = AutoTokenizer.from_pretrained(dnabert_model_name, trust_remote_code=True)
    original_model = AutoModel.from_pretrained(dnabert_model_name, trust_remote_code=True)
    dna_config     = original_model.config
    
    # Create Symmetry Dimension
    if decoder_num_layers is None:
        decoder_num_layers = dna_config.num_hidden_layers
    if decoder_num_heads is None:
        decoder_num_heads = dna_config.num_attention_heads
    if decoder_ff_dim is None:
        decoder_ff_dim = dna_config.intermediate_size
    
    print(f"    DNABERT-2 Config:")
    print(f"      hidden_size:     {dna_config.hidden_size}")
    print(f"      num_layers:      {dna_config.num_hidden_layers}")
    print(f"      num_heads:       {dna_config.num_attention_heads}")
    print(f"      intermediate:    {dna_config.intermediate_size}")
    print(f"    Decoder Config:")
    print(f"      num_layers:      {decoder_num_layers}")
    print(f"      num_heads:       {decoder_num_heads}")
    print(f"      ff_dim:          {decoder_ff_dim}")
    print(f"      use_alibi:       {decoder_use_alibi}")
    print(f"      use_moe:         {decoder_use_moe}")
    
    # Build Encoder
    print(f"\n[2] Building BigBird Encoder (block_size={block_size})")
    encoder = Encoder(
        num_layers         = dna_config.num_hidden_layers,
        embed_dim          = dna_config.hidden_size,
        num_heads          = dna_config.num_attention_heads,
        ff_dim             = dna_config.intermediate_size,
        dropout            = dna_config.hidden_dropout_prob,
        attn_dropout       = dna_config.attention_probs_dropout_prob,
        use_alibi          = True,
        use_bigbird_sparse = True,
        block_size         = block_size,
        num_random_blocks  = num_rand_blocks
    )
    
    # Transfer Encoder Weights
    print("\n[3] Transferring Encoder Weights from DNABERT-2")
    original_layers = original_model.bert.encoder.layer
    
    for idx, (orig, new) in enumerate(zip(original_layers, encoder.layers)):
        orig_attn = orig.attention.self
        new_attn  = new.self_attn
        
        # Q, K, V, O
        new_attn.q.weight.data.copy_(orig_attn.query.weight.data)
        new_attn.k.weight.data.copy_(orig_attn.key.weight.data)
        new_attn.v.weight.data.copy_(orig_attn.value.weight.data)
        new_attn.o.weight.data.copy_(orig.attention.output.dense.weight.data)
        
        # Layer norm
        new.norm1.weight.data.copy_(orig.attention.output.LayerNorm.weight.data)
        
        # FFN
        new.ff.wi_0.weight.data.copy_(orig.intermediate.dense.weight.data)
        new.ff.wi_1.weight.data.copy_(orig.intermediate.dense.weight.data)
        new.ff.wo.weight.data.copy_(orig.output.dense.weight.data)
        new.norm2.weight.data.copy_(orig.output.LayerNorm.weight.data)
    
    encoder.final_norm.weight.data.copy_(original_model.bert.encoder.LayerNorm.weight.data)
    print("    ✓ Encoder weights copied")
    
    # Build Decoder
    print(f"\n[4] Building Decoder (layers={decoder_num_layers}, moe={decoder_use_moe})")
    decoder = Decoder(
        num_layers       = decoder_num_layers,
        embed_dim        = dna_config.hidden_size,
        num_heads        = decoder_num_heads,
        ff_dim           = decoder_ff_dim,
        dropout          = decoder_dropout,
        attn_dropout     = decoder_dropout,
        use_alibi        = decoder_use_alibi,
        use_moe          = decoder_use_moe,
        num_experts      = decoder_num_experts,
        moe_top_k        = decoder_moe_top_k
    )
    
    # Transfer Decoder Self-Attention from Encoder
    print("\n[5] Transferring Decoder Self-Attention from Encoder")
    num_copy = min(decoder_num_layers, len(encoder.layers))
    
    for idx in range(num_copy):
        enc_attn = encoder.layers[idx].self_attn
        dec_attn = decoder.layers[idx].self_attn
        
        dec_attn.q.weight.data.copy_(enc_attn.q.weight.data)
        dec_attn.k.weight.data.copy_(enc_attn.k.weight.data)
        dec_attn.v.weight.data.copy_(enc_attn.v.weight.data)
        dec_attn.o.weight.data.copy_(enc_attn.o.weight.data)
        
        decoder.layers[idx].norm1.weight.data.copy_(encoder.layers[idx].norm1.weight.data)
    
    print(f"    ✓ Copied {num_copy} layers")
    
    # Random Init Cross-Attention
    print("\n[6] Random Init Cross-Attention")
    for layer in decoder.layers:
        nn.init.xavier_uniform_(layer.cross_attn.q.weight)
        nn.init.xavier_uniform_(layer.cross_attn.k.weight)
        nn.init.xavier_uniform_(layer.cross_attn.v.weight)
        nn.init.xavier_uniform_(layer.cross_attn.o.weight)
        nn.init.ones_(layer.norm2.weight)
    print("    ✓ Cross-attention initialized")
    
    # Random Init Decoder FFN (or MoE)
    print("\n[7] Random Init Decoder FFN")
    for layer in decoder.layers:
        if decoder_use_moe:
            # MoE has different structure
            if hasattr(layer.ff, 'experts'):
                for expert in layer.ff.experts:
                    if hasattr(expert, 'wi_0'):
                        nn.init.xavier_uniform_(expert.wi_0.weight)
                        nn.init.xavier_uniform_(expert.wi_1.weight)
                        nn.init.xavier_uniform_(expert.wo.weight)
            if hasattr(layer.ff, 'router'):
                nn.init.xavier_uniform_(layer.ff.router.weight)
        else:
            if hasattr(layer.ff, 'wi_0'):
                nn.init.xavier_uniform_(layer.ff.wi_0.weight)
                nn.init.xavier_uniform_(layer.ff.wi_1.weight)
                nn.init.xavier_uniform_(layer.ff.wo.weight)
        nn.init.ones_(layer.norm3.weight)
    nn.init.ones_(decoder.final_norm.weight)
    print("    ✓ FFN initialized")
    
    # Build Embeddings and LM Head
    print("\n[8] Building Embeddings and LM Head")
    decoder_embed = nn.Embedding(vocab_size, dna_config.hidden_size)
    lm_head       = nn.Linear(dna_config.hidden_size, vocab_size, bias=False)
    
    nn.init.normal_(decoder_embed.weight, mean=0.0, std=0.02)
    
    if tie_weights:
        lm_head.weight = decoder_embed.weight
        print("    ✓ Weights tied (embed = lm_head)")
    else:
        nn.init.xavier_uniform_(lm_head.weight)
        print("    ✓ Separate weights initialized")
    
    # Save Checkpoint
    print(f"\n[9] Saving Checkpoint to {save_dir}")
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    # Config
    config = {
        "embed_dim":          dna_config.hidden_size,
        "encoder_num_layers": dna_config.num_hidden_layers,
        "encoder_num_heads":  dna_config.num_attention_heads,
        "encoder_ff_dim":     dna_config.intermediate_size,
        "decoder_num_layers": decoder_num_layers,
        "decoder_num_heads":  decoder_num_heads,
        "decoder_ff_dim":     decoder_ff_dim,
        "decoder_dropout":    decoder_dropout,
        "decoder_use_alibi":  decoder_use_alibi,
        "decoder_use_moe":    decoder_use_moe,
        "decoder_num_experts":decoder_num_experts,
        "decoder_moe_top_k":  decoder_moe_top_k,
        "vocab_size":         vocab_size,
        "tie_weights":        tie_weights,
        "block_size":         block_size,
        "num_rand_blocks":    num_rand_blocks
    }
    
    # Save config.json
    with open(save_path / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    print("    ✓ config.json saved")
    
    # Save weights
    checkpoint = {
        "encoder":       encoder.state_dict(),
        "decoder":       decoder.state_dict(),
        "decoder_embed": decoder_embed.state_dict(),
        "lm_head":       lm_head.state_dict()
    }
    torch.save(checkpoint, save_path / "pytorch_model.bin")
    print("    ✓ pytorch_model.bin saved")
    
    # Save tokenizer
    tokenizer.save_pretrained(save_path)
    print("    ✓ tokenizer saved")
    
    # Cleanup
    del original_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Stats
    total_params = sum(p.numel() for p in encoder.parameters())
    total_params += sum(p.numel() for p in decoder.parameters())
    total_params += sum(p.numel() for p in decoder_embed.parameters())
    total_params += sum(p.numel() for p in lm_head.parameters())
    
    print("\n" + "=" * 60)
    print(f"✓ GeneT5 Built Successfully")
    print(f"  Total Parameters: {total_params:,}")
    print(f"  Saved to: {save_path}")
    print("=" * 60)
    
    return str(save_path)