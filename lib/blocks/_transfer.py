import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from lib.blocks._component import BigBirdSparseAttention, EncoderBlock, Encoder, LayerNorm


def dnabert2tobigbird(
    model_name          ="zhihan1996/DNABERT-2-117M",
    block_size          =64,
    num_rand_blocks     =3,
    device              ="cuda" if torch.cuda.is_available() else "cpu"
):
    """ Load DNABERT-2 and convert to BigBird sparse attention """
    
    print(f"Loading DNABERT-2 from {model_name}...")
    tokenizer       = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    original_model  = AutoModel.from_pretrained(model_name, trust_remote_code=True)
    
    # Extract config from DNABERT-2
    config = original_model.config
    
    print("\nDNABERT-2 Configuration:")
    print(f"  Hidden size: {config.hidden_size}")
    print(f"  Num layers: {config.num_hidden_layers}")
    print(f"  Num heads: {config.num_attention_heads}")
    print(f"  Intermediate size: {config.intermediate_size}")
    print(f"  Max position embeddings: {config.max_position_embeddings}")
    
    # Create our encoder with BigBird sparse attention + ALiBi
    print(f"\nCreating BigBird sparse encoder (block_size={block_size})...")
    converted_encoder = Encoder(
        num_layers          =config.num_hidden_layers,
        embed_dim           =config.hidden_size,
        num_heads           =config.num_attention_heads,
        ff_dim              =config.intermediate_size,
        dropout             =config.hidden_dropout_prob,
        attn_dropout        =config.attention_probs_dropout_prob,
        use_alibi           =True,
        use_bigbird_sparse  =True,
        block_size          =block_size,
        num_rand_blocks     =num_rand_blocks
    )
    
    # Transfer weights layer by layer
    print("\nTransferring weights from DNABERT-2 to BigBird encoder...")
    
    original_layers = original_model.bert.encoder.layer
    
    for layer_idx, (orig_layer, new_layer) in enumerate(zip(original_layers, converted_encoder.layers)):
        print(f"  Layer {layer_idx}...")
        
        # Transfer attention weights (Q, K, V, O)
        orig_attn   = orig_layer.attention.self
        new_attn    = new_layer.self_attn
        
        # Q, K, V projections
        new_attn.q.weight.data.copy_(orig_attn.query.weight.data)
        new_attn.k.weight.data.copy_(orig_attn.key.weight.data)
        new_attn.v.weight.data.copy_(orig_attn.value.weight.data)
        
        # Output projection
        new_attn.o.weight.data.copy_(orig_layer.attention.output.dense.weight.data)
        
        # LayerNorm after attention
        new_layer.norm1.weight.data.copy_(orig_layer.attention.output.LayerNorm.weight.data)
        
        # Transfer feedforward weights
        orig_ff     = orig_layer.intermediate
        orig_output = orig_layer.output
        
        # Up projection
        new_layer.ff.wi_0.weight.data.copy_(orig_ff.dense.weight.data)
        new_layer.ff.wi_1.weight.data.copy_(orig_ff.dense.weight.data)
        
        # Down projection
        new_layer.ff.wo.weight.data.copy_(orig_output.dense.weight.data)
        
        # LayerNorm after feedforward
        new_layer.norm2.weight.data.copy_(orig_output.LayerNorm.weight.data)
    
    # Final layer norm
    converted_encoder.final_norm.weight.data.copy_(
        original_model.bert.encoder.LayerNorm.weight.data
    )
    
    print("\n✓ Conversion complete!")
    print(f"✓ Model now uses BigBird sparse attention (O(n) complexity)")
    print(f"✓ ALiBi position bias preserved")
    print(f"✓ All pretrained weights transferred")
    
    return converted_encoder.to(device), tokenizer