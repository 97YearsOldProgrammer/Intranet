
import argparse
import sys
from pathlib import Path

from lib.build_model import build_gt5

parser = argparse.ArgumentParser(description="Initialize GeneT5 weights from DNABERT-2 and save to disk.")
parser.add_argument("--save_dir", type=str, default="./checkpoints/genet5_init", 
    help="Directory to save the initialized model weights and config.")
parser.add_argument("--dnabert_path", type=str, default="zhihan1996/DNABERT-2-117M",
    help="HuggingFace model ID or local path to DNABERT-2.")
parser.add_argument("--vocab_size", type=int, default=4096, 
    help="Vocabulary size.")
parser.add_argument("--block_size", type=int, default=64, 
    help="Block size for BigBird sparse attention.")
parser.add_argument("--decoder_layers", type=int, default=None, 
    help="Number of decoder layers. Defaults to matching encoder (12).")
parser.add_argument("--decoder_heads", type=int, default=None,
    help="Number of decoder heads. Defaults to matching encoder.")
parser.add_argument("--tie_weights", action="store_true", default=True,
    help="Tie input/output embeddings.")
parser.add_argument("--no_tie_weights", action="store_false", dest="tie_weights",
    help="Do not tie input/output embeddings.")
parser.add_argument("--use_moe", action="store_true", 
    help="Enable Mixture of Experts in Decoder.")
parser.add_argument("--num_experts", type=int, default=8, 
    help="Number of experts for MoE.")
parser.add_argument("--moe_top_k", type=int, default=2, 
    help="Top-K routing for MoE.")
args = parser.parse_args()

# Visual separator
print(f"\n{' GeneT5 Initialization ':=^60}")
    
# Run the build function
saved_path = build_gt5(
        dnabert_model_name  = args.dnabert_path,
        save_dir            = args.save_dir,
        block_size          = args.block_size,
        decoder_num_layers  = args.decoder_layers,
        decoder_num_heads   = args.decoder_heads,
        decoder_use_moe     = args.use_moe,
        decoder_num_experts = args.num_experts,
        decoder_moe_top_k   = args.moe_top_k,
        vocab_size          = args.vocab_size,
        tie_weights         = args.tie_weights
    )
        
print(f"\nSUCCESS: Model initialized and saved to:")
print(f"  -> {saved_path}")
print(f"  -> {saved_path}/config.json")
print(f"  -> {saved_path}/pytorch_model.bin")
