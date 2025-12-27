import argparse
from lib import dataset as tk


parser = argparse.ArgumentParser(
    description="Tokenise WormBase FASTA/GFF annotations.")
parser.add_argument("gff3", type=str, 
    help="Path to GFF3 file.")
parser.add_argument("fa", type=str, 
    help="Path to FASTA file.")
parser.add_argument("output", type=str, 
    help="Output tokenized file.")
# default is 6mer based on other popular tokenizer
parser.add_argument("--kmer", type=int, default=6, 
    help="K-mer size [%(default)i].")
# default is 1stride based on other popular tokenizer
parser.add_argument("--stride", type=int, default=1, 
    help="Stride [%(default)i].")
arg = parser.parse_args()



# Load dataset
print(f"Loading dataset...")
ds = tk.WormBaseDataset(arg.gff3, [arg.fa])

# Print statistics
print(f"\nDataset Statistics:")
print(f"  Introns:      {len(ds.introns):,}")
print(f"  Proteins:     {len(ds.proteins):,}")
print(f"  Transcripts:  {len(ds.transcripts):,}")
