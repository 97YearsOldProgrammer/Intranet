
import argparse
import sys
from lib import tokenizer as tk

parser = argparse.ArgumentParser(
    description="Tokenise FASTA/GFF annotations into a corpus for GloVe.")
parser.add_argument("input", type=str,
    help="Path to directory containing FASTA/GFF files.")
parser.add_argument("output", type=str,
    help="Destination file for the generated corpus (plain text).")
parser.add_argument("--kmer", type=int, default=8,
    help="Size of the sliding k-mer window used for tokenisation [%(default)i].")
parser.add_argument("--stride", type=int, default=3,
    help="Stride of the sliding window used in the tokenizer [%(default)i].")
parser.add_argument("--unk", default="<UNK>", type=str,
    help="Unknown token for unseen k-mers [%(default)s].")
parser.add_argument("--utr", action="store_true",
    help="Include UTR regions (5' and 3' UTRs) in transcripts.")

arg = parser.parse_args()

# Determine which feature types to include
if arg.utr:
    feature_filter = {"exon", "intron", "three_prime_utr", "five_prime_utr"}
else:
    feature_filter = {"exon", "intron"}

# Find files
fastas, gffs = tk.find_files(arg.input)
print(f"Found {len(fastas)} FASTA files and {len(gffs)} GFF files", flush=True)

# Load all sequences
sequences = {}
for fasta in fastas:
    for name, seq in tk.read_fasta(fasta):
        if name is None:
            continue
        sequences[name] = seq
print(f"Finished parsing FASTA: {len(sequences)} sequences loaded", flush=True)

# Parse all GFF files
features = []
for gff in gffs:
    features.extend(tk.parse_gff(gff))

grouped = tk.group_features(features, feature_filter)
print(f"Grouped into {len(grouped)} transcripts", flush=True)

# Build transcripts
transcripts = tk.build_transcript(grouped, sequences)
print(f"Built {len(transcripts)} transcripts", flush=True)

if not transcripts:
    print("Error: No transcripts were built; corpus is empty.", flush=True)
    sys.exit(1)

# Create tokenizer
vocabulary = tk.apkmer(arg.kmer)
print(f"Finished vocabulary generation: {len(vocabulary)} k-mers", flush=True)

tokenizer = tk.KmerTokenizer(
    k=arg.kmer,
    stride=arg.stride,
    vocabulary=vocabulary,
    unk_token=arg.unk
)

# Tokenize transcripts
tokenized = tk.tokenize_transcripts(transcripts, tokenizer)

# Write corpus
tk.write_tokenized_corpus(arg.output, tokenized)
print(f"Corpus successfully written to {arg.output}", flush=True)