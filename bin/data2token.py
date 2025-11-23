"""
this is data2token for small gene set
"""

import argparse
import sys
from lib import dataset as tk

parser = argparse.ArgumentParser(
    description="Tokenise FASTA/GFF annotations into a corpus for GloVe.")
parser.add_argument("input", type=str,
    help="Path to directory containing FASTA/GFF files.")
parser.add_argument("tk_op", type=str,
    help="Destination file for the generated corpus (plain text).")
parser.add_argument("labels_op", type=str,
    help="Optional destination file for the generated label sequences.")
# default is 6mer based on other popular tokenizer
parser.add_argument("--kmer", type=int, default=6,
    help="Size of the sliding k-mer window used for tokenisation [%(default)i].")
# default is 1stride based on other popular tokenizer
parser.add_argument("--stride", type=int, default=1,
    help="Stride of the sliding window used in the tokenizer [%(default)i].")
parser.add_argument("--utr", action="store_true",
    help="Include UTR regions (5' and 3' UTRs) in transcripts.")
parser.add_argument("-io","--include_orphan", action="store_true",
    help="Include single intron that exists in RNA Sequencing Data")
arg = parser.parse_args()



# Determine which feature types to include
if arg.utr:
    feature_filter = {"exon", "intron", "three_prime_utr", "five_prime_utr"}
else:
    feature_filter = {"exon", "intron"}

# Prepare label mapping for the requested features.
feature_label_map = {
    feature: tk.DEFAULT_FEATURE_LABELS[feature]
    for feature in feature_filter
    if feature in tk.DEFAULT_FEATURE_LABELS
}

next_label = max(feature_label_map.values(), default=-1) + 1
for feature in sorted(feature_filter):
    if feature not in feature_label_map:
        feature_label_map[feature] = next_label
        next_label += 1

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
features        = []
total_orphans   = 0

for gff in gffs:
    parsed_features, orphan_count = tk.parse_gff(gff, adopt_orphan=arg.include_orphan)
    features.extend(parsed_features)
    total_orphans += orphan_count

print(f"Parsed {len(features)} total features", flush=True)
if arg.include_orphan:
    print(f"  Including {total_orphans} orphan introns", flush=True)

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
    vocabulary=vocabulary
)

# Tokenize transcripts
tokenized, labels = tk.tokenize_transcripts(transcripts, tokenizer, feature_label_map=feature_label_map)

# Write corpus
tk.write_tokenized_corpus(arg.tk_op, tokenized)
print(f"Corpus successfully written to {arg.tk_op}", flush=True)

# Write label output
labels_output = arg.labels_op
tk.write_label_sequences(labels_output, labels)
print(f"Label sequences written to {labels_output}", flush=True)