"""
data to token for WormBase gene set
"""

import argparse
from lib import dataset as tk

parser = argparse.ArgumentParser(
    description="Tokenise WormBase FASTA/GFF annotations into a corpus for GloVe.")
parser.add_argument("gff3", type=str,
    help="Path to the GFF3 file.")
parser.add_argument("fa", type=str,
    help="Path to the FASTA file.")
parser.add_argument("tk_op", type=str,
    help="Destination file for the generated corpus (plain text).")
parser.add_argument("labels_op", type=str,
    help="Destination file for the generated label sequences.")
# default is 6mer based on other popular tokenizer
parser.add_argument("--kmer", type=int, default=6,
    help="Size of the sliding k-mer window used for tokenisation [%(default)i].")
# default is 1stride based on other popular tokenizer
parser.add_argument("--stride", type=int, default=1,
    help="Stride of the sliding window used in the tokenizer [%(default)i].")
arg = parser.parse_args()



# Load dataset using WormBaseDataset class
print(f"Loading WormBase dataset from {arg.gff3} and {arg.fa}", flush=True)
ds = tk.WormBaseDataset(arg.gff3, [arg.fa])

print(f"Dataset loaded successfully", flush=True)
print(f"  Forward transcripts: {len(ds.fw_transcripts)}", flush=True)
print(f"  Reverse transcripts: {len(ds.rv_transcripts)}", flush=True)

# Create vocabulary and tokenizer
vocabulary = tk.apkmer(arg.kmer)
print(f"Finished vocabulary generation: {len(vocabulary)} k-mers", flush=True)

tokenizer = tk.KmerTokenizer(
    k=arg.kmer,
    stride=arg.stride,
    vocabulary=vocabulary
)

# Tokenize all transcripts
result                  = ds.tokenize_all(tokenizer)
fw_tokens, fw_labels    = result['forward']
rv_tokens, rv_labels    = result['reverse']

print(f"Tokenized {len(fw_tokens)} forward and {len(rv_tokens)} reverse transcripts", flush=True)

# Combine forward and reverse
all_tokenized   = {**fw_tokens, **rv_tokens}
all_labels      = {**fw_labels, **rv_labels}

# Write corpus
tk.write_tokenized_corpus(arg.tk_op, all_tokenized)
print(f"Corpus successfully written to {arg.tk_op}", flush=True)

# Write label sequences
tk.write_label_sequences(arg.labels_op, all_labels)
print(f"Label sequences written to {arg.labels_op}", flush=True)