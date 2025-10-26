
import argparse
from lib import util


parser = argparse.ArgumentParser(
    description="Train GloVe word embeddings on a text corpus.")
parser.add_argument("corpus", type=str,
    help="Path to input corpus text file.")
parser.add_argument("save_file", type=str,
    help="Output filename prefix for saving the trained vectors.")
parser.add_argument("--vc", type=int, default=2,
    help="Minimum word frequency to include in vocabulary [%(default)s].")
parser.add_argument("--vs", type=int, default=100,
    help="Dimensionality of word embeddings [%(default)s].")
parser.add_argument("--mi", type=int, default=15,
    help="Number of maximum training iterations [%(default)s].")
parser.add_argument("--ws", type=int, default=15,
    help="Context window size (smaller=syntax, larger=semantics) [%(default)s].")
parser.add_argument("--m", type=float, default=4.0,
    help="Memory limit in GB for cooccurrence and shuffle [%(default)s].")
parser.add_argument("--nt", type=int, default=8,
    help="Number of threads for parallel training [%(default)s].")
parser.add_argument("--xm", type=int, default=10,
    help="Cutoff in weighting function [%(default)s].")
parser.add_argument("--bn", type=int, default=2, choices=[0, 1, 2],
    help="Save output in binary format (0=text, 1=binary, 2=both) [%(default)s].")
parser.add_argument("--vb", type=int, default=2, choices=[0, 1, 2],
    help="Verbosity level [%(default)s].")
parser.add_argument("--dr", type=str, default="build",
    help="Dir for GloVe Model [%(default)s].")
parser.add_argument("--clean", action="store_true",
    help="Clean or not")

args = parser.parse_args()
    

vocab_file              = "vocab.txt"
cooccurrence_file       = "cooccurrence.bin"
cooccurrence_shuf_file  = "cooccurrence.shuf.bin"

# build up vocabulary
util.build_vocab(
    corpus_path=    args.corpus,
    vocab_file=     vocab_file,
    glove_dir=      args.dr,
    min_count=      args.vc,
    verbose=        args.vb
)

# build cooccureence matrix
util.build_cooccurrence(
    corpus_path=    args.corpus,
    vocab_file=     vocab_file,
    cooccur_file=   cooccurrence_file,
    glove_dir=      args.dr,
    memory=         args.m,
    window_size=    args.ws,
    verbose=        args.vb
)

# Shuffle cooccurrence
util.shuffle_cooccurrence(
    cooccur_file=       cooccurrence_file,
    cooccur_shuf_file=  cooccurrence_shuf_file,
    glove_dir=          args.dr,
    memory=             args.m,
    verbose=            args.vb
)

# Train GloVe
util.train_glove(
    cooccur_shuf_file=  cooccurrence_shuf_file,
    vocab_file=         vocab_file,
    save_file=          args.save_file,
    glove_dir=          args.dr,
    vector_size=        args.vs,
    max_iter=           args.mi,
    num_threads=        args.nt,
    x_max=              args.xm,
    binary=             args.bn,
    verbose=            args.vb
)

# Cleanup
if args.clean:
    util.cleanup(vocab_file, cooccurrence_file, cooccurrence_shuf_file)