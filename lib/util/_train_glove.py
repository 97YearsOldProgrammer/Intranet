import subprocess
import os
import sys



def build_vocab(corpus_path, vocab_file, glove_dir, min_count, verbose):
    
    cmd = [
        f"{glove_dir}/vocab_count",
        "-min-count", str(min_count),
        "-verbose", str(verbose)
    ]
    
    with open(corpus_path, 'r') as infile, open(vocab_file, 'w') as outfile:
        subprocess.run(cmd, stdin=infile, stdout=outfile, check=True)

def build_cooccurrence(corpus_path, vocab_file, cooccur_file, glove_dir, 
                       memory, window_size, verbose):
    
    cmd = [
        f"{glove_dir}/cooccur",
        "-memory", str(memory),
        "-vocab-file", vocab_file,
        "-verbose", str(verbose),
        "-window-size", str(window_size)
    ]
    
    with open(corpus_path, 'r') as infile, open(cooccur_file, 'wb') as outfile:
        subprocess.run(cmd, stdin=infile, stdout=outfile, check=True)

def shuffle_cooccurrence(cooccur_file, cooccur_shuf_file, glove_dir, 
                         memory, verbose):
    
    cmd = [
        f"{glove_dir}/shuffle",
        "-memory", str(memory),
        "-verbose", str(verbose)
    ]

    with open(cooccur_file, 'rb') as infile, open(cooccur_shuf_file, 'wb') as outfile:
        subprocess.run(cmd, stdin=infile, stdout=outfile, check=True)

def train_glove(cooccur_shuf_file, vocab_file, save_file, glove_dir,
                vector_size, max_iter, num_threads, x_max, 
                binary, verbose):
    
    cmd = [
        f"{glove_dir}/glove",
        "-save-file", save_file,
        "-threads", str(num_threads),
        "-input-file", cooccur_shuf_file,
        "-x-max", str(x_max),
        "-iter", str(max_iter),
        "-vector-size", str(vector_size),
        "-binary", str(binary),
        "-vocab-file", vocab_file,
        "-verbose", str(verbose)
    ]

    subprocess.run(cmd, check=True)

def cleanup(vocab_file, cooccur_file, cooccur_shuf_file):

    for f in [vocab_file, cooccur_file, cooccur_shuf_file]:
        if os.path.exists(f):
            os.remove(f)
            print(f"  Removed: {f}")