from __future__ import annotations

import gzip
import math
import sys
import os
import re
import typing as tp
from typing import List
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader, random_split

from pathlib        import Path
from contextlib     import closing
from dataclasses    import dataclass



def anti(seq):
	comp = str.maketrans('ACGTRYMKBDHVNacgtrymkbdhvn', 'TGCAYRKMVHDBNtgcayrkmvhdbn')
	anti = seq.translate(comp)[::-1]
	return anti



#####################
## UTILITY SECTION ##
#####################


def getfp(filename):
    
	if   filename.endswith('.gz'):
		return gzip.open(filename, 'rt', encoding='ISO-8859-1')
	elif filename == '-':
		return sys.stdin
	return open(filename)

def read_fasta(filename):

	name = None
	seqs = []

	fp = getfp(filename)

	for line in fp:
		line = line.rstrip()
		if line.startswith('>'):
			if len(seqs) > 0:
				seq = ''.join(seqs)
				yield(name, seq)
				name = line[1:].split()[0]
				seqs = []
			else:
				name = line[1:].split()[0]
		else:
			seqs.append(line)
	yield(name, ''.join(seqs))
	fp.close()

def find_files(input_path):

	fa      = re.compile(r'\.(fasta|fa|fna)(\.gz)?$', re.IGNORECASE)
	gff     = re.compile(r'\.(gff|gff3)(\.gz)?$', re.IGNORECASE)
 
	fastas  = []
	gffs    = []
	
	input_path = Path(input_path)
	
	if not input_path.exists():
		raise FileNotFoundError(f"Input path does not exist: {input_path}")
	
	if not input_path.is_dir():
		raise ValueError(f"Input path must be a directory: {input_path}")
	
	for root, dirs, files in os.walk(input_path):
		for file in files:
			filepath = os.path.join(root, file)
			
			if fa.search(file):
				fastas.append(filepath)
			elif gff.search(file):
				gffs.append(filepath)
	
	return fastas, gffs

@dataclass
class Feature:
    """ Parse Single GFF line"""

    seqid:  str
    source: str
    typ:    str
    beg:    int
    end:    int
    score:  tp.Optional[float]
    strand: str
    phase:  tp.Optional[int]
    att:    tp.Dict[str, str]

def parse_att(att):

    attributes = {}

    if not att or att == ".":
        return attributes

    for stuff in att.split(";"):
        stuff = stuff.strip()
        if not stuff:
            continue
        if "=" in stuff:
            key, value = stuff.split("=", 1)
        elif " " in stuff:
            key, value = stuff.split(" ", 1)
        else:
            key, value = stuff, ""
        attributes[key.strip()] = value.strip()

    return attributes

def parse_gff(filename, adopt_orphan: bool=False):
    
    fp          = getfp(filename)
    features    = []
    orphan_count = 0

    with closing(fp):
        for line in fp:

            line = line.strip()
            if not line or line.startswith("#"):
                continue
            line = line.split("\t")
            
            if len(line) == 8:
                line.append(".")
            elif len(line) != 9:
                continue

            seqid, source, typ, beg, end, score, strand, phase, att = line
            score = None if score == "." else float(score)
            phase = None if phase == "." else int(phase)

            att = parse_att(att)
            
            if not att:
                if adopt_orphan and typ.lower() == "intron":
                    orphan_id = f"{source}:{seqid}:{beg}-{end}"
                    att = {
                        "ID": orphan_id,
                        "Parent": orphan_id,
                    }
                    orphan_count += 1
                else:
                    continue

            feature = Feature(
                seqid=  seqid,
                source= source,
                typ=    typ.lower(),
                beg=    int(beg),
                end=    int(end),
                score=  score,
                strand= strand,
                phase=  phase,
                att=    att,
            )
            features.append(feature)

    return features, orphan_count

def choose_parent_id(feature):
    
    if "Parent" in feature.att:
        return feature.att["Parent"].split(",")[0]
    if "ID" in feature.att:
        return feature.att["ID"]
    return feature.seqid

def group_features(features, filter):

    group = {}

    for feature in features:
        if feature.typ not in filter: continue
        parent_id = choose_parent_id(feature)
        group.setdefault(parent_id, []).append(feature)

    return group

def build_line(features, seqid, strand, seq):

    line = []

    # iterate through all feature under a parent ID
    for feature in sorted(features, key=lambda x: x.beg):

        if feature.seqid != seqid:
            raise ValueError(
                f"Transcript {seqid} has exons on multiple sequences "
                f"({seqid} vs {feature.seqid})"
            )
            
        if feature.strand != strand:
            raise ValueError(
                f"Transcript {seqid} mixes strands ({strand} vs {feature.strand})"
            )
        
        word = seq[feature.beg-1 : feature.end]
        if strand == "-":
            word = anti(word)
        line.append((word, feature.typ))

    return line

def build_transcript(grouped, sequences):

    transcripts = {}
    
    for parent_id, features in grouped.items():
        if not features:
            continue
        
        seqid   = features[0].seqid
        strand  = features[0].strand
        
        if seqid not in sequences:
            raise KeyError(
                f"Sequence '{seqid}' referenced in GFF but absent from FASTA"
            )

        seq = sequences[seqid]
        transcripts[parent_id] = build_line(features, seqid, strand, seq)

    return transcripts

def tokenize_transcripts(transcripts, tokenizer, feature_label_map, default_label=-1):

    tokenized = {}
    labels = {}

    for parent_id, segments in transcripts.items():
        token_ids = []
        label_ids = []

        for segment, feature_type in segments:
            segment_token_ids = tokenizer(segment)
            token_ids.extend(segment_token_ids)

            # mapping token one-to-one labels
            label_value = feature_label_map.get(feature_type, default_label)
            label_ids.extend([label_value] * len(segment_token_ids))

        tokenized[parent_id]    = token_ids
        labels[parent_id]       = label_ids

    return tokenized, labels


######################
#### Tokenisation ####
######################


BASE_PAIR   = ("A", "C", "G", "T")
BASE2IDX    = {base: idx for idx, base in enumerate(BASE_PAIR)}

DEFAULT_FEATURE_LABELS = {
    "exon": 0,
    "intron": 1,
    "five_prime_utr": 2,
    "three_prime_utr": 3,
}

def apkmer(k: int):

    if k <= 0:
        raise ValueError("k must be a positive integer")

    if k == 1:
        return list(BASE_PAIR)

    prev_kmers = apkmer(k - 1)
    return [prefix + base for prefix in prev_kmers for base in BASE_PAIR]

@dataclass
class KmerTokenizer:
    """DNA seq to kmer ids by sliding window algo"""

    k           : int
    stride      : int = 1
    vocabulary  : list = None
    unk_token   : str = None

    def __post_init__(self):
        
        # map allkmer with a int
        self.token2id = {token: idx for idx, token in enumerate(self.vocabulary)}

        # map the unknown bps as last element
        if self.unk_token is not None and self.unk_token not in self.token2id:
            self.token2id[self.unk_token] = len(self.token2id)

    def __call__(self, seq):

        seq     = seq.upper()
        tokens  = []

        # sliding window algo
        for t in range(0, max(len(seq) - self.k + 1, 0), self.stride):
            token = seq[t:t+self.k]
            tokens.append(self.token2id.get(token, self.token2id.get(self.unk_token, 0)))
        if not tokens:
            tokens.append(self.token2id.get(self.unk_token, 0))
        return tokens


################
#### Output ####
################


def write_tokenized_corpus(output_path, tokenized):

    with open(output_path, "w", encoding="utf-8") as fp:
        for parent_id in sorted(tokenized):
            token_line = " ".join(str(token_id) for token_id in tokenized[parent_id])
            fp.write(token_line + "\n")

def write_label_sequences(output_path, labels):

    with open(output_path, "w", encoding="utf-8") as fp:
        for parent_id in sorted(labels):
            label_line = " ".join(str(label_id) for label_id in labels[parent_id])
            fp.write(label_line + "\n")


###################
#### Embedding ####
###################


def parse_glove_matrix(vectors):

    embeddings = {}
    
    with open(vectors, 'r') as fp:
        for line in fp:
            parts = line.strip().split()
            if not parts:
                continue
            
            idx     = int(parts[0])
            vector  = torch.tensor([float(x) for x in parts[1:]], dtype=torch.float32)
            embeddings[idx] = vector
    
    return embeddings

def embed_sentence(line, embeddings):

    idxs = [int(x) for x in line.strip().split()]
    
    if not idxs:
        return torch.tensor([])

    embeddings = [embeddings[token_id] for token_id in idxs]
    # stack all embedding vectors
    return torch.stack(embeddings, dim=0)

def embed_whole_corpus(corpus, embeddings):

    with open(corpus, 'r') as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue

            embedded = embed_sentence(line, embeddings)
            yield embedded


###################
##### DataSet #####
###################


class SplicingDataset(Dataset):

    def __init__(self, corpus_file, embeddings, labels_file):
        self.embeddings = embeddings
        self.sentences = []
        self.labels = []
        
        # Load embeddings
        with open(corpus_file, 'r') as fp:
            for line in fp:
                line = line.strip()
                if not line:
                    continue
                tensor = embed_sentence(line, embeddings)  # (L, E)
                self.sentences.append(tensor)
        
        with open(labels_file, 'r') as fp:
            for line in fp:
                line = line.strip()
                if not line:
                    continue

                label_seq = [int(x) for x in line.split()]
                self.labels.append(label_seq)
        
        assert len(self.sentences) == len(self.labels), \
            f"Mismatch: {len(self.sentences)} sequences vs {len(self.labels)} labels"

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        x = self.sentences[idx]         # (L, E)
        y = self.labels[idx]            # list of labels, length L
        length = len(x)                 # L
        return x, y, length


#####################
#### IntraNet NN ####
#####################


class IntraNet(nn.Module):
    """
    CNN → BLSTM hybrid for embedded DNA sequences with TOKEN-LEVEL classification.
    Predicts one label (exon/intron) per token position.
    """

    def __init__(
        self,
        embedding_dim:          int,
        num_classes:            int,
        conv_channels:          List[int]   = [64, 128, 256],
        kernel_sizes:           List[tuple] = [(3, 3), (3, 3), (3, 3)],
        pool_sizes:             List[tuple] = [(2, 2), (2, 2), (2, 2)],
        lstm_hidden:            int = 128,
        lstm_layers:            int = 1,
        dropout:                float = 0.5,
    ):
        
        super().__init__()
        assert len(conv_channels) == len(kernel_sizes) == len(pool_sizes)

        self.embedding_dim = embedding_dim
        self.pool_sizes = pool_sizes

        # Convolutional Layer
        self.convs = nn.ModuleList()
        self.bns   = nn.ModuleList()
        self.pools = nn.ModuleList()

        in_ch = 1

        for out_ch, ksz, psz in zip(conv_channels, kernel_sizes, pool_sizes):
            pad = self._same_pad(ksz)
            self.convs.append(nn.Conv2d(in_ch, out_ch, kernel_size=ksz, padding=pad, bias=False))
            self.bns.append(nn.BatchNorm2d(out_ch))
            self.pools.append(nn.MaxPool2d(kernel_size=psz))
            in_ch = out_ch

        c_last = conv_channels[-1]

        # BLSTM Layer
        self.blstm = nn.LSTM(
            input_size=     c_last,
            hidden_size=    lstm_hidden,
            num_layers=     lstm_layers,
            batch_first=    True,
            bidirectional=  True,
            dropout=        dropout if lstm_layers > 1 else 0.0,
        )

        # Token-level classifier (predicts for each timestep)
        self.dropout = nn.Dropout(dropout)
        self.token_classifier = nn.Sequential(
            nn.Linear(2 * lstm_hidden, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    @staticmethod
    def _same_pad(kernel_hw: tuple) -> tuple:
        kh, kw = kernel_hw
        return (kh // 2, kw // 2)

    def _downscale_lengths(self, lengths: torch.Tensor) -> torch.Tensor:
        L = lengths.clone()
        for (ph, pw) in self.pool_sizes:
            L = torch.div(L, ph, rounding_mode='floor')
        L = torch.clamp(L, min=1)
        return L

    def forward(self, x, lengths):
        """
        Returns token-level logits: (B, L', num_classes)
        where L' is the downsampled sequence length after CNN pooling
        """
        
        B = x.size(0)

        # Add channel → (B, 1, L, E)
        x = x.unsqueeze(1)

        # CNN layers
        for conv, bn, pool in zip(self.convs, self.bns, self.pools):
            x = pool(F.relu(bn(conv(x))))  # (B, C, L', E')

        # Mean over embedding dimension
        x = x.mean(dim=3)  # (B, C, L')

        # Swap for LSTM: (B, L', C)
        x = x.permute(0, 2, 1).contiguous()

        # BLSTM with length-aware packing
        if lengths is not None:
            lens_ds = self._downscale_lengths(lengths.to(x.device))
            packed = nn.utils.rnn.pack_padded_sequence(
                x, lens_ds.cpu(), batch_first=True, enforce_sorted=False
            )
            packed_out, _ = self.blstm(packed)
            out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)
        else:
            out, _ = self.blstm(x)  # (B, L', 2*H)

        # Apply token classifier to each timestep
        # out: (B, L', 2*H) -> logits: (B, L', num_classes)
        logits = self.token_classifier(self.dropout(out))
        
        return logits


###################
#### Trainning ####
###################


def collate_batch(batch):

    sequences, labels, lengths = zip(*batch)
    
    # Pad sequences to max length in batch
    padded_seqs     = pad_sequence(sequences, batch_first=True, padding_value=0.0)
    # Pad labels also val=-100 tells CrossEntropyLoss to ignore these positions
    label_tensors   = [torch.tensor(label_seq, dtype=torch.long) for label_seq in labels]
    padded_labels   = pad_sequence(label_tensors, batch_first=True, padding_value=-100)
    # Get original length for LSTM track
    lengths = torch.tensor(lengths, dtype=torch.long)
    
    return padded_seqs, padded_labels, lengths

def prepare_dataloaders(dataset, batch_size, test_split, num_workers, seed):
    
    if not 0 <= test_split < 1:
        raise ValueError("test_split must be within [0, 1).")

    dataset_size = len(dataset)
    if dataset_size == 0:
        raise ValueError("Dataset is empty. Ensure corpus and labels are non-empty.")

    test_size = int(math.floor(dataset_size * test_split))
    train_size = dataset_size - test_size

    if train_size == 0:
        raise ValueError("test_split too large: no samples left for training.")

    generator = torch.Generator().manual_seed(seed)
    if test_size > 0:
        train_dataset, test_dataset = random_split(
            dataset,
            [train_size, test_size],
            generator=generator,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate_batch,
        )
    else:
        train_dataset = dataset
        test_loader = None

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_batch,
    )

    return train_loader, test_loader

def evaluate(model, dataloader, device, loss_fn):
    
    if dataloader is None:
        return float("nan"), float("nan")

    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    with torch.no_grad():
        for inputs, targets, lengths in dataloader:
            inputs  = inputs.to(device)
            targets = targets.to(device)
            lengths = lengths.to(device)

            logits  = model(inputs, lengths)
            loss    = loss_fn(logits, targets)

            total_loss += loss.item() * targets.size(0)
            predictions = torch.argmax(logits, dim=1)
            total_correct += (predictions == targets).sum().item()
            total_examples += targets.size(0)

    avg_loss = total_loss / total_examples if total_examples else float("nan")
    accuracy = total_correct / total_examples if total_examples else float("nan")
    return avg_loss, accuracy
