import math
import numpy as np
from typing         import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader, random_split

from dataclasses    import dataclass



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
            
            try:
                idx = int(parts[0])
            # glove would include <unk> automatically
            except ValueError:
                print(f"Skipping non-integer token: {parts[0]}")
                continue
            
            vector  = torch.tensor([float(x) for x in parts[1:]], dtype=torch.float32)
            embeddings[idx] = vector
    
    return embeddings

def embed_sentence(line, embeddings):
    
    idxs = [int(x) for x in line.strip().split()]
    
    if not idxs:
        return torch.tensor([])

    # some token are too rare for embedding threshold
    valid_embeddings = []
    for token_id in idxs:
        if token_id in embeddings:
            valid_embeddings.append(embeddings[token_id])

    # return empty string for rare kmer
    if not valid_embeddings:
        return torch.tensor([])

    # Stack all embedding vectors
    return torch.stack(valid_embeddings, dim=0)

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

class BiLSTM(nn.Module):
    """BLSTM for vertical and horizontal scan"""
    
    def __init__(
        self, 
        c_in: int, 
        hidden: int
    ):
        super().__init__()
        self.vgru = nn.LSTM(input_size=c_in, hidden_size=hidden, batch_first=True, bidirectional=True)
        self.hgru = nn.LSTM(input_size=2*hidden, hidden_size=hidden, batch_first=True, bidirectional=True)

    def forward(self, x):
        B, C, H, W = x.shape    # (B, C, H, W)

        # vertical scan
        v_in        = x.permute(0, 3, 2, 1).contiguous().view(B*W, H, C)    # (B*W, H, C)
        v_out, _    = self.vgru(v_in)                                       # (B*W, H, 2h)
        v_out       = v_out.view(B, W, H, -1).permute(0, 2, 1, 3)           # (B, H, W, 2h)

        # horizontal scan
        h_in        = v_out.contiguous().view(B*H, W, -1)                   # (B*H, W, 2h)
        h_out, _    = self.hgru(h_in)                                       # (B*H, W, 2h)
        h_out       = h_out.view(B, H, W, -1).permute(0, 3, 1, 2)           # (B, 2h, H, W)

        return h_out


class IntraNet(nn.Module):
    """
    VGG-16 -> BiLSTM -> Upsample -> Compress W dimension -> FC with BatchNorm
    
    For DNA sequence data:
    - H = number of tokens (sequence length)
    - W = embedding dimension (e.g., 4 for one-hot DNA)
    """

    def __init__(
        self,
        input_height:           int = None, # H = num_tokens
        input_width:            int = 4,    # W = embedding_dim (e.g., 4 for DNA)
        num_classes:            int = 2,
        lstm_hidden:            int = 256,
        dropout:                float = 0.5,
        fc_hidden:              list = [256, 128, 64],  # FC layer sizes
        compression_method:     str = 'adaptive_pool',  # 'adaptive_pool', 'avg_pool', 'max_pool', 'conv'
    ):
        super().__init__()
        self.input_height = input_height
        self.input_width = input_width
        self.num_classes = num_classes
        self.dropout_2d = nn.Dropout2d(dropout)

        # VGG-16 encoder (deeper)
        self.encoder = nn.Sequential(
            self._vgg_block(1,   64,  2),       # H -> H/2
            self._vgg_block(64,  128, 2),       # H/2 -> H/4
            self._vgg_block(128, 256, 3),       # H/4 -> H/8
            self._vgg_block(256, 512, 3),       # H/8 -> H/16
        )

        # ReSeg layers
        self.renet1 = BiLSTM(512, lstm_hidden)                    
        self.renet2 = BiLSTM(2*lstm_hidden, lstm_hidden)          
        
        # Transposed convolution
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(2*lstm_hidden, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )

        # ===== COMPRESS W DIMENSION =====
        if compression_method == 'adaptive_pool':
            self.compress_w = nn.AdaptiveAvgPool2d((None, 1))  # (B, 64, H, W) -> (B, 64, H, 1)
        elif compression_method == 'avg_pool':
            self.compress_w = lambda x: torch.mean(x, dim=3, keepdim=True)
        elif compression_method == 'max_pool':
            self.compress_w = lambda x: torch.max(x, dim=3, keepdim=True)[0]
        elif compression_method == 'conv':
            # Learnable compression
            self.compress_w = nn.Conv2d(64, 64, kernel_size=(1, input_width), padding=0)
        else:
            raise ValueError(f"Unknown compression method: {compression_method}")
        
        # FC CLASSIFIER with BatchNorm
        # Input: (B, 64, H, 1) -> reshape to (B*H, 64)
        fc_layers = []
        prev_dim = 64  # number of channels after compression
        
        for hidden_dim in fc_hidden:
            fc_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        # Final output layer (per-token classification)
        fc_layers.append(nn.Linear(prev_dim, num_classes))
        
        self.fc_classifier = nn.Sequential(*fc_layers)
        
    @staticmethod
    def _vgg_block(c_in, c_out, n_convs):
        layers = []
        for i in range(n_convs):
            layers += [
                nn.Conv2d(c_in if i == 0 else c_out, c_out, kernel_size=3, padding=1, bias=True),
                nn.ReLU(inplace=True),
            ]
        layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        return nn.Sequential(*layers)

    def forward(self, x):
        """
        Args:
            x: (B, 1, H, W) where H=num_tokens, W=embedding_dim
        
        Returns:
            (B, num_classes, H) - per-token predictions
        """
        
        B, _, H, W = x.shape
        
        # VGG encoding (downsample H)
        x = self.encoder(x)                 # (B, 512, H/16, W/16)
        
        # BiLSTM layers
        x = self.renet1(x)                  # (B, 2*hidden, H/16, W/16)
        x = self.dropout_2d(x)
        x = self.renet2(x)                  # (B, 2*hidden, H/16, W/16)
        x = self.dropout_2d(x)
        
        # Upsample back to ORIGINAL (H, W)
        x = self.upsample(x)                # (B, 64, H, W) ← back to original size!
        
        # ===== COMPRESS W DIMENSION =====
        # (B, 64, H, W) -> (B, 64, H, 1)
        x = self.compress_w(x)              # (B, 64, H, 1)
        
        # Reshape for FC: (B, 64, H, 1) -> (B, H, 64)
        x = x.squeeze(3)                    # (B, 64, H)
        x = x.permute(0, 2, 1)              # (B, H, 64)
        x = x.contiguous().view(-1, 64)     # (B*H, 64) - each token is a 64-dim vector
        
        # ===== FC CLASSIFIER =====
        x = self.fc_classifier(x)           # (B*H, num_classes)
        
        # Reshape back: (B*H, num_classes) -> (B, H, num_classes)
        x = x.view(B, H, self.num_classes)  # (B, H, num_classes)
        x = x.permute(0, 2, 1)              # (B, num_classes, H)
        
        return x

    
###################
#### Trainning ####
###################


def downsample_single_label_sequence(label_seq, pool_sizes):
    """CNN would downsaple the label"""
    
    labels = torch.tensor(label_seq, dtype=torch.long)
    
    # Apply pooling iteratively like the CNN does
    for (ph, pw) in pool_sizes:
        L = len(labels)
        new_L = L // ph  # Floor division at each stage
        # Take every ph-th element, truncate to new_L
        labels = labels[::ph][:new_L]
    
    return labels.tolist()


def collate_batch(batch):

    sequences, labels, lengths = zip(*batch)
    
    # Pad sequences FIRST
    padded_seqs = pad_sequence(sequences, batch_first=True, padding_value=0.0)
    
    # Get the padded length (what CNN will see)
    padded_length = padded_seqs.shape[1]  # The L dimension
    
    # Calculate what length CNN will output
    pool_sizes = [(2, 2), (2, 2), (2, 2)]
    cnn_output_length = padded_length
    for (ph, pw) in pool_sizes:
        cnn_output_length = cnn_output_length // ph
    
    # Downsample ALL labels to this SAME length
    downsampled_labels = []
    for label_seq in labels:
        # Pad or truncate each label sequence to cnn_output_length
        labels_tensor = torch.tensor(label_seq, dtype=torch.long)
        
        # Downsample this label sequence
        ds_labels = downsample_single_label_sequence(label_seq, pool_sizes)
        
        # Truncate or pad to match cnn_output_length exactly
        if len(ds_labels) > cnn_output_length:
            ds_labels = ds_labels[:cnn_output_length]
        elif len(ds_labels) < cnn_output_length:
            # Pad with -100
            ds_labels = ds_labels + [-100] * (cnn_output_length - len(ds_labels))
        
        downsampled_labels.append(torch.tensor(ds_labels, dtype=torch.long))
    
    # Stack (all same length now)
    padded_labels = torch.stack(downsampled_labels)
    
    # Original lengths
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
    total_tokens = 0
    total_correct = 0

    with torch.no_grad():
        for inputs, targets, lengths in dataloader:
            inputs  = inputs.to(device)
            targets = targets.to(device)
            lengths = lengths.to(device)

            logits  = model(inputs, lengths)                       # (B, L', C)
            loss    = loss_fn(logits.permute(0, 2, 1), targets)    # (B, C, L')

            # Mask padding for accuracy
            pred = torch.argmax(logits, dim=2)                     # (B, L')
            mask = (targets != -100)
            correct = (pred.eq(targets) & mask).sum().item()
            tokens  = mask.sum().item()

            total_loss   += loss.item() * tokens
            total_correct += correct
            total_tokens  += tokens

    avg_loss = total_loss / total_tokens if total_tokens else float("nan")
    accuracy = total_correct / total_tokens if total_tokens else float("nan")
    return avg_loss, accuracy