### Tokenize dataset to corpus  

First step of training is converting all the DNA sequences we have through a kmer sliding window algorithm.     
It's amazing design that uses a kmer length way of tokenizing DNA sequences instead of one hot string.     
Based on two datasets, there are two tokenizer programs.   

First one: [`data2token.py`](/bin/data2token.py) is built for small gene set     

```zsh
python3 bin/data2token.py dataset/smg data/corpus/c_smg.txt data/label/l_smg.txt -io
```

Second one: [`data2token2.py`](/bin/data2token2.py) is built for a more generic purpose.

```zsh
python3 bin/data2token2.py dataset/whole_celegan/ce_exintron.gff3 dataset/whole_celegan/caenorhabditis_elegans.PRJNA13758.WBPS19.genomic.fa data/corpus/whole_celegan.tx
t data/label/whole_celegan.txt 
```

The part of sequence that being tokenized for embedding is transripts, which is 5'UTR/exon/intron/3'UTR or reversed genes.  
Introns and proteins are not considered. Since they are also a combinatoric parts of transcripts.   

---



### Embed corpus with GloVe

Second step of the training is dumping all tokenized DNA into a Euclidean Space through official GloVe, an unsupervised learning embedding model.     
As a statistical way of linear projection. Still considering using a another way to substitude that.    

```zsh
python3 bin/tglove.py data/corpus/c_smg.txt data/trained/vector_smg glove/build
```

As embedding dimension is some parameter that would decide directly about the weights of input data.    
For using the whole genome of C elegans, it seem like using embeeding dimension 256 instead 100 would results in better trainning loss.     

| Dimension  | Epoch                 | Loss     |
|------------|-----------------------|----------|
| 100        | 15                    | 0.105146 |
| 256        | 150                   | 0.065872 |