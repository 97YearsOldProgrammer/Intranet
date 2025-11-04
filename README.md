### Tokenize dataset to corpus  

First step of trainning is converting all the DNA sequence we have through a kmer sliding window algorithm.      

```zsh
python3 bin/data2token.py data/smg/smallgenes data/corpus/c_smg.txt data/label/l_smg.txt -io
```

---

### Embed corpus with GloVe

Second step of the trainning is dumping all tokenized DNA into a Euclidean Space through official glove, a unsupervised learning embedding model.   

```zsh
python3 bin/tglove.py data/corpus/c_smg.txt data/trained/vector_smg glove/build
```