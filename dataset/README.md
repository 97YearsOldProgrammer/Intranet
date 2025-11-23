### Get Whole C elegans gff3 file from wormbase  

filter all CDS, exon, 5'UTR, 3'UTR region out     

```zsh
python3 filter_typ.py celegans/caenorhabditis_elegans.PRJNA13758.WBPS19.annotations.gff3 celegansgenes.gff3
```


---


### Stats Check

using linux/unix cmd to grep inforamtion we need

```zsh
cut -f3 celegansgenes.gff3 | sort | uniq -c    
```

```
3067597 CDS
405331 exon
27493 five_prime_UTR
1066189 intron
47719 three_prime_UTR
```

```zsh
cut -f2 celegansgenes.gff3 | sort | uniq -c
```

```
133896 Genefinder
132967 GeneMarkHMM
133716 history
123207 jigsaw
134050 mGene
168585 mSplicer_orf
171688 mSplicer_transcript
872936 RNASeq_splice
1541006 RNASEQ.Hillier
342659 RNASEQ.Hillier.Aggregate
   2 Transposon_ncRNA
124073 twinscan
21344 UTRome
5331 WBPaper00056245
707870 WormBase
 999 WormBase_transposon
```

| Source                    | Meaning                                   |
|---------------------------|-------------------------------------------|
| Genefinder                | ab initio gene predictor                  |
| GeneMarkHMM               | Hidden Markov Model gene predictor        |
| history                   | legacy manual annotations                 |
| jigsaw                    | evidence-integration gene predictor       |
| mGene                     | ML-based gene predictor                   |
| mSplicer_orf              | splice-site + ORF predictor               |
| mSplicer_transcript       | transcript-level splicing predictor       |
| RNASeq_splice             | raw splice junctions from RNA-seq         |
| RNASEQ.Hillier            | RNA-seq expression evidence               |
| RNASEQ.Hillier.Aggregate  | merged RNA-seq datasets                   |
| Transposon_ncRNA          | ncRNA related to transposons              |
| twinscan                  | comparative genomics gene predictor       |
| UTRome                    | experimental UTR annotations              |
| WBPaper00056245           | annotations from a specific paper         |
| WormBase                  | curated final gene models                 |

We only need the RNASeq_splice for GT AG and intron information


---


### Second Filter

