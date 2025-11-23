import os
import re
import gzip
import sys
import typing as tp

from pathlib        import Path
from contextlib     import closing
from dataclasses    import dataclass
from collections    import defaultdict



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

def parse_fasta(filename):

	fp = getfp(filename)
 
	chms    = {}
	chm     = None
	seq     = []
	
	with closing(fp):
		for line in fp:
			line = line.strip()
			if line.startswith('>'):
				if chm is not None:
					chms[chm] = ''.join(seq)
				chm = line[1:].split()[0]
				seq = []
			else:
				seq.append(line)
		
		if chm is not None:
			chms[chm] = ''.join(seq)
	
	return chms

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


############################
## Small Gene Set Parsing ##
############################


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


############################
## WormBase GFF3 Feature ##
############################


@dataclass
class WormBaseFeature:
    """Parse Single WormBase GFF3 line with chromosome info"""
    
    chm: str
    src: str
    typ: str
    beg: int
    end: int
    scr: tp.Optional[float]
    std: str
    pha: tp.Optional[int]
    att: tp.Dict[str, str]

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

def parse_wormbase_gff(filename):
    
    fp = getfp(filename)
    
    # Three separate datasets
    introns     = []                    # RNASeq_splice introns
    proteins    = defaultdict(list)     # CDS grouped by parent
    genes       = defaultdict(list)     # Full transcripts with UTRs/exons/introns
    
    with closing(fp):
        for line in fp:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            
            parts = line.split("\t")
            
            if len(parts) == 8:
                parts.append(".")
            elif len(parts) != 9:
                continue
            
            chm, src, typ, beg, end, scr, std, pha, att = parts
            scr = None if scr == "." else float(scr)
            pha = None if pha == "." else int(pha)
            
            att = parse_att(att)
            
            feature = WormBaseFeature(
                chm=chm,
                src=src,
                typ=typ.lower(),
                beg=int(beg),
                end=int(end),
                scr=scr,
                std=std,
                pha=pha,
                att=att,
            )
            
            if src == "RNASeq_splice":
                introns.append(feature)
            
            elif src == "WormBase":
                # Proteins
                if typ.lower() == "cds":
                    parent_id = att.get("Parent", att.get("ID", f"{chm}:{beg}-{end}"))
                    proteins[parent_id].append(feature)
                # Genes
                else:
                    parent_id = att.get("Parent", att.get("ID", f"{chm}:{beg}-{end}"))
                    genes[parent_id].append(feature)
    
    return introns, proteins, genes

def process_introns(introns, chm_dict):

    intron = []
    
    for feature in introns:
        
        seq = chm_dict[feature.chm][feature.beg-1:feature.end]
        if feature.std == "-":
            seq = anti(seq)
        intron.append(seq)
    
    return intron

def process_cds(proteins, chm_dict):

    protein = []
    
    for parent_id, features in proteins.items():
        
        features = sorted(features, key=lambda x: x.beg)
        
        if not features:
            continue

        chm = features[0].chm
        std = features[0].std
        
        cds = []
        for feature in features:
            beg = feature.beg
            
            if feature.pha is not None and feature.pha > 0:
                beg = feature.beg + feature.pha
            
            seq = chm_dict[chm][beg-1:feature.end]
            if std == "-":
                seq = anti(seq)

            cds.append(seq)
        protein.append(cds)

    return protein

def separate_by_strand(genes):

    fw_genes = defaultdict(list)
    rv_genes = defaultdict(list)
    
    for parent_id, features in genes.items():
        if not features:
            continue
        
        std = features[0].std
        
        if std == "+":
            fw_genes[f"{parent_id}"] = features
        else:
            rv_genes[f"{parent_id}"] = features
    
    return fw_genes, rv_genes

def align_genes(features, chm_dict):

    if not features:
        return []
    
    # Sort by position
    features = sorted(features , key=lambda x: x.beg)
    
    chm = features[0].chm
    std = features[0].std
    
    aligned_parts = []
    
    for feature in features:
        seq = ''
        seq = chm_dict[chm][feature.beg-1:feature.end]
        
        if std == "-":
            seq = anti(seq)
        
        # Determine feature label
        if feature.typ == "five_prime_utr":
            label = "5UTR"
        elif feature.typ == "three_prime_utr":
            label = "3UTR"
        elif feature.typ == "exon":
            label = "exon"
        elif feature.typ == "intron":
            label = "intron"
        else:
            label = feature.typ
        
        aligned_parts.append({
            'typ': label,
            'beg': feature.beg,
            'end': feature.end,
            'seq': seq,
            'std': std
        })
    
    # If negative strand, reverse the order
    if std == "-":
        aligned_parts = aligned_parts[::-1]
    
    return aligned_parts

def process_transcripts(genes, chm_dict):

    fw_transcripts, rv_transcripts = separate_by_strand(genes)
    
    fw_aligned = {}
    for fw_id, features in fw_transcripts.items():
        fw_aligned[fw_id] = align_genes(features, chm_dict)
    
    rv_aligned = {}
    for rv_id, features in rv_transcripts.items():
        rv_aligned[rv_id] = align_genes(features, chm_dict)
    
    return fw_aligned, rv_aligned

def tokenize_sequences(aligned_transcripts, tokenizer):

    label_map = {
        'exon': 0,
        'intron': 1,
        '5UTR': 2,
        '3UTR': 3,
    }

    tokenized = {}
    labels = {}
    
    for transcript_id, parts in aligned_transcripts.items():
        tokens = []
        label_ids = []
        
        for part in parts:
            if part['seq']:
                part_tokens = tokenizer(part['seq'])
                tokens.extend(part_tokens)
                
                label_value = label_map.get(part['typ'], -1)
                label_ids.extend([label_value] * len(part_tokens))
        
        tokenized[transcript_id]    = tokens
        labels[transcript_id]       = label_ids
    
    return tokenized, labels

class WormBaseDataset:

    def __init__(self, gff_file, fasta_files):
        """Initialize with GFF and FASTA files"""

        # Parse FASTA files into per chromosome
        self.chms = {}
        for fasta in fasta_files:
            self.chms.update(parse_fasta(fasta))

        # Parse GFF3
        self.introns, self.proteins, self.genes = parse_wormbase_gff(gff_file)
        
        # Process datasets
        self.introns    = process_introns(self.introns, self.chms)
        self.proteins   = process_cds(self.proteins, self.chms)
        self.fw_transcripts, self.rv_transcripts = process_transcripts(
            self.genes, self.chms
        )

    def get_introns(self):
        return self.introns
    
    def get_cds(self):
        return self.proteins
    
    def get_forward_transcripts(self):
        return self.fw_transcripts
    
    def get_reverse_transcripts(self):
        return self.rv_transcripts
    
    def tokenize_all(self, tokenizer):
        fw_tokens, fw_labels = tokenize_sequences(self.fw_transcripts, tokenizer)
        rv_tokens, rv_labels = tokenize_sequences(self.rv_transcripts, tokenizer)
        
        return {
            'forward': (fw_tokens, fw_labels),
            'reverse': (rv_tokens, rv_labels)
        }


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