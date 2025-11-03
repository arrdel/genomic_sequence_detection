import torch
from itertools import product
from Bio import SeqIO
class KmerTokenizer:
    def __init__(self, k=6, use_canonical=False):
        self.k = k
        self.use_canonical = use_canonical
        self.kmers = ["".join(p) for p in product("ACGT", repeat=k)]
        self.stoi = {kmer: idx for idx, kmer in enumerate(self.kmers)}
        self.pad_token = "<PAD>"
        self.pad_id = len(self.kmers)
        self.stoi[self.pad_token] = self.pad_id
        self.itos = {i: k for k, i in self.stoi.items()}
        print(f"Tokenizer built: vocab size = {len(self.stoi)}")

    def encode_seq(self, seq, max_len=150):
        # extract ordered k-mers
        tokens = []
        for i in range(len(seq) - self.k + 1):
            kmer = seq[i:i+self.k]
            if set(kmer) <= {"A","C","G","T"}:  # skip N bases
                if self.use_canonical:
                    rc = kmer.translate(str.maketrans("ACGT", "TGCA"))[::-1]
                    kmer = min(kmer, rc)
                tokens.append(self.stoi.get(kmer, self.pad_id))  # unknown → PAD
        # pad/truncate
        if len(tokens) < max_len:
            tokens = tokens + [self.pad_id] * (max_len - len(tokens))
        else:
            tokens = tokens[:max_len]
        return tokens
    
    def decode_tokens(self, token_ids, remove_pad=True, reconstruct=False):
        """
        token_ids: list of ints (IDs)
        remove_pad: remove <PAD> tokens
        reconstruct: if True, rebuild approximate nucleotide sequence 
                     by overlapping k-mers.
        """
        kmers = []
        for tid in token_ids:
            if tid == self.pad_id and remove_pad:
                continue
            kmers.append(self.itos.get(int(tid), "<UNK>"))

        if not reconstruct:
            return kmers  # just list of k-mers

        # Reconstruct sequence from overlapping k-mers
        if not kmers:
            return ""
        seq = kmers[0]
        for kmer in kmers[1:]:
            seq += kmer[-1]  # overlap last base
        return seq

# Torch Dataset
class FastqKmerDataset():
    def __init__(self, fastq_file, tokenizer, max_len=150):
        self.records = list(SeqIO.parse(fastq_file, "fastq"))
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        seq = str(self.records[idx].seq)
        tokens = self.tokenizer.encode_seq(seq, self.max_len)
        return torch.tensor(tokens, dtype=torch.long)
