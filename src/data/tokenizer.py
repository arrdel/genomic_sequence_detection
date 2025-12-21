import torch
from itertools import product
from Bio import SeqIO

class KmerTokenizer:
    def __init__(self, k=6, use_canonical=False):
        self.k = k
        self.use_canonical = use_canonical
        
        # build canonical DNA k-mers
        self.kmers = ["".join(p) for p in product("ACGT", repeat=k)]
        self.stoi = {kmer: idx for idx, kmer in enumerate(self.kmers)}

        # --- Add essential special tokens ---
        self.special_tokens = ["<PAD>", "<UNK>", "<MASK>"]
        for tok in self.special_tokens:
            if tok not in self.stoi:
                self.stoi[tok] = len(self.stoi)

        # convenient references
        self.pad_id = self.stoi["<PAD>"]
        self.unk_id = self.stoi["<UNK>"]
        self.mask_id = self.stoi["<MASK>"]

        # build inverse map
        self.itos = {i: k for k, i in self.stoi.items()}

        print(f"Tokenizer built: vocab size = {len(self.stoi)} (includes PAD/UNK/MASK)")

    def add_special_token(self, token):
        """Adds a new special token if it does not exist."""
        if token not in self.stoi:
            self.stoi[token] = len(self.stoi)
            self.itos[self.stoi[token]] = token
        setattr(self, f"{token.strip('<>')}_id", self.stoi[token])

    def encode_seq(self, seq, max_len=150):
        """Convert nucleotide sequence to list of k-mer IDs."""
        tokens = []
        for i in range(len(seq) - self.k + 1):
            kmer = seq[i:i+self.k]
            if set(kmer) <= {"A", "C", "G", "T"}:
                if self.use_canonical:
                    rc = kmer.translate(str.maketrans("ACGT", "TGCA"))[::-1]
                    kmer = min(kmer, rc)
                # if unseen k-mer, map to UNK
                tokens.append(self.stoi.get(kmer, self.unk_id))

        # pad/truncate
        if len(tokens) < max_len:
            tokens += [self.pad_id] * (max_len - len(tokens))
        else:
            tokens = tokens[:max_len]

        return tokens

    def decode_tokens(self, token_ids, remove_pad=True, reconstruct=False):
        """Decode list of IDs into k-mers or full nucleotide sequence."""
        kmers = []
        for tid in token_ids:
            if tid == self.pad_id and remove_pad:
                continue
            kmers.append(self.itos.get(int(tid), "<UNK>"))

        if not reconstruct:
            return kmers

        # reconstruct approximate nucleotide sequence from overlapping k-mers
        if not kmers:
            return ""
        seq = kmers[0]
        for kmer in kmers[1:]:
            seq += kmer[-1]
        return seq


# -------------------------------------
# Torch Dataset
# -------------------------------------
class FastqKmerDataset:
    def __init__(self, fastq_file, tokenizer, max_len=150):
        self.records = list(SeqIO.parse(fastq_file, "fastq"))
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        seq = str(self.records[idx].seq)
        tokens = self.tokenizer.encode_seq(seq, self.max_len)
        true_len = min(len(seq) - self.tokenizer.k + 1, self.max_len)
        return torch.tensor(tokens, dtype=torch.long), torch.tensor(true_len, dtype=torch.long)