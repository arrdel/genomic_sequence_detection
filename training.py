import os
import Bio
import vqvae
import torch
import wandb
from tokenizer import KmerTokenizer, FastqKmerDataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
from Bio import SeqIO
import numpy as np
from datetime import datetime
import random
import tqdm

# Initialize wandb
wandb.init(
    project="vqvae-genomics",
    config={
        "architecture": "VQ-VAE",
        "dataset": "viral-sequences",
        "vocab_size": 4097,
        "num_codes": 512,
        "code_dim": 64,
        "embed_dim": 128,
        "hidden_dim": 256,
        "batch_size": 64,
        "learning_rate": 2e-4,
        "num_epochs": 100,
        "max_seq_length": 150
    }
)



def train_one_epoch(model, dataloader, optimizer, device, pad_id, recon_loss_weight=1.0):
    model.train()
    total_loss, total_recon, total_vq, n_tokens = 0, 0, 0, 0

    for batch_tokens, batch_lengths in dataloader:
        # Move both tensors to device
        batch_tokens = batch_tokens.to(device)  # Shape: [batch_size, max_len]
        batch_lengths = batch_lengths.to(device)  # Shape: [batch_size]

        optimizer.zero_grad()

        # Forward pass - only pass tokens to model
        logits, loss_vq, codes = model(batch_tokens)  # logits: [batch_size, max_len, vocab_size]
        B, L, V = logits.shape

        # Prevent model from predicting PAD tokens
        logits[:, :, pad_id] = float('-inf')

        # Create mask based on sequence lengths
        mask = torch.arange(L, device=device)[None, :] < batch_lengths[:, None]  # [B, L]
        mask = mask & (batch_tokens != pad_id)

        # Flatten for loss calculation
        logits_flat = logits[mask]  # [valid_tokens, vocab_size]
        targets_flat = batch_tokens[mask]  # [valid_tokens]

        if logits_flat.size(0) == 0:
            continue

        recon_loss = F.cross_entropy(logits_flat, targets_flat)
        loss = recon_loss_weight * recon_loss + loss_vq

        loss.backward()
        optimizer.step()

        # Update statistics
        n_valid = targets_flat.size(0)
        total_loss += loss.item() * n_valid
        total_recon += recon_loss.item() * n_valid
        total_vq += loss_vq.item() * n_valid
        n_tokens += n_valid

        # Free up memory
        del logits, loss_vq, codes, loss
        torch.cuda.empty_cache()

    return {
        "loss": total_loss / n_tokens,
        "recon": total_recon / n_tokens,
        "vq": total_vq / n_tokens
    }

def collate_fn(batch):
    tokens = torch.stack([b[0] for b in batch])
    lengths = torch.stack([b[1] for b in batch])
    return tokens, lengths


Dataset = "/home/adelechinda/home/semester_projects/fall_25/deep_learning/project/cleaned_reads.fastq"

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
        tokens_tensor = torch.tensor(tokens, dtype=torch.long)
        
        # Calculate actual sequence length (excluding padding)
        length = (tokens_tensor != self.tokenizer.pad_id).sum()
        
        return tokens_tensor, length

def collate_fn(batch):
    # Separate tokens and lengths
    tokens = torch.stack([item[0] for item in batch])
    lengths = torch.tensor([item[1] for item in batch])
    return tokens, lengths

# Model configuration
VOCAB_SIZE = 4097  # 4^6 + 1 for PAD
PAD_ID = 4096
NUM_CODES = 512
CODE_DIM = 64
EMBED_DIM = 128
HIDDEN_DIM = 256
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_epochs = 100

print(f"Using device: {DEVICE}")

# Initialize model
model = vqvae.VQVAE(VOCAB_SIZE, PAD_ID, num_codes=NUM_CODES, code_dim=CODE_DIM,
                embed_dim=EMBED_DIM, hidden_dim=HIDDEN_DIM, commitment_cost=0.25).to(DEVICE)

# Log model architecture to wandb
wandb.watch(model, log="all")

# Create dataset and dataloader
tokenizer_ = KmerTokenizer(k=6, use_canonical=True)
dataset = FastqKmerDataset("/home/adelechinda/home/semester_projects/fall_25/deep_learning/project/cleaned_reads.fastq", 
                          tokenizer_, max_len=150)

# Use DataLoader with custom collate_fn
dataloader = DataLoader(
    dataset, 
    batch_size=64, 
    shuffle=True, 
    num_workers=4,
    collate_fn=collate_fn,
    pin_memory=True  # This helps with GPU transfer
)

optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)

# Create output directory for checkpoints
output_dir = f"outputs/run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
os.makedirs(output_dir, exist_ok=True)

# Training loop with error handling and wandb logging
for epoch in range(1, num_epochs+1):
    try:
        stats = train_one_epoch(model, dataloader, optimizer, DEVICE, PAD_ID)
        
        # Log metrics to wandb
        wandb.log({
            "epoch": epoch,
            "loss": stats['loss'],
            "reconstruction_loss": stats['recon'],
            "vq_loss": stats['vq']
        })
        
        print(f"Epoch {epoch}: loss {stats['loss']:.6f} recon {stats['recon']:.6f} vq {stats['vq']:.6f}")
        
        # Save checkpoint
        if epoch % 1 == 0:
            checkpoint_path = os.path.join(output_dir, f'checkpoint_epoch_{epoch}.pt')
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'stats': stats
            }
            torch.save(checkpoint, checkpoint_path)
            
            # Log checkpoint as artifact
            artifact = wandb.Artifact(f'model-checkpoint-{epoch}', type='model')
            artifact.add_file(checkpoint_path)
            wandb.log_artifact(artifact)
            
    except Exception as e:
        print(f"Error in epoch {epoch}: {str(e)}")
        raise  # Re-raise the exception for debugging

def log_sequence_reconstructions(model, dataloader, tokenizer, device, num_examples=5):
    """Log sequence reconstruction examples to wandb"""
    batch_tokens, batch_lengths = next(iter(dataloader))
    examples = reconstruct_and_decode(model, batch_tokens, batch_lengths, tokenizer, device)
    
    # Create a table to log
    columns = ["example_id", "input_sequence", "reconstructed_sequence"]
    data = []
    
    for i, (in_seq, pred_seq) in enumerate(examples[:num_examples]):
        # Truncate sequences to first 120 bp for visualization
        data.append([i, in_seq[:120], pred_seq[:120]])
    
    table = wandb.Table(data=data, columns=columns)
    wandb.log({"sequence_reconstructions": table})

def reconstruct_and_decode(model, batch_tokens, batch_lengths, tokenizer, device):
    model.eval()
    batch_tokens = batch_tokens.to(device)
    batch_lengths = batch_lengths.to(device)

    with torch.no_grad():
        logits, _, _ = model(batch_tokens)
        preds = logits.argmax(dim=-1).cpu()

    results = []
    for i in range(batch_tokens.size(0)):
        L_true = int(batch_lengths[i].item())
        input_ids = batch_tokens[i, :L_true].cpu().numpy()
        pred_ids = preds[i, :L_true].numpy()

        input_seq = tokenizer.decode_tokens(input_ids, remove_pad=True, reconstruct=True)
        recon_seq = tokenizer.decode_tokens(pred_ids, remove_pad=True, reconstruct=True)

        results.append((input_seq, recon_seq))

    return results


# # Get one batch (tokens + lengths)
# batch_tokens, batch_lengths = next(iter(dataloader))

# # Run reconstruction
# examples = reconstruct_and_decode(model, batch_tokens, batch_lengths, tokenizer_, DEVICE)

# # Print results
# for i, (in_seq, pred_seq) in enumerate(examples[:5]):
#     print(f"==== Example {i} ====")
#     print(f"Input seq (first 120 bp):  {in_seq[:120]}")
#     print(f"Recon seq (first 120 bp):  {pred_seq[:120]}")
#     print()



# -----------------------------
# Utilities
# -----------------------------
# def extract_codes(model, dataloader, device):
#     model.eval()
#     all_codes = []
#     with torch.no_grad():
#         for batch in dataloader:
#             batch = batch.to(device)
#             _, _, codes = model(batch)  # (B, L)
#             all_codes.append(codes.cpu().numpy())
#     return all_codes  # list of arrays per batch



# def reconstruct_and_decode(model, batch_tokens, batch_lengths, tokenizer, device):
#     model.eval()
#     batch_tokens = batch_tokens.to(device)
#     batch_lengths = batch_lengths.to(device)

#     with torch.no_grad():
#         logits, _, _ = model(batch_tokens)
#         preds = logits.argmax(dim=-1).cpu()

#     results = []
#     for i in range(batch_tokens.size(0)):
#         L_true = int(batch_lengths[i].item())
#         input_ids = batch_tokens[i, :L_true].cpu().numpy()
#         pred_ids = preds[i, :L_true].numpy()

#         input_seq = tokenizer.decode_tokens(input_ids, remove_pad=True, reconstruct=True)
#         recon_seq = tokenizer.decode_tokens(pred_ids, remove_pad=True, reconstruct=True)

#         results.append((input_seq, recon_seq))

#     return results

# # %%
# # Get one batch (tokens + lengths)
# batch_tokens, batch_lengths = next(iter(dataloader))

# # Run reconstruction
# examples = reconstruct_and_decode(model, batch_tokens, batch_lengths, tokenizer_, DEVICE)

# # Print results
# for i, (in_seq, pred_seq) in enumerate(examples[:]):
#     print(f"==== Example {i} ====")
#     print(f"Input seq (first 120 bp):  {in_seq[:120]}")
#     print(f"Recon seq (first 120 bp):  {pred_seq[:120]}")
    # print()



output_dir = f"outputs/run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
os.makedirs(output_dir, exist_ok=True)

print("Generating reconstruction examples from 100 random batches...")
reconstruction_file = os.path.join(output_dir, 'sequence_reconstructions.txt')

with open(reconstruction_file, 'w') as f:
    # Convert dataloader to list for random sampling
    all_batches = list(dataloader)

    # Sample 100 random batches
    num_batches = min(100, len(all_batches))
    random_batches = random.sample(all_batches, num_batches)
    
    for batch_idx, (batch_tokens, batch_lengths) in enumerate(tqdm.tqdm(random_batches)):
        # Run reconstruction
        examples = reconstruct_and_decode(model, batch_tokens, batch_lengths, tokenizer_, DEVICE)
        
        # Sample 3 random examples from this batch
        batch_size = len(examples)
        sample_indices = random.sample(range(batch_size), min(3, batch_size))
        
        # Write batch header
        batch_header = f"\n{'='*20} Batch {batch_idx + 1} {'='*20}\n"
        f.write(batch_header)
        print(batch_header, end='')
        
        # Write sampled examples
        for sample_idx, i in enumerate(sample_indices):
            in_seq, pred_seq = examples[i]
            
            example_text = f"---- Example {sample_idx + 1} ----\n"
            example_text += f"Input seq (first 120 bp):  {in_seq[:120]}\n"
            example_text += f"Recon seq (first 120 bp):  {pred_seq[:120]}\n"
            example_text += f"Match rate: {sum(a==b for a,b in zip(in_seq[:120], pred_seq[:120]))/len(in_seq[:120]):.2%}\n\n"
            
            # Write to file
            f.write(example_text)
            # Also print to console
            print(example_text, end='')

print(f"\nReconstruction examples saved to: {reconstruction_file}")

# Clean up wandb
wandb.finish()


