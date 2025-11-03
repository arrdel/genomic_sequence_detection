# Git Setup Guide

## Quick Start - Push to GitHub

### 1. Initialize Git Repository (if not already done)
```bash
cd /home/adelechinda/home/semester_projects/fall_25/deep_learning/project
git init
```

### 2. Add All Files (respecting .gitignore)
```bash
git add .
```

### 3. Create Initial Commit
```bash
git commit -m "Initial commit: VQ-VAE for genomics project structure"
```

### 4. Create GitHub Repository
Go to https://github.com/new and create a new repository (e.g., `vqvae-genomics`)

**Important**: Do NOT initialize with README, .gitignore, or license (we already have them)

### 5. Add Remote and Push
```bash
# Replace YOUR_USERNAME and YOUR_REPO with your actual GitHub username and repo name
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO.git

# Push to GitHub
git branch -M main
git push -u origin main
```

---

## What's Excluded (Won't Be Pushed)

The `.gitignore` file excludes:

### Large Files
- ✅ Checkpoints (*.pt, *.pth, checkpoints/)
- ✅ Datasets (*.fastq, *.fasta, *.fa, etc.)
- ✅ K-mer batches (*.npy)
- ✅ Training outputs (outputs/, logs/)
- ✅ Wandb runs (wandb/)

### Generated Files
- ✅ Python cache (__pycache__/)
- ✅ FastQC reports (*_fastqc.html, *_fastqc.zip)
- ✅ Preprocessed data (cleaned_reads*)
- ✅ Evaluation results (evaluation_results/)

### System Files
- ✅ OS-specific files (.DS_Store, Thumbs.db)
- ✅ IDE configs (.vscode/, .idea/)
- ✅ Virtual environments (venv/, env/)

---

## What WILL Be Pushed (Source Code Only)

✓ `src/` - All source code
✓ `scripts/` - Training, evaluation, preprocessing scripts
✓ `configs/` - Configuration files
✓ `requirements.txt` - Dependencies
✓ `README.md` - Documentation
✓ `.gitignore` - Git ignore rules

---

## Verify Before Pushing

Check what will be committed:
```bash
git status
```

Check what's ignored:
```bash
git status --ignored
```

See file sizes to ensure no large files:
```bash
git ls-files | xargs ls -lh | sort -k5 -hr | head -20
```

---

## Useful Git Commands

### Check current status
```bash
git status
```

### Add specific files
```bash
git add src/
git add scripts/
git add README.md
```

### Commit changes
```bash
git commit -m "Your commit message"
```

### Push to remote
```bash
git push
```

### Pull latest changes
```bash
git pull
```

### Create a new branch
```bash
git checkout -b feature-branch-name
```

### See commit history
```bash
git log --oneline
```

---

## Storing Large Files (Optional)

If you need to share checkpoints or datasets:

### Option 1: Git LFS (Large File Storage)
```bash
# Install Git LFS
git lfs install

# Track specific file types
git lfs track "*.pt"
git lfs track "*.fastq"

# Commit the .gitattributes file
git add .gitattributes
git commit -m "Add Git LFS tracking"
```

### Option 2: External Storage
- Store checkpoints on: Google Drive, Dropbox, OneDrive
- Store datasets on: Institutional storage, AWS S3, Zenodo
- Add download links in README.md

### Option 3: GitHub Releases
For trained models, use GitHub Releases:
1. Go to your repo → Releases → Create new release
2. Upload checkpoint files as release assets
3. Link to them in your README

---

## Recommended Workflow

### Daily Work
```bash
# Pull latest changes
git pull

# Make changes to code
# ...

# Add and commit
git add .
git commit -m "Description of changes"
git push
```

### Before Major Experiments
```bash
# Create a branch for experiments
git checkout -b experiment-large-model
# Make changes, commit, push
git push -u origin experiment-large-model
```

### After Successful Experiment
```bash
# Merge back to main
git checkout main
git merge experiment-large-model
git push
```

---

## Troubleshooting

### Large files accidentally added
```bash
# Remove from staging
git rm --cached path/to/large/file

# Or remove all checkpoints
git rm --cached -r checkpoints/

# Commit the removal
git commit -m "Remove large files"
```

### Already pushed large files
```bash
# Use BFG Repo Cleaner or git filter-branch
# See: https://github.com/rtyley/bfg-repo-cleaner
```

### Check repository size
```bash
git count-objects -vH
```

---

## Best Practices

1. ✅ Commit often with clear messages
2. ✅ Never commit sensitive data (API keys, passwords)
3. ✅ Never commit large data files or checkpoints
4. ✅ Use branches for experiments
5. ✅ Keep main branch stable
6. ✅ Update README with results and instructions
7. ✅ Tag releases for reproducibility
8. ✅ Document hyperparameters in configs

---

## Example Commit Messages

Good commit messages:
```
✓ "Add multi-GPU training support"
✓ "Fix checkpoint loading for DataParallel models"
✓ "Update evaluation metrics to include codebook usage"
✓ "Reorganize project structure"
```

Bad commit messages:
```
✗ "updates"
✗ "fix"
✗ "changes"
✗ "asdfasdf"
```
