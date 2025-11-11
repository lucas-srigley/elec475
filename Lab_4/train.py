#!/usr/bin/env python3
"""
ELEC 475 Lab 4: Fine-tuning ResNet50 for CLIP
Complete Training Script

Usage:
    python train.py

Requirements:
    - Kaggle API credentials in ~/.kaggle/kaggle.json
    - GPU enabled in Colab
    - ~25GB disk space for dataset
"""

import os
import sys
import json
import time
import subprocess

# Install dependencies
print("Installing dependencies...")
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
                      "transformers", "ftfy", "regex", "tqdm"])

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from transformers import CLIPTextModel, CLIPTokenizer
from PIL import Image
import numpy as np
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

# Set seeds
torch.manual_seed(42)
np.random.seed(42)

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\nDevice: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
else:
    print("WARNING: No GPU detected! Training will be very slow.")

# ============================================================================
# CONFIGURATION
# ============================================================================

CONFIG = {
    'batch_size': 32,
    'num_epochs': 10,
    'learning_rate': 1e-4,
    'embedding_dim': 512,
    'temperature': 0.07,
    'num_workers': 2,
    'eval_samples': 1000,  # Number of samples for evaluation
}

# Paths
TRAIN_IMG_DIR = "coco_data/train2014"
VAL_IMG_DIR = "coco_data/val2014"
TRAIN_CAPTION_FILE = "coco_data/annotations/captions_train2014.json"
VAL_CAPTION_FILE = "coco_data/annotations/captions_val2014.json"

# CLIP constants
CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]

print("\nConfiguration:")
for k, v in CONFIG.items():
    print(f"  {k}: {v}")

# ============================================================================
# DATASET DOWNLOAD
# ============================================================================

def check_kaggle():
    """Check if Kaggle API is configured."""
    kaggle_json = os.path.expanduser('~/.kaggle/kaggle.json')
    if not os.path.exists(kaggle_json):
        print("\n" + "="*70)
        print("ERROR: Kaggle API not configured!")
        print("="*70)
        print("\nPlease run these commands in your Colab notebook first:")
        print("\n  from google.colab import files")
        print("  uploaded = files.upload()  # Upload kaggle.json")
        print("  !mkdir -p ~/.kaggle")
        print("  !cp kaggle.json ~/.kaggle/")
        print("  !chmod 600 ~/.kaggle/kaggle.json")
        print("\nThen run this script again.")
        print("="*70)
        sys.exit(1)
    print("✓ Kaggle API configured")

def download_dataset():
    """Download COCO dataset if not present."""
    if os.path.exists("coco_data") and os.path.exists(TRAIN_IMG_DIR):
        print("✓ Dataset already downloaded")
        return
    
    print("\nDownloading COCO 2014 dataset...")
    print("This will take 15-20 minutes and use ~25GB disk space")
    
    os.system("kaggle datasets download -d jeffaudi/coco-2014-dataset-for-yolov3")
    
    print("\nExtracting dataset...")
    os.system("unzip -q coco-2014-dataset-for-yolov3.zip -d coco_data")
    
    if os.path.exists("coco-2014-dataset-for-yolov3.zip"):
        os.remove("coco-2014-dataset-for-yolov3.zip")
    
    print("✓ Dataset ready")

# ============================================================================
# DATASET CLASS
# ============================================================================

class COCODataset(Dataset):
    """COCO Dataset for CLIP training."""
    
    def __init__(self, img_dir, caption_file, transform=None, text_cache=None):
        self.img_dir = img_dir
        self.transform = transform
        self.text_cache = text_cache
        
        with open(caption_file, 'r') as f:
            coco_data = json.load(f)
        
        self.images = {img['id']: img['file_name'] for img in coco_data['images']}
        self.annotations = [ann for ann in coco_data['annotations']
                           if ann['image_id'] in self.images]
        
        print(f"  Loaded {len(self.annotations)} pairs")
    
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, idx):
        ann = self.annotations[idx]
        img_path = os.path.join(self.img_dir, self.images[ann['image_id']])
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        if self.text_cache is not None:
            return image, self.text_cache[idx], idx
        return image, ann['caption'], idx

# ============================================================================
# TEXT EMBEDDING CACHE
# ============================================================================

def create_text_cache(caption_file, tokenizer, text_encoder, device, cache_file):
    """Pre-encode all captions."""
    
    if os.path.exists(cache_file):
        print(f"  Loading cache: {cache_file}")
        return torch.load(cache_file)
    
    print(f"  Creating cache: {cache_file}")
    
    with open(caption_file, 'r') as f:
        captions = [ann['caption'] for ann in json.load(f)['annotations']]
    
    all_embeddings = []
    batch_size = 256
    
    text_encoder.eval()
    with torch.no_grad():
        for i in tqdm(range(0, len(captions), batch_size), desc="  Encoding"):
            batch = captions[i:i+batch_size]
            inputs = tokenizer(batch, padding=True, truncation=True,
                             max_length=77, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            outputs = text_encoder(**inputs)
            embeddings = F.normalize(outputs.pooler_output, p=2, dim=1)
            all_embeddings.append(embeddings.cpu())
    
    all_embeddings = torch.cat(all_embeddings, dim=0)
    torch.save(all_embeddings, cache_file)
    print(f"  Saved: {cache_file} | Shape: {all_embeddings.shape}")
    
    return all_embeddings

# ============================================================================
# MODEL
# ============================================================================

class CLIPImageEncoder(nn.Module):
    """ResNet50 image encoder with projection head."""
    
    def __init__(self, embedding_dim=512):
        super().__init__()
        resnet = models.resnet50(pretrained=True)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.projection = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.GELU(),
            nn.Linear(1024, embedding_dim)
        )
    
    def forward(self, x):
        features = self.backbone(x).squeeze(-1).squeeze(-1)
        embeddings = self.projection(features)
        return F.normalize(embeddings, p=2, dim=1)

# ============================================================================
# LOSS
# ============================================================================

class InfoNCELoss(nn.Module):
    """InfoNCE contrastive loss for CLIP."""
    
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature
        self.ce_loss = nn.CrossEntropyLoss()
    
    def forward(self, image_emb, text_emb):
        image_emb = F.normalize(image_emb, p=2, dim=1)
        text_emb = F.normalize(text_emb, p=2, dim=1)
        logits = torch.matmul(image_emb, text_emb.t()) / self.temperature
        labels = torch.arange(len(image_emb), device=image_emb.device)
        loss = (self.ce_loss(logits, labels) + self.ce_loss(logits.t(), labels)) / 2
        return loss

# ============================================================================
# TRAINING
# ============================================================================

def train_epoch(model, loader, criterion, optimizer, device, epoch):
    """Train one epoch."""
    model.train()
    total_loss = 0
    
    pbar = tqdm(loader, desc=f"Epoch {epoch}/{CONFIG['num_epochs']}")
    for images, text_emb, _ in pbar:
        images, text_emb = images.to(device), text_emb.to(device)
        
        image_emb = model(images)
        loss = criterion(image_emb, text_emb)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    return total_loss / len(loader)

def validate(model, loader, criterion, device):
    """Validate the model."""
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for images, text_emb, _ in tqdm(loader, desc="Validating", leave=False):
            images, text_emb = images.to(device), text_emb.to(device)
            image_emb = model(images)
            loss = criterion(image_emb, text_emb)
            total_loss += loss.item()
    
    return total_loss / len(loader)

# ============================================================================
# EVALUATION
# ============================================================================

def compute_recall(sim_matrix, k_values=[1, 5, 10]):
    """Compute Recall@K metrics."""
    n = sim_matrix.shape[0]
    
    i2t_ranks = [(sim_matrix[i].argsort(descending=True) == i).nonzero()[0].item()
                 for i in range(n)]
    t2i_ranks = [(sim_matrix[:, i].argsort(descending=True) == i).nonzero()[0].item()
                 for i in range(n)]
    
    results = {}
    for k in k_values:
        results[f'I2T_R@{k}'] = sum(r < k for r in i2t_ranks) / n
        results[f'T2I_R@{k}'] = sum(r < k for r in t2i_ranks) / n
    
    return results

def evaluate_retrieval(model, loader, device, n_samples=1000):
    """Evaluate retrieval performance."""
    model.eval()
    
    img_embs, txt_embs = [], []
    count = 0
    
    with torch.no_grad():
        for images, text_emb, _ in tqdm(loader, desc="Evaluating", leave=False):
            if count >= n_samples:
                break
            images = images.to(device)
            img_embs.append(model(images).cpu())
            txt_embs.append(text_emb)
            count += images.size(0)
    
    img_embs = torch.cat(img_embs)[:n_samples]
    txt_embs = torch.cat(txt_embs)[:n_samples]
    sim_matrix = torch.matmul(img_embs, txt_embs.t())
    
    return compute_recall(sim_matrix)

# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_loss_curves(train_losses, val_losses):
    """Plot and save loss curves."""
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, 'o-', label='Train Loss', linewidth=2)
    plt.plot(val_losses, 's-', label='Val Loss', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Training and Validation Loss', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.savefig('loss_curves.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: loss_curves.png")

def visualize_sample(dataset):
    """Visualize a random sample."""
    idx = np.random.randint(len(dataset))
    img, _, _ = dataset[idx]
    
    img = img.permute(1, 2, 0).numpy()
    img = img * np.array(CLIP_STD) + np.array(CLIP_MEAN)
    img = np.clip(img, 0, 1)
    
    plt.figure(figsize=(5, 5))
    plt.imshow(img)
    plt.title("Sample Image from Dataset")
    plt.axis('off')
    plt.savefig('sample_image.png', bbox_inches='tight', dpi=150)
    plt.close()
    print("✓ Saved: sample_image.png")

# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main training pipeline."""
    
    print("\n" + "="*70)
    print("ELEC 475 LAB 4: CLIP TRAINING")
    print("="*70 + "\n")
    
    # Check Kaggle and download dataset
    check_kaggle()
    download_dataset()
    
    # Load CLIP text encoder
    print("\nLoading CLIP text encoder...")
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")
    text_encoder = text_encoder.to(device).eval()
    
    for param in text_encoder.parameters():
        param.requires_grad = False
    print("✓ Text encoder loaded and frozen")
    
    # Create text caches
    print("\nCreating text embedding caches...")
    train_cache = create_text_cache(TRAIN_CAPTION_FILE, tokenizer,
                                    text_encoder, device, "train_text_cache.pt")
    val_cache = create_text_cache(VAL_CAPTION_FILE, tokenizer,
                                  text_encoder, device, "val_text_cache.pt")
    
    # Prepare datasets
    print("\nPreparing datasets...")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=CLIP_MEAN, std=CLIP_STD)
    ])
    
    train_dataset = COCODataset(TRAIN_IMG_DIR, TRAIN_CAPTION_FILE,
                               transform, train_cache)
    val_dataset = COCODataset(VAL_IMG_DIR, VAL_CAPTION_FILE,
                             transform, val_cache)
    
    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'],
                             shuffle=True, num_workers=CONFIG['num_workers'],
                             pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'],
                           shuffle=False, num_workers=CONFIG['num_workers'],
                           pin_memory=True)
    
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")
    
    # Visualize sample
    print("\nGenerating sample visualization...")
    visualize_sample(train_dataset)
    
    # Initialize model
    print("\nInitializing model...")
    model = CLIPImageEncoder(CONFIG['embedding_dim']).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters: {n_params:,}")
    print(f"  Trainable parameters: {n_trainable:,}")
    
    # Loss and optimizer
    criterion = InfoNCELoss(CONFIG['temperature'])
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG['learning_rate'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=CONFIG['num_epochs']
    )
    
    # Training loop
    print("\n" + "="*70)
    print("STARTING TRAINING")
    print("="*70 + "\n")
    
    train_losses, val_losses = [], []
    start_time = time.time()
    
    for epoch in range(1, CONFIG['num_epochs'] + 1):
        print(f"\nEpoch {epoch}/{CONFIG['num_epochs']}")
        print("-" * 70)
        
        train_loss = train_epoch(model, train_loader, criterion,
                                optimizer, device, epoch)
        val_loss = validate(model, val_loader, criterion, device)
        scheduler.step()
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        
        # Save checkpoint
        if epoch % 2 == 0 or epoch == CONFIG['num_epochs']:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'config': CONFIG
            }
            torch.save(checkpoint, f'checkpoint_epoch_{epoch}.pt')
            print(f"✓ Saved: checkpoint_epoch_{epoch}.pt")
    
    training_time = time.time() - start_time
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE!")
    print(f"Total time: {training_time/3600:.2f} hours")
    print("="*70)
    
    # Save final model
    torch.save(model.state_dict(), 'clip_resnet50_final.pt')
    print("\n✓ Saved: clip_resnet50_final.pt")
    
    # Plot loss curves
    print("\nGenerating loss curves...")
    plot_loss_curves(train_losses, val_losses)
    
    # Evaluate
    print("\n" + "="*70)
    print("EVALUATING RETRIEVAL PERFORMANCE")
    print("="*70 + "\n")
    
    metrics = evaluate_retrieval(model, val_loader, device,
                                n_samples=CONFIG['eval_samples'])
    
    print(f"Recall Metrics ({CONFIG['eval_samples']} validation samples):")
    print("-" * 70)
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.4f} ({value*100:.2f}%)")
    
    # Save metrics
    with open('training_results.txt', 'w') as f:
        f.write("="*70 + "\n")
        f.write("ELEC 475 Lab 4: Training Results\n")
        f.write("="*70 + "\n\n")
        f.write("Configuration:\n")
        for k, v in CONFIG.items():
            f.write(f"  {k}: {v}\n")
        f.write(f"\nTraining Time: {training_time/3600:.2f} hours\n")
        f.write(f"Final Train Loss: {train_losses[-1]:.4f}\n")
        f.write(f"Final Val Loss: {val_losses[-1]:.4f}\n\n")
        f.write("Recall Metrics:\n")
        for metric, value in metrics.items():
            f.write(f"  {metric}: {value:.4f} ({value*100:.2f}%)\n")
    
    print("\n✓ Saved: training_results.txt")
    
    print("\n" + "="*70)
    print("LAB 4 TRAINING COMPLETE!")
    print("="*70)
    print("\nGenerated files:")
    print("  - clip_resnet50_final.pt")
    print("  - checkpoint_epoch_*.pt")
    print("  - loss_curves.png")
    print("  - sample_image.png")
    print("  - training_results.txt")
    print("  - train_text_cache.pt, val_text_cache.pt")
    print("\nNext step: Run test.py for visualization and evaluation")

if __name__ == "__main__":
    main()