import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from transformers import CLIPTokenizer, CLIPTextModel
from PIL import Image
import json
import os
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import gc

# ==================== Configuration ====================
class Config:
    # Paths - UPDATE THESE TO MATCH YOUR ACTUAL FOLDER STRUCTURE
    # The paths where your actual images are located
    TRAIN_IMG_DIR = "/content/drive/MyDrive/ELEC475_Lab_4/coco2014/train2014"  # UPDATE THIS
    VAL_IMG_DIR = "/content/drive/MyDrive/ELEC475_Lab_4/coco2014/val2014"      # UPDATE THIS
    TRAIN_CAPTION_FILE = "/content/drive/MyDrive/ELEC475_Lab_4/coco2014/annotations/captions_train2014.json"
    VAL_CAPTION_FILE = "/content/drive/MyDrive/ELEC475_Lab_4/coco2014/annotations/captions_val2014.json"
    
    # Cache directories for preprocessed embeddings
    CACHE_DIR = "/content/drive/MyDrive/ELEC475_Lab_4/cache"
    
    # Model parameters
    CLIP_MODEL = "openai/clip-vit-base-patch32"
    EMBEDDING_DIM = 512
    IMAGE_SIZE = 224
    
    # CLIP normalization values
    CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
    CLIP_STD = [0.26862954, 0.26130258, 0.27577711]
    
    # Training parameters
    BATCH_SIZE = 64  # Adjust based on GPU memory
    NUM_EPOCHS = 10
    LEARNING_RATE = 1e-4
    WARMUP_EPOCHS = 1
    
    # Dataset subset (set to None to use full dataset)
    TRAIN_SUBSET_SIZE = 20000  # Use subset for faster training
    VAL_SUBSET_SIZE = 2000
    
    # Device
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Checkpoint
    CHECKPOINT_DIR = "/content/drive/MyDrive/ELEC475_Lab_4/checkpoints"
    SAVE_EVERY = 2  # Save checkpoint every N epochs

# ==================== Dataset Preprocessing ====================
def load_coco_annotations(caption_file, subset_size=None):
    """Load COCO captions and return list of (image_id, caption) pairs"""
    with open(caption_file, 'r') as f:
        data = json.load(f)
    
    # Create mapping from image_id to captions
    image_to_captions = {}
    for ann in data['annotations']:
        image_id = ann['image_id']
        if image_id not in image_to_captions:
            image_to_captions[image_id] = []
        image_to_captions[image_id].append(ann['caption'])
    
    # Create image_id to filename mapping
    image_info = {img['id']: img['file_name'] for img in data['images']}
    
    # Create dataset entries
    dataset_entries = []
    for image_id, captions in image_to_captions.items():
        if image_id in image_info:
            # Use first caption for training
            dataset_entries.append({
                'image_id': image_id,
                'filename': image_info[image_id],
                'caption': captions[0],
                'all_captions': captions
            })
    
    # Apply subset if specified
    if subset_size is not None and subset_size < len(dataset_entries):
        np.random.seed(42)
        indices = np.random.choice(len(dataset_entries), subset_size, replace=False)
        dataset_entries = [dataset_entries[i] for i in indices]
    
    return dataset_entries

def cache_text_embeddings(config):
    """Pre-compute and cache text embeddings to save time during training"""
    print("Loading CLIP text encoder...")
    tokenizer = CLIPTokenizer.from_pretrained(config.CLIP_MODEL)
    text_encoder = CLIPTextModel.from_pretrained(config.CLIP_MODEL).to(config.DEVICE)
    text_encoder.eval()
    
    os.makedirs(config.CACHE_DIR, exist_ok=True)
    
    for split, caption_file, subset_size in [
        ('train', config.TRAIN_CAPTION_FILE, config.TRAIN_SUBSET_SIZE),
        ('val', config.VAL_CAPTION_FILE, config.VAL_SUBSET_SIZE)
    ]:
        cache_file = os.path.join(config.CACHE_DIR, f'{split}_text_embeddings.pt')
        
        if os.path.exists(cache_file):
            print(f"Cache file for {split} already exists. Skipping...")
            continue
        
        print(f"\nProcessing {split} captions...")
        entries = load_coco_annotations(caption_file, subset_size)
        
        text_embeddings = {}
        with torch.no_grad():
            for entry in tqdm(entries, desc=f"Encoding {split} captions"):
                caption = entry['caption']
                image_id = entry['image_id']
                
                # Tokenize
                inputs = tokenizer(
                    caption,
                    padding='max_length',
                    max_length=77,
                    truncation=True,
                    return_tensors='pt'
                ).to(config.DEVICE)
                
                # Encode
                outputs = text_encoder(**inputs)
                text_embedding = outputs.pooler_output.cpu()
                
                text_embeddings[image_id] = {
                    'embedding': text_embedding,
                    'caption': caption,
                    'filename': entry['filename']
                }
        
        # Save cache
        torch.save(text_embeddings, cache_file)
        print(f"Saved {len(text_embeddings)} text embeddings to {cache_file}")
        
        # Clean up
        del text_embeddings
        gc.collect()
        torch.cuda.empty_cache()
    
    del tokenizer, text_encoder
    gc.collect()
    torch.cuda.empty_cache()

# ==================== Dataset Class ====================
class COCOCLIPDataset(Dataset):
    def __init__(self, image_dir, cache_file, transform=None, verify_images=True):
        self.image_dir = image_dir
        self.transform = transform
        
        # Load cached embeddings
        print(f"Loading cached embeddings from {cache_file}...")
        self.data = torch.load(cache_file)
        all_image_ids = list(self.data.keys())
        
        # Verify which images actually exist
        if verify_images:
            print("Verifying image files exist...")
            self.image_ids = []
            missing_count = 0
            
            for image_id in tqdm(all_image_ids, desc="Checking images"):
                entry = self.data[image_id]
                filename = entry['filename']
                
                # Handle path separators
                if '/' in filename or '\\' in filename:
                    filename = os.path.basename(filename)
                
                image_path = os.path.join(self.image_dir, filename)
                
                if os.path.exists(image_path):
                    self.image_ids.append(image_id)
                else:
                    missing_count += 1
            
            print(f"✓ Found {len(self.image_ids)} valid images")
            if missing_count > 0:
                print(f"⚠ Skipped {missing_count} missing images")
        else:
            self.image_ids = all_image_ids
            print(f"Loaded {len(self.image_ids)} samples (no verification)")
    
    def __len__(self):
        return len(self.image_ids)
    
    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        entry = self.data[image_id]
        
        # Get filename - handle cases where filename might have path components
        filename = entry['filename']
        
        # If filename contains path separators, extract just the filename
        if '/' in filename or '\\' in filename:
            filename = os.path.basename(filename)
        
        # Construct full path
        image_path = os.path.join(self.image_dir, filename)
        
        # Load image
        image = Image.open(image_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        text_embedding = entry['embedding'].squeeze(0)
        
        return image, text_embedding, image_id

# ==================== Model Architecture ====================
class ImageEncoder(nn.Module):
    def __init__(self, embedding_dim=512):
        super().__init__()
        # Load pretrained ResNet50
        resnet = models.resnet50(pretrained=True)
        
        # Remove the final classification layer
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        
        # Projection head (2048 -> 512 -> 512)
        self.projection = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.GELU(),
            nn.Linear(1024, embedding_dim)
        )
    
    def forward(self, x):
        # Extract features
        features = self.backbone(x)
        features = features.view(features.size(0), -1)
        
        # Project to embedding space
        embeddings = self.projection(features)
        
        # L2 normalize
        embeddings = F.normalize(embeddings, p=2, dim=1)
        
        return embeddings

class CLIPModel(nn.Module):
    def __init__(self, embedding_dim=512):
        super().__init__()
        self.image_encoder = ImageEncoder(embedding_dim)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
    
    def forward(self, images, text_embeddings):
        # Encode images
        image_embeddings = self.image_encoder(images)
        
        # Normalize text embeddings (they're already pre-computed)
        text_embeddings = F.normalize(text_embeddings, p=2, dim=1)
        
        # Compute logit scale
        logit_scale = self.logit_scale.exp()
        
        return image_embeddings, text_embeddings, logit_scale

# ==================== Loss Function ====================
def clip_loss(image_embeddings, text_embeddings, logit_scale):
    """
    InfoNCE loss for CLIP
    
    For each image-text pair in the batch:
    - Positive: the paired text
    - Negatives: all other texts in the batch
    """
    # Compute similarity matrix
    logits = logit_scale * image_embeddings @ text_embeddings.T
    
    # Labels are diagonal (each image matches its corresponding text)
    batch_size = image_embeddings.shape[0]
    labels = torch.arange(batch_size, device=image_embeddings.device)
    
    # Symmetric loss: image-to-text and text-to-image
    loss_i2t = F.cross_entropy(logits, labels)
    loss_t2i = F.cross_entropy(logits.T, labels)
    
    loss = (loss_i2t + loss_t2i) / 2
    
    return loss

# ==================== Training ====================
def train_one_epoch(model, dataloader, optimizer, device, epoch):
    model.train()
    total_loss = 0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for images, text_embeddings, _ in pbar:
        images = images.to(device)
        text_embeddings = text_embeddings.to(device)
        
        # Forward pass
        image_embeddings, text_embeddings, logit_scale = model(images, text_embeddings)
        
        # Compute loss
        loss = clip_loss(image_embeddings, text_embeddings, logit_scale)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        pbar.set_postfix({'loss': loss.item()})
    
    return total_loss / len(dataloader)

@torch.no_grad()
def validate(model, dataloader, device):
    model.eval()
    total_loss = 0
    
    all_image_embeds = []
    all_text_embeds = []
    
    for images, text_embeddings, _ in tqdm(dataloader, desc="Validating"):
        images = images.to(device)
        text_embeddings = text_embeddings.to(device)
        
        # Forward pass
        image_embeddings, text_embeddings, logit_scale = model(images, text_embeddings)
        
        # Compute loss
        loss = clip_loss(image_embeddings, text_embeddings, logit_scale)
        total_loss += loss.item()
        
        # Collect embeddings
        all_image_embeds.append(image_embeddings.cpu())
        all_text_embeds.append(text_embeddings.cpu())
    
    # Concatenate all embeddings
    all_image_embeds = torch.cat(all_image_embeds, dim=0)
    all_text_embeds = torch.cat(all_text_embeds, dim=0)
    
    return total_loss / len(dataloader), all_image_embeds, all_text_embeds

# ==================== Evaluation Metrics ====================
def compute_recall_at_k(similarity_matrix, k_values=[1, 5, 10]):
    """
    Compute Recall@K for both I2T and T2I retrieval
    
    similarity_matrix: (num_images, num_texts) cosine similarity
    """
    results = {}
    
    # Image to Text retrieval
    for k in k_values:
        # Get top-k text indices for each image
        top_k_indices = similarity_matrix.topk(k, dim=1)[1]
        
        # Check if correct text (diagonal) is in top-k
        correct = torch.zeros(similarity_matrix.shape[0], dtype=torch.bool)
        for i in range(similarity_matrix.shape[0]):
            if i in top_k_indices[i]:
                correct[i] = True
        
        recall = correct.float().mean().item()
        results[f'I2T_R@{k}'] = recall
    
    # Text to Image retrieval
    for k in k_values:
        # Get top-k image indices for each text
        top_k_indices = similarity_matrix.T.topk(k, dim=1)[1]
        
        # Check if correct image (diagonal) is in top-k
        correct = torch.zeros(similarity_matrix.shape[0], dtype=torch.bool)
        for i in range(similarity_matrix.shape[0]):
            if i in top_k_indices[i]:
                correct[i] = True
        
        recall = correct.float().mean().item()
        results[f'T2I_R@{k}'] = recall
    
    return results

# ==================== Main Training Script ====================
def main():
    config = Config()
    
    # Create necessary directories
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    
    print(f"Using device: {config.DEVICE}")
    print(f"Training subset size: {config.TRAIN_SUBSET_SIZE}")
    print(f"Validation subset size: {config.VAL_SUBSET_SIZE}")
    
    # Step 1: Cache text embeddings (only needs to be done once)
    print("\n=== Step 1: Caching text embeddings ===")
    cache_text_embeddings(config)
    
    # Step 2: Create datasets
    print("\n=== Step 2: Creating datasets ===")
    transform = transforms.Compose([
        transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.CLIP_MEAN, std=config.CLIP_STD)
    ])
    
    train_dataset = COCOCLIPDataset(
        config.TRAIN_IMG_DIR,
        os.path.join(config.CACHE_DIR, 'train_text_embeddings.pt'),
        transform=transform
    )
    
    val_dataset = COCOCLIPDataset(
        config.VAL_IMG_DIR,
        os.path.join(config.CACHE_DIR, 'val_text_embeddings.pt'),
        transform=transform
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    
    # Step 3: Initialize model
    print("\n=== Step 3: Initializing model ===")
    model = CLIPModel(embedding_dim=config.EMBEDDING_DIM).to(config.DEVICE)
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=0.01
    )
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.NUM_EPOCHS
    )
    
    # Step 4: Training loop
    print("\n=== Step 4: Training ===")
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    for epoch in range(1, config.NUM_EPOCHS + 1):
        print(f"\nEpoch {epoch}/{config.NUM_EPOCHS}")
        
        # Train
        train_loss = train_one_epoch(model, train_loader, optimizer, config.DEVICE, epoch)
        train_losses.append(train_loss)
        
        # Validate
        val_loss, image_embeds, text_embeds = validate(model, val_loader, config.DEVICE)
        val_losses.append(val_loss)
        
        # Compute metrics
        similarity_matrix = image_embeds @ text_embeds.T
        metrics = compute_recall_at_k(similarity_matrix)
        
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_loss:.4f}")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")
        
        # Save checkpoint
        if epoch % config.SAVE_EVERY == 0 or val_loss < best_val_loss:
            checkpoint_path = os.path.join(
                config.CHECKPOINT_DIR,
                f'clip_model_epoch_{epoch}.pt'
            )
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'metrics': metrics
            }, checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_path = os.path.join(config.CHECKPOINT_DIR, 'best_model.pt')
                torch.save(model.state_dict(), best_model_path)
        
        scheduler.step()
    
    # Plot training curves
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    plt.savefig(os.path.join(config.CHECKPOINT_DIR, 'loss_curve.png'))
    plt.close()
    
    print("\n=== Training Complete ===")
    print(f"Best validation loss: {best_val_loss:.4f}")

if __name__ == "__main__":
    main()