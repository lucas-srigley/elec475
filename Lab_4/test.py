#!/usr/bin/env python3
"""
ELEC 475 Lab 4: Testing and Visualization Script

Usage:
    python test.py

Requirements:
    - Trained model: clip_resnet50_final.pt
    - Text caches: val_text_cache.pt
    - Dataset: coco_data/
"""

import os
import sys
import json
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
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Set seeds
torch.manual_seed(42)
np.random.seed(42)

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# ============================================================================
# CONFIGURATION
# ============================================================================

# Paths
MODEL_PATH = 'clip_resnet50_final.pt'
VAL_IMG_DIR = "coco_data/val2014"
VAL_CAPTION_FILE = "coco_data/annotations/captions_val2014.json"
VAL_TEXT_CACHE = "val_text_cache.pt"

# Constants
CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]
EMBEDDING_DIM = 512
BATCH_SIZE = 32

# ============================================================================
# CHECK FILES
# ============================================================================

def check_files():
    """Check if all required files exist."""
    required_files = [
        MODEL_PATH,
        VAL_TEXT_CACHE,
        VAL_IMG_DIR,
        VAL_CAPTION_FILE
    ]
    
    missing = []
    for f in required_files:
        if not os.path.exists(f):
            missing.append(f)
    
    if missing:
        print("\n" + "="*70)
        print("ERROR: Missing required files!")
        print("="*70)
        for f in missing:
            print(f"  - {f}")
        print("\nPlease run train.py first to generate these files.")
        print("="*70)
        sys.exit(1)
    
    print("✓ All required files found")

# ============================================================================
# DATASET CLASS
# ============================================================================

class COCODataset(Dataset):
    """COCO Dataset for evaluation."""
    
    def __init__(self, img_dir, caption_file, transform=None, text_cache=None):
        self.img_dir = img_dir
        self.transform = transform
        self.text_cache = text_cache
        
        with open(caption_file, 'r') as f:
            coco_data = json.load(f)
        
        self.images = {img['id']: img['file_name'] for img in coco_data['images']}
        self.annotations = [ann for ann in coco_data['annotations']
                           if ann['image_id'] in self.images]
    
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, idx):
        ann = self.annotations[idx]
        img_path = os.path.join(self.img_dir, self.images[ann['image_id']])
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        if self.text_cache is not None:
            return image, self.text_cache[idx], idx, ann['caption']
        return image, ann['caption'], idx, ann['caption']

# ============================================================================
# MODEL
# ============================================================================

class CLIPImageEncoder(nn.Module):
    """ResNet50 image encoder."""
    
    def __init__(self, embedding_dim=512):
        super().__init__()
        resnet = models.resnet50(pretrained=False)
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
        for images, text_emb, _, _ in tqdm(loader, desc="Computing embeddings"):
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

def text_to_image_retrieval(query_text, model, dataset, tokenizer, 
                           text_encoder, device, top_k=5, search_size=1000):
    """Retrieve top-K images for a text query."""
    model.eval()
    
    print(f"\n  Query: '{query_text}'")
    
    # Encode query
    with torch.no_grad():
        inputs = tokenizer([query_text], padding=True, truncation=True,
                          max_length=77, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        outputs = text_encoder(**inputs)
        query_emb = F.normalize(outputs.pooler_output, p=2, dim=1)
    
    # Compute similarities
    similarities = []
    search_size = min(search_size, len(dataset))
    
    with torch.no_grad():
        for idx in tqdm(range(search_size), desc="  Searching", leave=False):
            image, _, _, _ = dataset[idx]
            image = image.unsqueeze(0).to(device)
            img_emb = model(image)
            sim = torch.matmul(query_emb, img_emb.t()).item()
            similarities.append(sim)
    
    # Get top-K
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    
    # Visualize
    fig, axes = plt.subplots(1, top_k, figsize=(15, 3))
    if top_k == 1:
        axes = [axes]
    
    for i, idx in enumerate(top_indices):
        image, _, _, _ = dataset[idx]
        image = image.permute(1, 2, 0).numpy()
        image = image * np.array(CLIP_STD) + np.array(CLIP_MEAN)
        image = np.clip(image, 0, 1)
        
        axes[i].imshow(image)
        axes[i].set_title(f"Sim: {similarities[idx]:.3f}", fontsize=10)
        axes[i].axis('off')
    
    plt.suptitle(f"Text Query: '{query_text}'", fontsize=12, y=0.98)
    plt.tight_layout()
    filename = f"retrieval_{query_text.replace(' ', '_')}.png"
    plt.savefig(filename, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {filename}")

def zero_shot_classification(image_idx, class_labels, model, dataset,
                            tokenizer, text_encoder, device):
    """Zero-shot image classification."""
    model.eval()
    
    print(f"\n  Image index: {image_idx}")
    
    # Get image
    image, _, _, caption = dataset[image_idx]
    image_display = image.permute(1, 2, 0).numpy()
    image_display = image_display * np.array(CLIP_STD) + np.array(CLIP_MEAN)
    image_display = np.clip(image_display, 0, 1)
    
    # Encode image
    with torch.no_grad():
        image_tensor = image.unsqueeze(0).to(device)
        img_emb = model(image_tensor)
    
    # Encode class labels
    with torch.no_grad():
        inputs = tokenizer(class_labels, padding=True, truncation=True,
                          max_length=77, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        outputs = text_encoder(**inputs)
        text_embs = F.normalize(outputs.pooler_output, p=2, dim=1)
    
    # Compute similarities
    similarities = torch.matmul(img_emb, text_embs.t()).squeeze(0)
    similarities = similarities.cpu().numpy()
    
    # Get prediction
    pred_idx = np.argmax(similarities)
    pred_class = class_labels[pred_idx]
    
    # Visualize
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    ax1.imshow(image_display)
    ax1.set_title(f"Predicted: {pred_class}", fontsize=12, fontweight='bold')
    ax1.axis('off')
    
    colors = ['green' if i == pred_idx else 'steelblue' 
              for i in range(len(class_labels))]
    bars = ax2.barh(class_labels, similarities, color=colors)
    ax2.set_xlabel("Cosine Similarity", fontsize=11)
    ax2.set_title("Class Probabilities", fontsize=12)
    ax2.set_xlim(0, 1)
    
    # Add value labels
    for i, (bar, sim) in enumerate(zip(bars, similarities)):
        ax2.text(sim + 0.02, bar.get_y() + bar.get_height()/2,
                f'{sim:.3f}', va='center', fontsize=9)
    
    plt.tight_layout()
    filename = f"classification_{image_idx}.png"
    plt.savefig(filename, dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"  Caption: \"{caption}\"")
    print(f"  Predicted: {pred_class} (similarity: {similarities[pred_idx]:.3f})")
    print(f"  ✓ Saved: {filename}")
    
    return pred_class, similarities

# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main evaluation pipeline."""
    
    print("\n" + "="*70)
    print("ELEC 475 LAB 4: MODEL TESTING & VISUALIZATION")
    print("="*70 + "\n")
    
    # Check files
    check_files()
    
    # Load model
    print("\nLoading trained model...")
    model = CLIPImageEncoder(EMBEDDING_DIM).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    print(f"✓ Model loaded from: {MODEL_PATH}")
    
    # Load CLIP text encoder
    print("\nLoading CLIP text encoder...")
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")
    text_encoder = text_encoder.to(device).eval()
    print("✓ Text encoder loaded")
    
    # Load text cache
    print("\nLoading text embeddings cache...")
    val_text_cache = torch.load(VAL_TEXT_CACHE)
    print(f"✓ Cache loaded: {val_text_cache.shape}")
    
    # Prepare dataset
    print("\nPreparing validation dataset...")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=CLIP_MEAN, std=CLIP_STD)
    ])
    
    val_dataset = COCODataset(VAL_IMG_DIR, VAL_CAPTION_FILE,
                             transform, val_text_cache)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE,
                           shuffle=False, num_workers=2, pin_memory=True)
    
    print(f"✓ Dataset ready: {len(val_dataset)} samples")
    
    # Evaluate retrieval
    print("\n" + "="*70)
    print("RETRIEVAL EVALUATION")
    print("="*70)
    
    metrics = evaluate_retrieval(model, val_loader, device, n_samples=1000)
    
    print("\nRecall Metrics (1000 validation samples):")
    print("-" * 70)
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.4f} ({value*100:.2f}%)")
    
    # Save metrics
    with open('test_results.txt', 'w') as f:
        f.write("="*70 + "\n")
        f.write("ELEC 475 Lab 4: Test Results\n")
        f.write("="*70 + "\n\n")
        f.write("Recall Metrics (1000 validation samples):\n")
        for metric, value in metrics.items():
            f.write(f"  {metric}: {value:.4f} ({value*100:.2f}%)\n")
    print("\n✓ Saved: test_results.txt")
    
    # Text-to-Image retrieval
    print("\n" + "="*70)
    print("TEXT-TO-IMAGE RETRIEVAL")
    print("="*70)
    
    queries = ['sport', 'animal', 'food', 'car', 'person', 'beach', 'dog', 'pizza']
    for query in queries:
        text_to_image_retrieval(query, model, val_dataset, tokenizer,
                               text_encoder, device, top_k=5, search_size=1000)
    
    # Zero-shot classification
    print("\n" + "="*70)
    print("ZERO-SHOT CLASSIFICATION")
    print("="*70)
    
    class_labels = ['a person', 'an animal', 'a landscape', 'a vehicle', 'food']
    
    # Test on specific indices for reproducibility
    test_indices = [10, 50, 100, 200, 500]
    
    for idx in test_indices:
        zero_shot_classification(idx, class_labels, model, val_dataset,
                                tokenizer, text_encoder, device)
    
    print("\n" + "="*70)
    print("TESTING COMPLETE!")
    print("="*70)
    print("\nGenerated files:")
    print("  - test_results.txt")
    print("  - retrieval_*.png (8 files)")
    print("  - classification_*.png (5 files)")
    print("\nUse these images in your lab report!")

if __name__ == "__main__":
    main()