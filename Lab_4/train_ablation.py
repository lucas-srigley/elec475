import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import os
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd

# Import from train.py
from train import (
    Config, COCOCLIPDataset, clip_loss, 
    train_one_epoch, validate, compute_recall_at_k, CLIPModel
)

# ==================== Modified Image Encoder ====================
class ImageEncoderModified(nn.Module):
    def __init__(self, embedding_dim=512, use_batch_norm=False, use_dropout=False, dropout_rate=0.2):
        super().__init__()
        # Load pretrained ResNet50
        resnet = models.resnet50(pretrained=True)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        
        # Modified projection head with BatchNorm and Dropout
        layers = [nn.Linear(2048, 1024), nn.GELU()]
        
        if use_batch_norm:
            layers.append(nn.BatchNorm1d(1024))
        if use_dropout:
            layers.append(nn.Dropout(dropout_rate))
        
        layers.append(nn.Linear(1024, embedding_dim))
        
        self.projection = nn.Sequential(*layers)
    
    def forward(self, x):
        features = self.backbone(x)
        features = features.view(features.size(0), -1)
        embeddings = self.projection(features)
        embeddings = F.normalize(embeddings, p=2, dim=1)
        return embeddings

class CLIPModelModified(nn.Module):
    def __init__(self, embedding_dim=512, use_batch_norm=False, use_dropout=False, dropout_rate=0.2):
        super().__init__()
        self.image_encoder = ImageEncoderModified(embedding_dim, use_batch_norm, use_dropout, dropout_rate)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
    
    def forward(self, images, text_embeddings):
        image_embeddings = self.image_encoder(images)
        text_embeddings = F.normalize(text_embeddings, p=2, dim=1)
        logit_scale = self.logit_scale.exp()
        return image_embeddings, text_embeddings, logit_scale

# ==================== Data Augmentation ====================
def get_augmented_transform():
    """Get transform with data augmentation"""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=Config.CLIP_MEAN, std=Config.CLIP_STD)
    ])

def get_standard_transform():
    """Get standard transform without augmentation"""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=Config.CLIP_MEAN, std=Config.CLIP_STD)
    ])

# ==================== Training Function ====================
def train_modification(experiment_name, model, train_loader, val_loader, config, num_epochs=10):
    """Train a model with specific modifications"""
    
    device = config.DEVICE
    model = model.to(device)
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    # Track results
    train_losses = []
    val_losses = []
    all_metrics = []
    best_val_loss = float('inf')
    
    print(f"\n{'='*60}")
    print(f"Training: {experiment_name}")
    print(f"{'='*60}")
    
    for epoch in range(1, num_epochs + 1):
        print(f"\nEpoch {epoch}/{num_epochs}")
        
        # Train
        train_loss = train_one_epoch(model, train_loader, optimizer, device, epoch)
        train_losses.append(train_loss)
        
        # Validate
        val_loss, image_embeds, text_embeds = validate(model, val_loader, device)
        val_losses.append(val_loss)
        
        # Compute metrics
        similarity_matrix = image_embeds @ text_embeds.T
        metrics = compute_recall_at_k(similarity_matrix)
        all_metrics.append(metrics)
        
        print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_dir = Path(config.CHECKPOINT_DIR) / 'ablation' / experiment_name
            save_dir.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), save_dir / 'best_model.pt')
        
        scheduler.step()
    
    return {
        'experiment': experiment_name,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'metrics_history': all_metrics,
        'best_val_loss': best_val_loss,
        'final_metrics': all_metrics[-1]
    }

# ==================== Main Ablation Script ====================
def main():
    config = Config()
    
    print("="*60)
    print("ABLATION STUDY: 2 MODIFICATIONS")
    print("="*60)
    
    # ============================================================
    # EXPERIMENT 1: BASELINE (already trained)
    # ============================================================
    print("\nExperiment 1: Baseline (using existing results)")
    baseline_results = {
        'experiment': 'baseline',
        'best_val_loss': 0.8037,
        'final_metrics': {
            'I2T_R@1': 0.2131,
            'I2T_R@5': 0.4947,
            'I2T_R@10': 0.6568,
            'T2I_R@1': 0.2546,
            'T2I_R@5': 0.5468,
            'T2I_R@10': 0.6853
        }
    }
    
    # ============================================================
    # EXPERIMENT 2: MODIFICATION 1 - Regularization (BatchNorm + Dropout)
    # ============================================================
    print("\n" + "="*60)
    print("Experiment 2: Regularization (BatchNorm + Dropout)")
    print("="*60)
    
    # Load datasets with standard transform
    train_dataset = COCOCLIPDataset(
        config.TRAIN_IMG_DIR,
        os.path.join(config.CACHE_DIR, 'train_text_embeddings.pt'),
        transform=get_standard_transform()
    )
    
    val_dataset = COCOCLIPDataset(
        config.VAL_IMG_DIR,
        os.path.join(config.CACHE_DIR, 'val_text_embeddings.pt'),
        transform=get_standard_transform()
    )
    
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
    
    # Train with BatchNorm + Dropout
    model_reg = CLIPModelModified(
        embedding_dim=config.EMBEDDING_DIM,
        use_batch_norm=True,
        use_dropout=True,
        dropout_rate=0.2
    )
    
    reg_results = train_modification(
        'regularization',
        model_reg,
        train_loader,
        val_loader,
        config,
        num_epochs=config.NUM_EPOCHS
    )
    
    # ============================================================
    # EXPERIMENT 3: MODIFICATION 2 - Data Augmentation
    # ============================================================
    print("\n" + "="*60)
    print("Experiment 3: Data Augmentation")
    print("="*60)
    
    # Load datasets with augmentation
    train_dataset_aug = COCOCLIPDataset(
        config.TRAIN_IMG_DIR,
        os.path.join(config.CACHE_DIR, 'train_text_embeddings.pt'),
        transform=get_augmented_transform()
    )
    
    val_dataset_aug = COCOCLIPDataset(
        config.VAL_IMG_DIR,
        os.path.join(config.CACHE_DIR, 'val_text_embeddings.pt'),
        transform=get_standard_transform()  # No augmentation for validation
    )
    
    train_loader_aug = DataLoader(train_dataset_aug, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    val_loader_aug = DataLoader(val_dataset_aug, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
    
    # Train with augmentation (baseline model)
    model_aug = CLIPModel(embedding_dim=config.EMBEDDING_DIM)
    
    aug_results = train_modification(
        'augmentation',
        model_aug,
        train_loader_aug,
        val_loader_aug,
        config,
        num_epochs=config.NUM_EPOCHS
    )
    
    # ============================================================
    # COMPARISON TABLE
    # ============================================================
    print("\n" + "="*80)
    print("ABLATION STUDY RESULTS COMPARISON")
    print("="*80)
    
    all_results = [baseline_results, reg_results, aug_results]
    
    # Create comparison table
    comparison_data = []
    for result in all_results:
        metrics = result['final_metrics']
        row = {
            'Experiment': result['experiment'],
            'Best Val Loss': f"{result['best_val_loss']:.4f}",
            'I2T R@1': f"{metrics['I2T_R@1']:.4f}",
            'I2T R@5': f"{metrics['I2T_R@5']:.4f}",
            'I2T R@10': f"{metrics['I2T_R@10']:.4f}",
            'T2I R@1': f"{metrics['T2I_R@1']:.4f}",
            'T2I R@5': f"{metrics['T2I_R@5']:.4f}",
            'T2I R@10': f"{metrics['T2I_R@10']:.4f}",
        }
        comparison_data.append(row)
    
    df = pd.DataFrame(comparison_data)
    print("\n" + df.to_string(index=False))
    print("="*80)
    
    # Save comparison
    save_dir = Path(config.CHECKPOINT_DIR) / 'ablation'
    save_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(save_dir / 'ablation_comparison.csv', index=False)
    
    # ============================================================
    # VISUALIZATION
    # ============================================================
    print("\nGenerating comparison visualizations...")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    metrics_to_plot = ['I2T_R@1', 'I2T_R@5', 'T2I_R@1', 'T2I_R@5']
    
    for idx, (ax, metric) in enumerate(zip(axes.flat, metrics_to_plot)):
        experiments = [r['experiment'] for r in all_results]
        values = [r['final_metrics'][metric] for r in all_results]
        
        bars = ax.barh(experiments, values, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
        ax.set_xlabel('Recall', fontsize=12)
        ax.set_title(metric, fontsize=14, fontweight='bold')
        ax.set_xlim([0, 1])
        
        # Add value labels
        for bar, val in zip(bars, values):
            width = bar.get_width()
            ax.text(width + 0.02, bar.get_y() + bar.get_height()/2,
                   f'{val:.4f}', ha='left', va='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(save_dir / 'ablation_comparison.png', dpi=150, bbox_inches='tight')
    print(f"Saved comparison plot to {save_dir / 'ablation_comparison.png'}")
    
    # Plot training curves
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss curves
    for result in [reg_results, aug_results]:
        axes[0].plot(result['train_losses'], label=f"{result['experiment']} (train)")
        axes[0].plot(result['val_losses'], label=f"{result['experiment']} (val)", linestyle='--')
    
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    # I2T Recall@5 over epochs
    for result in [reg_results, aug_results]:
        r5_values = [m['I2T_R@5'] for m in result['metrics_history']]
        axes[1].plot(r5_values, marker='o', label=result['experiment'])
    
    axes[1].axhline(y=baseline_results['final_metrics']['I2T_R@5'], 
                    color='gray', linestyle='--', label='baseline')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('I2T Recall@5')
    axes[1].set_title('I2T Recall@5 Progress')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig(save_dir / 'training_curves.png', dpi=150, bbox_inches='tight')
    print(f"Saved training curves to {save_dir / 'training_curves.png'}")
    
    print("\n" + "="*80)
    print("✓ ABLATION STUDY COMPLETE!")
    print(f"Results saved to: {save_dir}")
    print("="*80)

if __name__ == "__main__":
    main()