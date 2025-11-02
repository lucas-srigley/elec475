import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import VOCSegmentation
from torchvision.models.segmentation import fcn_resnet50
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
from model import CompactSegmentationModel

# Hyperparameters
NUM_CLASSES = 21
BATCH_SIZE = 8
NUM_EPOCHS = 10
LEARNING_RATE = 1e-3
INPUT_SIZE = (256, 256)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Knowledge distillation hyperparameters
ALPHA = 0.5  # Weight for ground truth loss
BETA = 0.5   # Weight for distillation loss
TEMPERATURE = 3.0

# Feature-based distillation weights (weighted combination)
FEATURE_WEIGHTS = {'low': 0.2, 'mid': 0.3, 'high': 0.5}
FEATURE_LOSS_WEIGHT = 0.1  # Weight for feature distillation loss


class VOCSegmentationDataset(torch.utils.data.Dataset):
    """Wrapper for VOC Segmentation dataset with proper preprocessing"""
    def __init__(self, root, year='2012', image_set='train', transform=None, target_transform=None):
        self.dataset = VOCSegmentation(root=root, year=year, image_set=image_set, 
                                       download=True, transform=None, target_transform=None)
        self.transform = transform
        self.target_transform = target_transform
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        img, target = self.dataset[idx]
        
        if self.transform:
            img = self.transform(img)
        if self.target_transform:
            target = self.target_transform(target)
            
        return img, target


def get_transforms():
    """Get image and target transforms"""
    img_transform = transforms.Compose([
        transforms.Resize(INPUT_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    target_transform = transforms.Compose([
        transforms.Resize(INPUT_SIZE, interpolation=transforms.InterpolationMode.NEAREST),
        transforms.PILToTensor(),
        transforms.Lambda(lambda x: x.squeeze(0).long())
    ])
    
    return img_transform, target_transform


def calculate_miou(pred, target, num_classes=21, ignore_index=255):
    """Calculate mean Intersection over Union"""
    pred = pred.cpu().numpy()
    target = target.cpu().numpy()
    
    iou_list = []
    for cls in range(num_classes):
        pred_mask = (pred == cls)
        target_mask = (target == cls)
        
        intersection = np.logical_and(pred_mask, target_mask).sum()
        union = np.logical_or(pred_mask, target_mask).sum()
        
        if union == 0:
            continue
        
        iou = intersection / union
        iou_list.append(iou)
    
    return np.mean(iou_list) if len(iou_list) > 0 else 0.0


def modified_softmax(logits, temperature=1.0):
    """Modified softmax with temperature (Hinton-style)"""
    return torch.softmax(logits / temperature, dim=1)


def response_based_distillation_loss(student_logits, teacher_logits, targets, alpha=0.5, beta=0.5, temperature=3.0):
    """
    Response-based knowledge distillation loss (from lecture slide 7)
    Loss = α · H(σ(z_s; 1), y) + β · H(σ(z_s; τ), σ(z_t; τ))
    
    Args:
        student_logits: Student model output logits
        teacher_logits: Teacher model output logits  
        targets: Ground truth labels
        alpha: Weight for ground truth loss
        beta: Weight for distillation loss
        temperature: Temperature for softening distributions
    """
    # Ground truth loss (student with temperature 1)
    ce_loss = nn.CrossEntropyLoss(ignore_index=255)(student_logits, targets)
    
    # Distillation loss (both student and teacher with temperature τ)
    # Using KL divergence between soft predictions
    distill_loss = nn.KLDivLoss(reduction='batchmean')(
        torch.log_softmax(student_logits / temperature, dim=1),
        torch.softmax(teacher_logits / temperature, dim=1)
    ) * (temperature ** 2)  # Scale by T^2 as per Hinton
    
    # Combined loss
    total_loss = alpha * ce_loss + beta * distill_loss
    
    return total_loss, ce_loss.item(), distill_loss.item()


def feature_based_distillation_loss(student_features, teacher_features, weights):
    """
    Feature-based knowledge distillation using cosine similarity loss (from lecture slides 8, 10)
    L_feature = Σ w_i · (1 - cosine_similarity(f_s^i, f_t^i))
    
    Uses weighted combination across multiple feature levels.
    
    Args:
        student_features: Dictionary of student feature maps {'low', 'mid', 'high'}
        teacher_features: Dictionary of teacher feature maps {'low', 'mid', 'high'}
        weights: Dictionary of weights for each level
    """
    total_loss = 0.0
    losses = {}
    
    for level in ['low', 'mid', 'high']:
        s_feat = student_features[level]
        t_feat = teacher_features[level]
        
        # Ensure features have same spatial dimensions
        if s_feat.shape[2:] != t_feat.shape[2:]:
            t_feat = nn.functional.interpolate(t_feat, size=s_feat.shape[2:], 
                                               mode='bilinear', align_corners=False)
        
        # If different number of channels, project student to match teacher
        if s_feat.shape[1] != t_feat.shape[1]:
            # Use adaptive pooling or projection
            if not hasattr(feature_based_distillation_loss, f'proj_{level}'):
                proj = nn.Conv2d(s_feat.shape[1], t_feat.shape[1], 1, bias=False).to(s_feat.device)
                setattr(feature_based_distillation_loss, f'proj_{level}', proj)
            proj = getattr(feature_based_distillation_loss, f'proj_{level}')
            s_feat = proj(s_feat)
        
        # Cosine similarity loss (1 - cosine_similarity) from lecture
        # Reshape features: (B, C, H, W) -> (B*H*W, C)
        b, c, h, w = s_feat.shape
        s_flat = s_feat.permute(0, 2, 3, 1).reshape(-1, c)
        t_flat = t_feat.permute(0, 2, 3, 1).reshape(-1, c)
        
        # Normalize features for cosine similarity
        s_norm = nn.functional.normalize(s_flat, p=2, dim=1)
        t_norm = nn.functional.normalize(t_flat, p=2, dim=1)
        
        # Cosine similarity loss: 1 - cosine_similarity
        cos_sim = (s_norm * t_norm).sum(dim=1).mean()
        cos_loss = 1.0 - cos_sim
        
        losses[level] = cos_loss.item()
        total_loss += weights[level] * cos_loss
    
    return total_loss, losses


class TeacherFeatureExtractor(nn.Module):
    """Wrapper to extract intermediate features from FCN-ResNet50 teacher"""
    def __init__(self, teacher_model):
        super().__init__()
        self.teacher = teacher_model
        self.features = {}
        
        # Register hooks to capture intermediate features
        # FCN-ResNet50 uses ResNet50 backbone with specific layers
        # We'll extract features at similar strides as student (stride 4, 8, 16)
        self.teacher.backbone.layer1.register_forward_hook(self._get_features('low'))  # stride 4
        self.teacher.backbone.layer2.register_forward_hook(self._get_features('mid'))  # stride 8
        self.teacher.backbone.layer3.register_forward_hook(self._get_features('high'))  # stride 16
        
    def _get_features(self, name):
        def hook(module, input, output):
            self.features[name] = output
        return hook
    
    def forward(self, x):
        self.features = {}
        output = self.teacher(x)
        return output, self.features


def train_epoch(student_model, teacher_extractor, dataloader, optimizer, epoch, 
                use_distillation='none'):
    """Train for one epoch"""
    student_model.train()
    teacher_extractor.eval()
    
    running_loss = 0.0
    running_ce_loss = 0.0
    running_distill_loss = 0.0
    running_feature_loss = 0.0
    running_miou = 0.0
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{NUM_EPOCHS}')
    
    for images, targets in pbar:
        images = images.to(DEVICE)
        targets = targets.to(DEVICE)
        
        optimizer.zero_grad()
        
        # Forward pass through student (with or without features)
        if use_distillation == 'feature':
            student_logits, student_features = student_model(images, return_features=True)
        else:
            student_logits = student_model(images)
        
        # Calculate loss based on distillation method
        if use_distillation == 'response':
            # Response-based: Use final output logits only
            with torch.no_grad():
                teacher_output = teacher_extractor.teacher(images)
                teacher_logits = teacher_output['out']
            
            loss, ce_loss, dist_loss = response_based_distillation_loss(
                student_logits, teacher_logits, targets, ALPHA, BETA, TEMPERATURE
            )
            running_ce_loss += ce_loss
            running_distill_loss += dist_loss
            
        elif use_distillation == 'feature':
            # Feature-based: Use intermediate feature maps
            with torch.no_grad():
                teacher_output, teacher_features = teacher_extractor(images)
                teacher_logits = teacher_output['out']
            
            # Ground truth loss
            ce_loss = nn.CrossEntropyLoss(ignore_index=255)(student_logits, targets)
            
            # Feature distillation loss with weighted combination
            feat_loss, feat_losses = feature_based_distillation_loss(
                student_features, teacher_features, FEATURE_WEIGHTS
            )
            
            # Combined loss: ground truth + feature distillation
            loss = ce_loss + FEATURE_LOSS_WEIGHT * feat_loss
            
            running_ce_loss += ce_loss.item()
            running_feature_loss += feat_loss.item()
            
        else:
            # No distillation - standard training
            loss = nn.CrossEntropyLoss(ignore_index=255)(student_logits, targets)
            running_ce_loss += loss.item()
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        # Calculate mIoU
        with torch.no_grad():
            pred = torch.argmax(student_logits, dim=1)
            miou = calculate_miou(pred, targets, NUM_CLASSES)
            running_miou += miou
        
        # Update progress bar
        postfix = {'loss': running_loss / (pbar.n + 1), 'mIoU': running_miou / (pbar.n + 1)}
        if use_distillation == 'response':
            postfix['dist_loss'] = running_distill_loss / (pbar.n + 1)
        elif use_distillation == 'feature':
            postfix['feat_loss'] = running_feature_loss / (pbar.n + 1)
        pbar.set_postfix(postfix)
    
    avg_loss = running_loss / len(dataloader)
    avg_miou = running_miou / len(dataloader)
    
    return avg_loss, avg_miou


def validate(model, dataloader):
    """Validate the model"""
    model.eval()
    running_loss = 0.0
    running_miou = 0.0
    
    with torch.no_grad():
        for images, targets in tqdm(dataloader, desc='Validation'):
            images = images.to(DEVICE)
            targets = targets.to(DEVICE)
            
            outputs = model(images)
            loss = nn.CrossEntropyLoss(ignore_index=255)(outputs, targets)
            
            running_loss += loss.item()
            
            pred = torch.argmax(outputs, dim=1)
            miou = calculate_miou(pred, targets, NUM_CLASSES)
            running_miou += miou
    
    avg_loss = running_loss / len(dataloader)
    avg_miou = running_miou / len(dataloader)
    
    return avg_loss, avg_miou


def plot_losses(train_losses, val_losses, train_mious, val_mious, save_path='training_plots.png'):
    """Plot training curves"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    ax1.plot(train_losses, label='Train Loss')
    ax1.plot(val_losses, label='Val Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True)
    
    ax2.plot(train_mious, label='Train mIoU')
    ax2.plot(val_mious, label='Val mIoU')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('mIoU')
    ax2.set_title('Training and Validation mIoU')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Training plots saved to {save_path}")


def main():
    print(f"Using device: {DEVICE}")
    
    # Create datasets
    img_transform, target_transform = get_transforms()
    
    train_dataset = VOCSegmentationDataset(
        root='./data', year='2012', image_set='train',
        transform=img_transform, target_transform=target_transform
    )
    
    val_dataset = VOCSegmentationDataset(
        root='./data', year='2012', image_set='val',
        transform=img_transform, target_transform=target_transform
    )
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, 
                             shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, 
                           shuffle=False, num_workers=2)
    
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Val dataset size: {len(val_dataset)}")
    
    # Load teacher model (pretrained FCN-ResNet50)
    print("\nLoading teacher model (FCN-ResNet50)...")
    teacher_model = fcn_resnet50(pretrained=True).to(DEVICE)
    teacher_model.eval()
    for param in teacher_model.parameters():
        param.requires_grad = False
    
    # Wrap teacher for feature extraction
    teacher_extractor = TeacherFeatureExtractor(teacher_model)
    
    # Create student model
    print("Creating student model...")
    student_model = CompactSegmentationModel(num_classes=NUM_CLASSES, pretrained=True).to(DEVICE)
    print(f"Student model parameters: {student_model.get_num_parameters():,}")
    
    # Optimizer
    optimizer = optim.Adam(student_model.parameters(), lr=LEARNING_RATE)
    
    # Training
    train_losses = []
    val_losses = []
    train_mious = []
    val_mious = []
    
    best_val_miou = 0.0
    
    # Choose distillation method: 'none', 'response', 'feature'
    # IMPORTANT: Train 3 separate times with each method:
    # 1. distillation_method = 'none'      -> generates best_model_none.pth
    # 2. distillation_method = 'response'  -> generates best_model_response.pth  
    # 3. distillation_method = 'feature'   -> generates best_model_feature.pth
    distillation_method = 'response'  # Change this to test different methods
    
    print(f"\nTraining with distillation method: {distillation_method}")
    print("=" * 50)
    print("NOTE: You need to train 3 times total with different methods:")
    print("  1. 'none' - Without distillation")
    print("  2. 'response' - Response-based KD") 
    print("  3. 'feature' - Feature-based KD")
    print("=" * 50)
    
    for epoch in range(NUM_EPOCHS):
        train_loss, train_miou = train_epoch(
            student_model, teacher_extractor, train_loader, optimizer, epoch,
            use_distillation=distillation_method
        )
        
        val_loss, val_miou = validate(student_model, val_loader)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_mious.append(train_miou)
        val_mious.append(val_miou)
        
        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}:")
        print(f"  Train Loss: {train_loss:.4f}, Train mIoU: {train_miou:.4f}")
        print(f"  Val Loss: {val_loss:.4f}, Val mIoU: {val_miou:.4f}")
        
        # Save best model
        if val_miou > best_val_miou:
            best_val_miou = val_miou
            torch.save(student_model.state_dict(), f'best_model_{distillation_method}.pth')
            print(f"  Saved best model with mIoU: {best_val_miou:.4f}")
    
    # Save final model
    torch.save(student_model.state_dict(), f'final_model_{distillation_method}.pth')
    
    # Plot training curves
    plot_losses(train_losses, val_losses, train_mious, val_mious, 
                save_path=f'training_plots_{distillation_method}.png')
    
    print("\nTraining completed!")
    print(f"Best validation mIoU: {best_val_miou:.4f}")


if __name__ == '__main__':
    main()