import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import VOCSegmentation
from torchvision.models.segmentation import fcn_resnet50
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import os
from model import CompactSegmentationModel

# Hyperparameters
NUM_CLASSES = 21
BATCH_SIZE = 8
INPUT_SIZE = (256, 256)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class VOCSegmentationDataset(torch.utils.data.Dataset):
    """Wrapper for VOC Segmentation dataset with proper preprocessing"""
    def __init__(self, root, year='2012', image_set='val', transform=None, target_transform=None):
        self.dataset = VOCSegmentation(root=root, year=year, image_set=image_set, 
                                       download=False, transform=None, target_transform=None)
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
    
    iou_per_class = []
    for cls in range(num_classes):
        pred_mask = (pred == cls)
        target_mask = (target == cls)
        
        intersection = np.logical_and(pred_mask, target_mask).sum()
        union = np.logical_or(pred_mask, target_mask).sum()
        
        if union == 0:
            continue
        
        iou = intersection / union
        iou_per_class.append(iou)
    
    return np.mean(iou_per_class) if len(iou_per_class) > 0 else 0.0, iou_per_class


def test_pretrained_fcn():
    """Step 2.1: Test pretrained FCN-ResNet50"""
    print("=" * 70)
    print("Step 2.1: Testing Pretrained FCN-ResNet50")
    print("=" * 70)
    
    # Load model
    model = fcn_resnet50(pretrained=True).to(DEVICE)
    model.eval()
    
    # Load dataset
    img_transform, target_transform = get_transforms()
    val_dataset = VOCSegmentationDataset(
        root='.', year='2012', image_set='val',
        transform=img_transform, target_transform=target_transform
    )
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, 
                           shuffle=False, num_workers=2)
    
    print(f"Val dataset size: {len(val_dataset)}")
    
    # Test
    all_mious = []
    total_time = 0
    num_images = 0
    
    with torch.no_grad():
        for images, targets in tqdm(val_loader, desc='Testing FCN-ResNet50'):
            images = images.to(DEVICE)
            targets = targets.to(DEVICE)
            
            # Measure inference time
            start_time = time.time()
            outputs = model(images)['out']
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            end_time = time.time()
            
            total_time += (end_time - start_time)
            num_images += images.size(0)
            
            pred = torch.argmax(outputs, dim=1)
            
            # Calculate mIoU for each image in batch
            for i in range(pred.size(0)):
                miou, _ = calculate_miou(pred[i:i+1], targets[i:i+1], NUM_CLASSES)
                all_mious.append(miou)
    
    avg_miou = np.mean(all_mious)
    avg_time_per_image = (total_time / num_images) * 1000  # Convert to ms
    num_params = sum(p.numel() for p in model.parameters())
    
    print(f"\nFCN-ResNet50 Results:")
    print(f"  Mean IoU: {avg_miou:.4f}")
    print(f"  Inference time: {avg_time_per_image:.2f} ms/image")
    print(f"  Total parameters: {num_params:,}")
    
    return avg_miou, avg_time_per_image, num_params


def test_student_model(model_path, model_name="Student"):
    """Test student model"""
    print("\n" + "=" * 70)
    print(f"Testing {model_name} Model: {model_path}")
    print("=" * 70)
    
    # Load model
    model = CompactSegmentationModel(num_classes=NUM_CLASSES, pretrained=False).to(DEVICE)
    
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        print(f"Loaded model from {model_path}")
    else:
        print(f"Warning: Model file {model_path} not found. Using untrained model.")
    
    model.eval()
    
    # Load dataset
    img_transform, target_transform = get_transforms()
    val_dataset = VOCSegmentationDataset(
        root='.', year='2012', image_set='val',
        transform=img_transform, target_transform=target_transform
    )
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, 
                           shuffle=False, num_workers=2)
    
    # Test
    all_mious = []
    total_time = 0
    num_images = 0
    
    with torch.no_grad():
        for images, targets in tqdm(val_loader, desc=f'Testing {model_name}'):
            images = images.to(DEVICE)
            targets = targets.to(DEVICE)
            
            # Measure inference time
            start_time = time.time()
            outputs = model(images)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            end_time = time.time()
            
            total_time += (end_time - start_time)
            num_images += images.size(0)
            
            pred = torch.argmax(outputs, dim=1)
            
            # Calculate mIoU for each image in batch
            for i in range(pred.size(0)):
                miou, _ = calculate_miou(pred[i:i+1], targets[i:i+1], NUM_CLASSES)
                all_mious.append(miou)
    
    avg_miou = np.mean(all_mious)
    avg_time_per_image = (total_time / num_images) * 1000  # Convert to ms
    num_params = model.get_num_parameters()
    
    print(f"\n{model_name} Results:")
    print(f"  Mean IoU: {avg_miou:.4f}")
    print(f"  Inference time: {avg_time_per_image:.2f} ms/image")
    print(f"  Total parameters: {num_params:,}")
    
    return avg_miou, avg_time_per_image, num_params


def visualize_predictions(model, dataset, num_samples=5, save_path='predictions.png'):
    """Visualize some predictions"""
    model.eval()
    
    # VOC color map
    def get_color_map():
        color_map = np.zeros((256, 3), dtype=np.uint8)
        color_map[0] = [0, 0, 0]  # background
        color_map[1] = [128, 0, 0]  # aeroplane
        color_map[2] = [0, 128, 0]  # bicycle
        color_map[3] = [128, 128, 0]  # bird
        color_map[4] = [0, 0, 128]  # boat
        color_map[5] = [128, 0, 128]  # bottle
        color_map[6] = [0, 128, 128]  # bus
        color_map[7] = [128, 128, 128]  # car
        color_map[8] = [64, 0, 0]  # cat
        color_map[9] = [192, 0, 0]  # chair
        color_map[10] = [64, 128, 0]  # cow
        color_map[11] = [192, 128, 0]  # table
        color_map[12] = [64, 0, 128]  # dog
        color_map[13] = [192, 0, 128]  # horse
        color_map[14] = [64, 128, 128]  # motorbike
        color_map[15] = [192, 128, 128]  # person
        color_map[16] = [0, 64, 0]  # plant
        color_map[17] = [128, 64, 0]  # sheep
        color_map[18] = [0, 192, 0]  # sofa
        color_map[19] = [128, 192, 0]  # train
        color_map[20] = [0, 64, 128]  # tv
        return color_map
    
    color_map = get_color_map()
    
    fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4*num_samples))
    
    indices = np.random.choice(len(dataset), num_samples, replace=False)
    
    with torch.no_grad():
        for idx, data_idx in enumerate(indices):
            img, target = dataset[data_idx]
            img_input = img.unsqueeze(0).to(DEVICE)
            
            output = model(img_input)
            pred = torch.argmax(output, dim=1).squeeze().cpu().numpy()
            
            # Denormalize image
            img_np = img.cpu().numpy().transpose(1, 2, 0)
            img_np = img_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
            img_np = np.clip(img_np, 0, 1)
            
            # Apply color map
            target_colored = color_map[target.cpu().numpy()]
            pred_colored = color_map[pred]
            
            # Plot
            axes[idx, 0].imshow(img_np)
            axes[idx, 0].set_title('Input Image')
            axes[idx, 0].axis('off')
            
            axes[idx, 1].imshow(target_colored)
            axes[idx, 1].set_title('Ground Truth')
            axes[idx, 1].axis('off')
            
            axes[idx, 2].imshow(pred_colored)
            axes[idx, 2].set_title('Prediction')
            axes[idx, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"\nPrediction visualizations saved to {save_path}")


def main():
    print(f"Using device: {DEVICE}\n")
    
    # Step 2.1: Test pretrained FCN-ResNet50
    fcn_miou, fcn_time, fcn_params = test_pretrained_fcn()
    
    # Test different student models
    results = {}
    
    # Without distillation
    if os.path.exists('best_model_none.pth'):
        miou, time_ms, params = test_student_model('best_model_none.pth', "Without Distillation")
        results['Without'] = {'mIoU': miou, 'time': time_ms, 'params': params}
    
    # Response-based distillation
    if os.path.exists('best_model_response.pth'):
        miou, time_ms, params = test_student_model('best_model_response.pth', "Response-based KD")
        results['Response-based'] = {'mIoU': miou, 'time': time_ms, 'params': params}
    
    # Feature-based distillation
    if os.path.exists('best_model_feature.pth'):
        miou, time_ms, params = test_student_model('best_model_feature.pth', "Feature-based KD")
        results['Feature-based'] = {'mIoU': miou, 'time': time_ms, 'params': params}
    
    # Print comparison table
    print("\n" + "=" * 70)
    print("COMPARISON TABLE")
    print("=" * 70)
    print(f"{'Knowledge Distillation':<25} {'mIoU':<12} {'# Parameters':<15} {'Inference Speed (ms)':<20}")
    print("-" * 70)
    print(f"{'FCN-ResNet50 (Teacher)':<25} {fcn_miou:<12.4f} {fcn_params:<15,} {fcn_time:<20.2f}")
    
    for method, data in results.items():
        method_label = method if method == 'Without' else method
        print(f"{method_label:<25} {data['mIoU']:<12.4f} {data['params']:<15,} {data['time']:<20.2f}")
    
    # Visualize predictions for best model
    # if results:
    # Map best_method to actual filenames
    model_file_map = {
        'Without': 'best_model_none.pth',
        'Response-based': 'best_model_response.pth',
        'Feature-based': 'best_model_feature.pth'
    }

    if results:
        best_method = max(results, key=lambda k: results[k]['mIoU'])
        model_path = model_file_map[best_method]

        if os.path.exists(model_path):
            print(f"\nVisualizing predictions from best model: {best_method}")
            
            # First create the model
            model = CompactSegmentationModel(num_classes=NUM_CLASSES).to(DEVICE)
            
            # Then load the weights
            model.load_state_dict(torch.load(model_path, map_location=DEVICE))
            
            img_transform, target_transform = get_transforms()
            val_dataset = VOCSegmentationDataset(
                root='.', year='2012', image_set='val',
                transform=img_transform, target_transform=target_transform
            )
            
            visualize_predictions(model, val_dataset, num_samples=5, 
                                save_path=f'predictions_{best_method.lower().replace("-", "")}.png')
        else:
            print(f"Error: {model_path} does not exist")

    
    print("\n" + "=" * 70)
    print("Testing completed!")
    print("=" * 70)


if __name__ == '__main__':
    main()