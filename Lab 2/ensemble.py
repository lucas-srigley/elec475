import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from dataset import PetNoseDataset
from SnoutNet import SnoutNet
from AlexNet import AlexNetNose
from VGG16 import VGG16Nose
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transform = transforms.Compose([transforms.Resize((227,227)), transforms.ToTensor()])
test_data = PetNoseDataset("images-original/images", "test_noses.txt", transform=transform)
test_loader = DataLoader(test_data, batch_size=16, shuffle=False)

snout = SnoutNet(); alex = AlexNetNose(); vgg = VGG16Nose()
snout.load_state_dict(torch.load("snoutnet_no_aug.pth", map_location=device))
alex.load_state_dict(torch.load("alexnet_no_aug.pth", map_location=device))
vgg.load_state_dict(torch.load("vgg16_no_aug.pth", map_location=device))
models = [snout.to(device).eval(), alex.to(device).eval(), vgg.to(device).eval()]

all_dists = []
with torch.no_grad():
    for imgs, labels in test_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        preds = sum([m(imgs) for m in models]) / len(models)
        dist = torch.sqrt(torch.sum((preds - labels)**2, dim=1))
        all_dists.extend(dist.cpu().numpy())

all_dists = np.array(all_dists)
print(f"Ensemble Localization Accuracy: Min={all_dists.min():.2f}, Mean={all_dists.mean():.2f}, Max={all_dists.max():.2f}, Std={all_dists.std():.2f}")
