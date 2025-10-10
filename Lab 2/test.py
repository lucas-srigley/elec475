import torch, argparse
from torchvision import transforms
from torch.utils.data import DataLoader
from dataset import PetNoseDataset
from SnoutNet import SnoutNet
from AlexNet import AlexNet
from VGG16 import VGG16

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, required=True)
parser.add_argument("--weights", type=str, required=True)
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([transforms.Resize((227,227)), transforms.ToTensor()])
test_data = PetNoseDataset("oxford-iiit-pet-noses/images-original/images", "oxford-iiit-pet-noses/test_noses.txt", transform=transform)
test_loader = DataLoader(test_data, batch_size=16, shuffle=False)

if args.model == "snoutnet": model = SnoutNet()
elif args.model == "alexnet": model = AlexNet()
else: model = VGG16()

model.load_state_dict(torch.load(args.weights, map_location=device))
model.to(device); model.eval()

distances = []
with torch.no_grad():
    for imgs, labels in test_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        preds = model(imgs)
        dist = torch.sqrt(torch.sum((preds - labels)**2, dim=1))
        distances.extend(dist.cpu().numpy())

import numpy as np
distances = np.array(distances)
print(f"Localization Accuracy (pixels): Min={distances.min():.2f}, Mean={distances.mean():.2f}, Max={distances.max():.2f}, Std={distances.std():.2f}")
