import torch, argparse
import torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from dataset import PetNoseDataset
from SnoutNet import SnoutNet
from AlexNet import AlexNetNose
from VGG16 import VGG16Nose

def train_model(model, train_loader, test_loader, device, num_epochs=10, lr=1e-3):
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    train_losses, val_losses = [], []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0
        for images, labels in tqdm(train_loader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
        train_loss = running_loss / len(train_loader.dataset)
        train_losses.append(train_loss)

        model.eval()
        val_loss_total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                val_loss_total += criterion(outputs, labels).item() * images.size(0)
        val_loss = val_loss_total / len(test_loader.dataset)
        val_losses.append(val_loss)

        print(f"Epoch [{epoch+1}/{num_epochs}] Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

    return model, train_losses, val_losses

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, choices=["snoutnet", "alexnet", "vgg16"])
    parser.add_argument("--aug", action="store_true", help="Use data augmentation")
    parser.add_argument("--epochs", type=int, default=10)
    args = parser.parse_args()

    transform_no_aug = transforms.Compose([
        transforms.Resize((227,227)),
        transforms.ToTensor()
    ])

    transform_aug = transforms.Compose([
        transforms.Resize((227,227)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor()
    ])

    train_transform = transform_aug if args.aug else transform_no_aug
    test_transform = transform_no_aug

    train_data = PetNoseDataset("oxford-iiit-pet-noses/images-original/images", "oxford-iiit-pet-noses/train_noses.txt", transform=train_transform)
    test_data = PetNoseDataset("oxford-iiit-pet-noses/images-original/images", "oxford-iiit-pet-noses/test_noses.txt", transform=test_transform)
    train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=16, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.model == "snoutnet": model = SnoutNet()
    elif args.model == "alexnet": model = AlexNetNose()
    else: model = VGG16Nose()

    model = train_model(model, train_loader, test_loader, device, num_epochs=args.epochs)
    torch.save(model.state_dict(), f"{args.model}_{'aug' if args.aug else 'no_aug'}.pth")
