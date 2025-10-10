import torch, argparse
import torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from dataset import PetNoseDataset
from models.SnoutNet import SnoutNet
from models.AlexNet import AlexNetNose
from models.VGG16 import VGG16Nose

def train_model(model, train_loader, test_loader, device, num_epochs=10, lr=1e-3):
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for imgs, labels in tqdm(train_loader):
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(imgs), labels)
            loss.backward(); optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {total_loss/len(train_loader):.4f}")
    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, choices=["snoutnet", "alexnet", "vgg16"])
    parser.add_argument("--aug", action="store_true", help="Use data augmentation")
    parser.add_argument("--epochs", type=int, default=10)
    args = parser.parse_args()

    transform_no_aug = transforms.Compose([transforms.Resize((227,227)), transforms.ToTensor()])
    transform_aug = transforms.Compose([
        transforms.Resize((227,227)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.2,0.2),
        transforms.RandomRotation(15),
        transforms.ToTensor()
    ])

    train_transform = transform_aug if args.aug else transform_no_aug
    test_transform = transform_no_aug

    train_data = PetNoseDataset("images-original/images", "train_noses.txt", transform=train_transform)
    test_data = PetNoseDataset("images-original/images", "test_noses.txt", transform=test_transform)
    train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=16, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.model == "snoutnet": model = SnoutNet()
    elif args.model == "alexnet": model = AlexNetNose()
    else: model = VGG16Nose()

    model = train_model(model, train_loader, test_loader, device, num_epochs=args.epochs)
    torch.save(model.state_dict(), f"{args.model}_{'aug' if args.aug else 'no_aug'}.pth")
