import torch, argparse
import matplotlib.pyplot as plt
from torchvision import transforms
from dataset import PetNoseDataset
from models.SnoutNet import SnoutNet
from models.AlexNet import AlexNetNose
from models.VGG16 import VGG16Nose

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, required=True)
parser.add_argument("--weights", type=str, required=True)
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([transforms.Resize((227,227)), transforms.ToTensor()])
dataset = PetNoseDataset("images-original/images", "test_noses.txt", transform=transform)

if args.model == "snoutnet": model = SnoutNet()
elif args.model == "alexnet": model = AlexNetNose()
else: model = VGG16Nose()

model.load_state_dict(torch.load(args.weights, map_location=device))
model.to(device); model.eval()

img, label = dataset[0]
plt.imshow(img.permute(1,2,0))
with torch.no_grad():
    pred = model(img.unsqueeze(0).to(device)).cpu().squeeze()
plt.scatter(pred[0], pred[1], c='r', label='Predicted Nose')
plt.scatter(label[0], label[1], c='g', label='Actual Nose')
plt.legend(); plt.title(f"{args.model} Visualization"); plt.show()
