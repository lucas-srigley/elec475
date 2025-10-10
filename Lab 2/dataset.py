from torch.utils.data import Dataset
from PIL import Image
import torch, os

class PetNoseDataset(Dataset):
    def __init__(self, images_dir, annotations_file, transform=None, target_size=(227, 227)):
        self.images_dir = images_dir
        self.transform = transform
        self.target_size = target_size
        self.annotations = []
        with open(annotations_file, 'r') as f:
            for line in f.readlines():
                image_name, coord_str = line.strip().split(',', 1)
                coord = tuple(map(int, coord_str.strip('"()').split(',')))
                self.annotations.append((image_name, coord))

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        image_name, coord = self.annotations[idx]
        image = Image.open(os.path.join(self.images_dir, image_name)).convert('RGB')
        w, h = image.size
        if self.transform: image = self.transform(image)
        x, y = coord
        label = torch.tensor([x * (self.target_size[0] / w), y * (self.target_size[1] / h)], dtype=torch.float32)
        return image, label
