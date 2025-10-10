import torch.nn as nn
import torchvision.models as models

class VGG16Nose(nn.Module):
    def __init__(self, pretrained=True):
        super(VGG16Nose, self).__init__()
        self.vgg = models.vgg16(pretrained=pretrained)
        self.vgg.classifier = nn.Sequential(
            nn.Linear(512*7*7, 4096),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(4096, 1024),
            nn.ReLU(),
            nn.Linear(1024, 2)
        )

    def forward(self, x):
        return self.vgg(x)
