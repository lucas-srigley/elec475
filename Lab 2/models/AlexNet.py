import torch.nn as nn
import torchvision.models as models

class AlexNetNose(nn.Module):
    def __init__(self, pretrained=True):
        super(AlexNetNose, self).__init__()
        self.alex = models.alexnet(pretrained=pretrained)
        self.alex.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 1024),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 2)
        )

    def forward(self, x):
        return self.alex(x)
