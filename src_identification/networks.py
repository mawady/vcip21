import torch
import torch.nn.functional as F
from torch import nn


def initialize_weights(*models):
    for model in models:
        for module in model.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.ConvTranspose2d) or isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()


class SCI(nn.Module):
    def __init__(self, input_channels, num_classes):  # 32x32
        super(SCI, self).__init__()

        self.convs = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=3, stride=2, padding=0),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True),

            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=0),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(inplace=True),

            nn.MaxPool2d(3, stride=2)
        )

        self.features = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(3*3*128, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(inplace=True),

            nn.Dropout(p=0.5),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(inplace=True),

            nn.Linear(512, num_classes)
        )

        initialize_weights(self)

    def forward(self, x):
        x = self.convs(x)
        x = x.view(x.size(0), -1)
        return self.features(x)
