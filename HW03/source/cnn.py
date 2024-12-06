import torch.nn as nn

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1), nn.ReLU(), # 256x256x32
            nn.MaxPool2d(kernel_size=2, stride=2),                 # 128x128x32
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(), # 128x128x64
            nn.MaxPool2d(kernel_size=2, stride=2),                  # 64x64x64
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(), # 64x64x128
            nn.MaxPool2d(kernel_size=2, stride=2),                   # 32x32x128
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1), nn.ReLU(), # 32x32x256
            nn.MaxPool2d(kernel_size=2, stride=2),                    # 16x16x256
            
            nn.Flatten(),
            nn.Linear(256 * 16 * 16, 512), nn.ReLU(),
            nn.Linear(512, 1), nn.Sigmoid()
        )

    def forward(self, x):
        return self.layers(x).view(-1)
