from torch import nn

class LeNet5(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=1,
                      out_channels=6,
                      kernel_size=3,
                      padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,
                         stride=2),
            nn.Conv2d(in_channels=6,
                      out_channels=16,
                      kernel_size=3,
                      padding=0),
            nn.ReLU(),
            nn.Conv2d(in_channels=16,
                      out_channels=16,
                      kernel_size=3,
                      padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,
                         stride=2),
            nn.Flatten(),

            nn.Linear(in_features=16*5*5,
                      out_features=120),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            
            nn.Linear(in_features=120,
                      out_features=84),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            
            nn.Linear(in_features=84,
                      out_features=33)
        )
    def forward(self,x):
        return self.cnn(x)