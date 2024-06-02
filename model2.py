import torch
from torch import nn
import numpy as np

class TetrisModel(nn.Module):
    def __init__(self, freeze=False):
        super().__init__()
        # Conolutional layers
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )
        
        self.field_flatter = nn.Flatten()
        
        # Linear layers
        self.network = nn.Sequential(
            nn.Linear(3599,out_features=512),
            nn.ReLU(),
            nn.Linear(512, 8)
        )

        if freeze:
            self._freeze()
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.to(self.device)

    def forward(self, field, params):
        convs = self.conv_layers(field)
        convs = self.field_flatter(convs)
        x = torch.cat((convs, params),dim=1)
        out = self.network(x)
        return out
    
    def _freeze(self):        
        for p in self.network.parameters():
            p.requires_grad = False