import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


class TetrisModel(nn.Module):
    def __init__(self):
        super(TetrisModel, self).__init__()
        # Convolutional layers for the field
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1,)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        
        # Fully connected layers for the combined features
        self.fc1 = nn.Linear(64 * 20 * 10 + 15, 128)  # Adjust input size accordingly
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 8)
        
    def forward(self, field, other_state):
        x = torch.relu(self.conv1(field))
        x = torch.relu(self.conv2(x))
        x = x.view(x.size(0), -1)  # Flatten the output of conv2
        
        combined = torch.cat((x, other_state), dim=1)
        x = torch.relu(self.fc1(combined))
        x = torch.relu(self.fc2(x))
        action_values = self.fc3(x)
        return action_values
    
    