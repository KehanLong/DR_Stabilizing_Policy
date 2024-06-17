import torch
import torch.nn as nn

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1, 6)
        self.fc2 = nn.Linear(6, 6)
        self.fc3 = nn.Linear(6, 1)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()  
        self.relu3 = nn.ReLU() # Added ReLU activation for the output

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)  
        x = self.fc3(x)
        x = torch.norm(x, dim=1)
        return x

def create_simple_model():
    return SimpleModel()

def box_data(lower_limit, upper_limit, scale):
    data_min = scale * torch.tensor(lower_limit, dtype=torch.get_default_dtype()).unsqueeze(0)
    data_max = scale * torch.tensor(upper_limit, dtype=torch.get_default_dtype()).unsqueeze(0)
    return {
        'data_min': data_min,
        'data_max': data_max,
    }


