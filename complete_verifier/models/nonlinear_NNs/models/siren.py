import torch
from torch import nn
import torch.nn.functional as F


class SineLayer(nn.Module):
    # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of omega_0.
    
    # If is_first=True, omega_0 is a frequency factor which simply multiplies the activations before the 
    # nonlinearity. Different signals may require different omega_0 in the first layer - this is a 
    # hyperparameter.
    
    # If is_first=False, then the weights will be divided by omega_0 so as to keep the magnitude of 
    # activations constant, but boost gradients to the weight matrix (see supplement Sec. 1.5)
    
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        
    def forward(self, input):
        return torch.sin(self.linear(input))
    
    
class Siren(nn.Module):
    def __init__(self, in_features=1*28*28, hidden_features=[1024, 100], hidden_layers=1, out_features=10):
        super().__init__()
        
        self.net = []
        self.net.append(SineLayer(in_features, hidden_features[0]))

        for i in range(hidden_layers):
            self.net.append(SineLayer(hidden_features[i], hidden_features[i+1]))

        self.net.append(SineLayer(hidden_features[-1], out_features))
        self.net = nn.Sequential(*self.net)
    
    def forward(self, coords):
        coords = coords.view(coords.size(0), -1)
        output = self.net(coords)
        return output