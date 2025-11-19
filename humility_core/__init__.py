import torch
import torch.nn as nn

class HumilityLayer(nn.Module):
    def __init__(self, input_dim, uncertainty_dim=None):
        super().__init__()
        self.input_dim = input_dim
        self.uncertainty_dim = uncertainty_dim or input_dim // 2
        
        # Core transformation
        self.transform = nn.Linear(input_dim, input_dim)
        
        # Uncertainty quantification
        self.uncertainty_net = nn.Sequential(
            nn.Linear(input_dim, self.uncertainty_dim),
            nn.ReLU(),
            nn.Linear(self.uncertainty_dim, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # Transform input
        out = self.transform(x)
        
        # Compute humility score (uncertainty)
        H = self.uncertainty_net(x)
        
        # Metadata
        meta = {'uncertainty_dim': self.uncertainty_dim}
        
        return out, H, meta

__all__ = ['HumilityLayer']
