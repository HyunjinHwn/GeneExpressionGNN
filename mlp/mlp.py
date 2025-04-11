import torch
import torch.nn as nn

class LinearRegression(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.W1 = nn.Linear(dim_in, dim_out)

    def forward(self, X):
        return self.W1(X)

class MLPRegression(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.W1 = nn.Linear(dim_in, dim_out)
        self.W2 = nn.Linear(dim_out, dim_out)
        self.apply(self.initialize_weights)

    def forward(self, X):
        return self.W2(torch.relu(self.W1(X)))
    
    def initialize_weights(self, m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)