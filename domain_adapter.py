import torch
import torch.nn as nn
import torch.nn.functional as F

class AutoEncoder(nn.Module):
    def __init__(self, input_dim=512):
        super(AutoEncoder, self).__init__()

        self.dim = [512, 512]
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, self.dim[0]),
            nn.ReLU(),
            nn.Linear(self.dim[0], self.dim[1]),
            nn.ReLU(),

        )
        self.decoder = nn.Sequential(

            nn.Linear(self.dim[1], self.dim[0]),
            nn.ReLU(),
            nn.Linear(self.dim[0], input_dim),
        )

    def forward(self, x):

        x = self.encoder(x)
  
        x = self.decoder(x)
   
        return x

class Attention(nn.Module):
    def __init__(self, input_dim=512):
        super(Attention, self).__init__()

        self.att = nn.Linear(input_dim, input_dim)

    def forward(self, x):
        return self.att(x)
    
class DomainAdapter(nn.Module):
    def __init__(self, input_dim=512):
        super(DomainAdapter, self).__init__()

        self.att = Attention(input_dim)
        self.autoenc = AutoEncoder(input_dim)

    def forward(self, x):
        y = self.att(x)
        z = self.autoenc(x)
        return x+y+z
    


class DistillKL(nn.Module):
    """KL divergence"""
    def __init__(self, T):
        super(DistillKL, self).__init__()
        self.T = T

    def forward(self, y_s, y_t):
        p_s = F.log_softmax(y_s/self.T, dim=1)
        p_t = F.softmax(y_t/self.T, dim=1)
        loss = F.kl_div(p_s, p_t, reduction='sum') * (self.T**2) / y_s.shape[0]
        return loss