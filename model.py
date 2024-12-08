
import torch
import torch.nn as nn
from torchaudio.transforms import MelSpectrogram
class resnet_tfatt(nn.Module):
    def __init__(self, use_tfatt = True) -> None:
        super().__init__()
        self.logmel = MelSpectrogram(sample_rate=16000, n_fft=640, n_mels=80)
        from att_resnet import resnet12
        self.resnet = resnet12(use_tfatt=use_tfatt, avg_pool=True)

    def forward(self, x):
        x = self.logmel(x)
        x = x.unsqueeze(1)
        x = self.resnet(x)

        return x

class cls(nn.Module):
    def __init__(self,classes_num=5) -> None:
        super().__init__()
        self.linear = nn.Linear(640,classes_num)
        self.soft = nn.Softmax(dim=1)
    def forward(self,x):
        x = self.linear(x)
        return self.soft(x)
        
