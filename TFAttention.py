from turtle import forward
import torch
import torch.nn as nn

class TimeAttentionModule(nn.Module):
    
    def __init__(self, in_channels):
        
        super(TimeAttentionModule, self).__init__()
        
        self.query_conv = nn.Conv2d(in_channels, in_channels//8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels, in_channels//8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        
        self.gamma = nn.Parameter(torch.zeros(1))
        
        self.softmax = nn.Softmax(dim=-1)
                
    
    def forward(self, x):
        
        N, C, H, W = x.shape
        
      
        query = self.query_conv(x).permute(0, 2, 1, 3).contiguous().view(N, H, -1)
  
        key = self.key_conv(x).permute(0, 1, 3, 2).contiguous().view(N, -1, H)
     
        energy = torch.bmm(query, key)
        
    
        attention = self.softmax(energy)
      
        value = self.value_conv(x).permute(0, 1, 3, 2).contiguous().view(N, -1, H)
       
        out = torch.bmm(value, attention.permute(0, 2, 1))  

        out = out.view(N, C, W, H).permute(0, 1, 3, 2).contiguous()

       
        out = self.gamma*out + x
        
        return out

class FreqAttentionModule(nn.Module):

    
    def __init__(self, in_channels):
        
        super(FreqAttentionModule, self).__init__()
        
        self.query_conv = nn.Conv2d(in_channels, in_channels//8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels, in_channels//8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        
     
        self.gamma = nn.Parameter(torch.zeros(1))
        
        self.softmax = nn.Softmax(dim=-1)
                
    
    def forward(self, x):
        
        N, C, H, W = x.shape
        
     
        query = self.query_conv(x).permute(0, 3, 1, 2).contiguous().view(N, W, -1)
 
        key = self.key_conv(x).view(N, -1, W)

        energy = torch.bmm(query, key)
        

        attention = self.softmax(energy)

        value = self.value_conv(x).view(N, -1, W)
   
        out = torch.bmm(value, attention.permute(0, 2, 1)) 

        out = out.view(N, C, H, W)

        out = self.gamma*out + x
        
        return out

class TFAttentionModule(nn.Module):

    
    def __init__(self, in_channels):
        
        super(TFAttentionModule, self).__init__()

        self.time = TimeAttentionModule(in_channels)
        self.freq = FreqAttentionModule(in_channels)
    
    def forward(self, x):

        x = x+ self.time(x) + self.freq(x)

        return x
