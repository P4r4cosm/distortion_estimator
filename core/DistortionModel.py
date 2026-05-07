
import torch
import torch.nn as nn
from torch.nn.utils import weight_norm

torch.backends.cudnn.benchmark = True


class FiLM(nn.Module):
    def __init__(self, num_features, cond_dim):
        super().__init__()
        self.adaptor = nn.Linear(cond_dim, num_features * 2)

    def forward(self, x, cond):
        out = self.adaptor(cond)
        gamma, beta = out.chunk(2, dim=1)
        return x * gamma.unsqueeze(2) + beta.unsqueeze(2)

class ResidualBlock(nn.Module):
    def __init__(self, channels, dilation, cond_dim):
        super().__init__()
        self.dilated_conv = weight_norm(nn.Conv1d(channels, channels * 2, kernel_size=3, padding=dilation, dilation=dilation))
        self.film = FiLM(channels, cond_dim) 
        self.output_conv = weight_norm(nn.Conv1d(channels, channels, kernel_size=1))
        
    def forward(self, x, params):
        residual = x
        x = self.dilated_conv(x)
        filter_x, gate_x = x.chunk(2, dim=1)
        x = torch.tanh(filter_x) * torch.sigmoid(gate_x)
        x = self.film(x, params)
        x = self.output_conv(x)
        return x + residual

class DistortionModel(nn.Module):
    def __init__(self, num_params=3):
        super().__init__()
        self.ch = 32 
        self.input_conv = weight_norm(nn.Conv1d(1, self.ch, kernel_size=1))
        self.blocks = nn.ModuleList()
        
        dilations = [1, 2, 4, 8, 16, 32, 64, 128, 256] * 3
        
        for d in dilations:
            self.blocks.append(ResidualBlock(self.ch, d, num_params))
            
        self.output_conv = nn.Sequential(
            nn.LeakyReLU(0.2),
            weight_norm(nn.Conv1d(self.ch, 16, kernel_size=1)),
            nn.LeakyReLU(0.2),
            weight_norm(nn.Conv1d(16, 1, kernel_size=1)),
            nn.Tanh() 
        )
        self.output_gain = nn.Linear(num_params, 1) 

    def forward(self, x, params):
        x = self.input_conv(x)
        skips = 0
        for block in self.blocks:
            x = block(x, params)
            skips = skips + x 
        
        audio_out = self.output_conv(skips)
        gain = self.output_gain(params).unsqueeze(2) 
        return audio_out * gain

