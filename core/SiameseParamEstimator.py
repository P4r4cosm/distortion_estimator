import torch
import torch.nn as nn
import torchaudio.transforms as T

class MelspectrogramLayer(nn.Module):
    def __init__(self, sample_rate=44100, n_mels=128, n_fft=2048, hop_length=512):
        super().__init__()
        self.melspec = T.MelSpectrogram(
            sample_rate=sample_rate, n_fft=n_fft, hop_length=hop_length,
            n_mels=n_mels, f_min=50.0, f_max=14000.0, power=1.0, normalized=True
        )
    def forward(self, x):
        x = x.float()
        return torch.log(self.melspec(x.squeeze(1)) + 1e-6).unsqueeze(1) 

class ExtendedSigmoid(nn.Module):
    def forward(self, x): return (torch.sigmoid(x) * 1.1) - 0.05

class SiameseParamEstimator(nn.Module):
    def __init__(self, num_params=3):
        super().__init__()
        self.to_spec = MelspectrogramLayer() 
        self.shared_encoder = self._build_conv_block()
        self.attention = nn.Sequential(
            nn.Conv2d(128, 1, kernel_size=1), 
            nn.Sigmoid()
        )
        self.time_steps = 2
        self.pool = nn.AdaptiveAvgPool2d((1, self.time_steps))
        self.regressor = nn.Sequential(
            nn.Linear(768, 512), 
            nn.LayerNorm(512), 
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            nn.Linear(512, 128), 
            nn.LayerNorm(128), 
            nn.LeakyReLU(0.2),
            nn.Linear(128, num_params), 
            ExtendedSigmoid()
        )

    def _build_conv_block(self):
        return nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1), 
            nn.GroupNorm(4, 16), 
            nn.LeakyReLU(0.2),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1), 
            nn.GroupNorm(8, 32), 
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), 
            nn.GroupNorm(16, 64), 
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), 
            nn.GroupNorm(32, 128), 
            nn.LeakyReLU(0.2)
        )

    def forward(self, dry_audio, wet_audio):
        spec_dry = self.to_spec(dry_audio)
        spec_wet = self.to_spec(wet_audio)

        feat_dry = self.shared_encoder(spec_dry)
        feat_wet = self.shared_encoder(spec_wet)

        feat_dry = feat_dry * self.attention(feat_dry)
        feat_wet = feat_wet * self.attention(feat_wet)
        
        pooled_dry = self.pool(feat_dry).view(feat_dry.size(0), -1)
        pooled_wet = self.pool(feat_wet).view(feat_wet.size(0), -1)
        
        diff = pooled_wet - pooled_dry 
        emb = torch.cat([pooled_dry, pooled_wet, diff], dim=1) 
        return self.regressor(emb)