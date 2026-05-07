import torch.nn as nn
import torchaudio.transforms as T
import torch

class Distortion2DNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.mel_transform = T.MelSpectrogram(sample_rate=44100, n_fft=1024, hop_length=512, n_mels=256)
        self.amplitude_to_db = T.AmplitudeToDB()
        self.features = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=3, padding=1), nn.InstanceNorm2d(32, affine=True), nn.ReLU(), nn.MaxPool2d(2), 
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.InstanceNorm2d(64, affine=True), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.InstanceNorm2d(128, affine=True), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1), nn.InstanceNorm2d(256, affine=True), nn.ReLU(), nn.AdaptiveAvgPool2d((1, 1))
        )
        self.regressor = nn.Sequential(
            nn.Flatten(), nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, 3), nn.Sigmoid()
        )

    def preprocess_spec(self, x):
        spec = self.mel_transform(x)
        spec = self.amplitude_to_db(spec)
        MIN_DB, MAX_DB = -80.0, 20.0
        spec = torch.clamp(spec, MIN_DB, MAX_DB)
        spec = (spec - MIN_DB) / (MAX_DB - MIN_DB)
        return spec

    def forward(self, dry, wet):
        s_dry = self.preprocess_spec(dry)
        s_wet = self.preprocess_spec(wet)
        x = torch.cat([s_dry.unsqueeze(1), s_wet.unsqueeze(1)], dim=1)
        return self.regressor(self.features(x))