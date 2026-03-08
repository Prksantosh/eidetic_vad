import torch
import torch.nn as nn
from models.encoder import Encoder
from models.decoder import Decoder
from models.timestamp import timestamp_transform
from models.emu_3d import EMU3D

class RHCNetEMU(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.encoder = Encoder(config.in_channels, config.base_channels)
        self.emu = EMU3D(config.base_channels*4)
        self.decoder = Decoder(config.base_channels, config.in_channels)

    def forward(self, x):
        B, T, C, H, W = x.shape

        # Encode each frame
        enc_feats = []
        for t in range(T):
            enc = self.encoder(x[:, t])
            enc_feats.append(enc)

        enc_feats = torch.stack(enc_feats, dim=1)
        enc_feats = timestamp_transform(enc_feats)

        prev_c = torch.zeros_like(enc_feats)
        h, c = self.emu(enc_feats, prev_c)

        # Take last timestep
        last_feat = h[:, :, -1]
        out = self.decoder(last_feat)


        return out
