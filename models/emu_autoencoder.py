import torch.nn as nn
from models.encoder import EncoderStage
from models.decoder import DecoderStage
from models.emu_timestamp import TimestampTransform
from models.improved3demu import ImprovedE3DLSTM

###############################################
# Full Autoencoder Model
###############################################
class RHCNetAutoencoder(nn.Module):

    def __init__(self, seq_len=4):

        super().__init__()

        self.seq_len = seq_len

        # Initial feature extraction
        self.initial = nn.Conv2d(3, 64, 3, padding=1)

        # Encoder
        self.enc1 = EncoderStage(64, 64)
        self.enc2 = EncoderStage(64, 128)
        self.enc3 = EncoderStage(128, 256)
        self.enc4 = EncoderStage(256, 512)

        # Timestamp transformation
        self.timestamp = TimestampTransform()

        # Temporal modeling
        self.e3d = E3DLSTM(512, 512)
        
        # Decoder
        self.dec1 = DecoderStage(512, 256)
        self.dec2 = DecoderStage(256, 128)
        self.dec3 = DecoderStage(128, 64)
        self.dec4 = DecoderStage(64, 32)

        # Final reconstruction
        self.final = nn.Conv2d(32, 3, 3, padding=1)


    def forward(self, x):

        B, T, C, H, W = x.shape

        # --------------------------------
        # Merge batch and time
        # --------------------------------
        x = x.view(B * T, C, H, W)

        # --------------------------------
        # Initial feature extraction
        # --------------------------------
        x = self.initial(x)

        # --------------------------------
        # Encoder
        # --------------------------------
        x = self.enc1(x)
        x = self.enc2(x)
        x = self.enc3(x)
        x = self.enc4(x)

        # --------------------------------
        # Convert to spatiotemporal tensor
        # (B,C,T,H,W)
        # --------------------------------
        x = self.timestamp(x, B, T)

        # --------------------------------
        # E3D-LSTM Temporal Modeling
        # --------------------------------
        x = self.e3d(x)

        # Take last time step feature map
        x = x[:, :, -1]

        # --------------------------------
        # Decoder
        # --------------------------------
        x = self.dec1(x)
        x = self.dec2(x)
        x = self.dec3(x)
        x = self.dec4(x)

        # --------------------------------
        # Final output frame
        # --------------------------------
        x = self.final(x)

        return x
