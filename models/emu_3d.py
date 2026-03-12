import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms.functional import resize
from PIL import Image 

class E3DLSTMCell(nn.Module):
    """E3D-LSTM Cell with 3D convolutions for spatiotemporal memory."""
    def __init__(self, in_channels, hidden_channels, kernel_size=(3, 3, 3)):
        super().__init__()
        self.hidden_channels = hidden_channels

        # Input modulation gates (3D convs)
        self.conv_xi = nn.Conv3d(in_channels, hidden_channels, kernel_size, padding=1)
        self.conv_hi = nn.Conv3d(hidden_channels, hidden_channels, kernel_size, padding=1, bias=False)
        
        # Forget gate
        self.conv_xf = nn.Conv3d(in_channels, hidden_channels, kernel_size, padding=1)
        self.conv_hf = nn.Conv3d(hidden_channels, hidden_channels, kernel_size, padding=1, bias=False)
        
        # Output gate
        self.conv_xo = nn.Conv3d(in_channels, hidden_channels, kernel_size, padding=1)
        self.conv_ho = nn.Conv3d(hidden_channels, hidden_channels, kernel_size, padding=1, bias=False)
        
        # Cell state
        self.conv_xc = nn.Conv3d(in_channels, hidden_channels, kernel_size, padding=1)
        self.conv_hc = nn.Conv3d(hidden_channels, hidden_channels, kernel_size, padding=1, bias=False)

    def forward(self, x, h_prev, c_prev):
        # x: (B, C, D, H, W), h_prev/c_prev: (B, hidden_channels, D, H, W)
        i = torch.sigmoid(self.conv_xi(x) + self.conv_hi(h_prev))
        f = torch.sigmoid(self.conv_xf(x) + self.conv_hf(h_prev))
        o = torch.sigmoid(self.conv_xo(x) + self.conv_ho(h_prev))
        c = f * c_prev + i * torch.tanh(self.conv_xc(x) + self.conv_hc(h_prev))
        h = o * torch.tanh(c)
        return h, c

class E3DLSTM(nn.Module):
    """2-layer E3D-LSTM for sequential processing."""
    def __init__(self, in_channels, hidden_channels):
        super().__init__()
        self.lstm1 = E3DLSTMCell(in_channels, hidden_channels)
        self.lstm2 = E3DLSTMCell(hidden_channels, hidden_channels)

    def forward(self, x):
        # x: (B, C, T, H, W)
        B, C, T, H, W = x.shape
        h1 = torch.zeros(B, self.lstm1.hidden_channels, T, H, W).to(x.device)
        c1 = torch.zeros(B, self.lstm1.hidden_channels, T, H, W).to(x.device)
        h2 = torch.zeros(B, self.lstm2.hidden_channels, T, H, W).to(x.device)
        c2 = torch.zeros(B, self.lstm2.hidden_channels, T, H, W).to(x.device)
        
        # Process all timesteps
        h1, c1 = self.lstm1(x, h1, c1)
        h2, c2 = self.lstm2(h1, h2, c2)
        
        return h2  # (B, hidden_channels, T, H, W)

