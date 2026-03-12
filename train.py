import torch
import os
import glob
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np


from configs.config import Config
from models.rhcnet_emu import RHCNetEMU
from losses.losses import PredictionLoss
from datasets.ucsd_dataset import UCSDPed2

# --------------------------------------------------
# Config
# --------------------------------------------------
config = Config()
device = torch.device(config.device if torch.cuda.is_available() else "cpu")

# --------------------------------------------------
# Dataset + Dataloader
# --------------------------------------------------
train_dataset = AvenueDataset(
    root_dir="Avenue/training_frames",
    seq_len=4,
    transform=transform
)

train_loader = DataLoader(
    train_dataset,
    batch_size=4,
    shuffle=True,
    num_workers=0
)

model = RHCNetAutoencoder(seq_len=4).to(device)

mse_loss = nn.MSELoss()
ssim_loss = SSIMLoss()
temporal_loss = TemporalLoss()

lambda_mse = 0.85
lambda_ssim = 0.1
lambda_temp = 0.05

optimizer = optim.Adam(
    model.parameters(),
    lr=1e-4,
    weight_decay=1e-5
)

scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer,
    step_size=20,
    gamma=0.5
)
# --------------------------------------------------
# Training Loop
# --------------------------------------------------
epochs = 100

for epoch in range(epochs):

    model.train()

    total_loss = 0
    total_mse = 0
    total_ssim = 0
    total_temp = 0

    for frames, target in train_loader:

        frames = frames.to(device)
        target = target.to(device)

        optimizer.zero_grad()

        pred = model(frames)

        prev_frame = frames[:, -1]

        # Individual losses
        loss_mse = mse_loss(pred, target)
        loss_ssim = ssim_loss(pred, target)
        loss_temp = temporal_loss(pred, prev_frame, target)

        # Total loss
        loss = (
            lambda_mse * loss_mse +
            lambda_ssim * loss_ssim +
            lambda_temp * loss_temp
        )

        loss.backward()
        optimizer.step()

        # Accumulate losses
        total_loss += loss.item()
        total_mse += loss_mse.item()
        total_ssim += loss_ssim.item()
        total_temp += loss_temp.item()

    scheduler.step()

    avg_loss = total_loss / len(train_loader)
    avg_mse = total_mse / len(train_loader)
    avg_ssim = total_ssim / len(train_loader)
    avg_temp = total_temp / len(train_loader)

    print(
        f"Epoch [{epoch+1}/{epochs}] | "
        f"Total Loss: {avg_loss:.4f} | "
        f"MSE: {avg_mse:.4f} | "
        f"SSIM: {avg_ssim:.4f} | "
        f"Temporal: {avg_temp:.4f}"
    )

    # Save checkpoint
    torch.save(
        model.state_dict(),
        f"rhcnet_avenue_epoch_{epoch+1}.pth"
    )

