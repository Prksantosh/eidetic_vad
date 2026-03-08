import torch
from torch.utils.data import DataLoader

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
train_dataset = UCSDPed2(
    root_dir="UCSD",
    sequence_length=config.sequence_length,
    split="train"
)

dataloader = DataLoader(
    train_dataset,
    batch_size=config.batch_size,
    shuffle=True,
    num_workers=4,
    pin_memory=True
)


# --------------------------------------------------
# Model
# --------------------------------------------------
model = RHCNetEMU(config).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
criterion = PredictionLoss()


# --------------------------------------------------
# Training Loop
# --------------------------------------------------
for epoch in range(config.epochs):

    model.train()
    total_loss = 0

    for batch in dataloader:

        x, target = batch
        x = x.to(device)
        target = target.to(device)

        pred = model(x)

        loss = criterion(pred, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()


    print(f"Epoch [{epoch+1}/{config.epochs}]  Loss: {total_loss/len(dataloader):.6f}")

