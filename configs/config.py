

class Config:
    # Data
    img_size = 256
    in_channels = 3
    sequence_length = 4

    # Model
    base_channels = 32
    memory_channels = 256
    emu_layers = 2

    # Training
    batch_size = 4
    epochs = 50
    lr = 1e-4
    device = "cuda"

    # Loss weights
    lambda_mse = 1.0

    lambda_ssim = 0.5
