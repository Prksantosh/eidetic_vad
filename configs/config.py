

class Config:
    DATASET_PATH = "datasets/Avenue"

    SEQ_LEN = 4
    IMG_SIZE = 256

    BATCH_SIZE = 4
    EPOCHS = 100
    LR = 1e-4

    LAMBDA_MSE = 1.0
    LAMBDA_SSIM = 0.5
    LAMBDA_TEMP = 0.2

    DEVICE = "cuda"

