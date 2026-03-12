# AE-RHCNet Video Anomaly Detection

PyTorch implementation of a hybrid spatiotemporal autoencoder for video anomaly detection.

The architecture combines:

- Residual Hybrid Convolution Network (RHCNet)
- Attention Enhanced RHCNet (AE-RHCNet)
- E3D-LSTM based spatiotemporal modeling

The model predicts the next frame given previous frames and detects anomalies using reconstruction error.

## Architecture

Encoder:
RHCNet-based hierarchical feature extraction.

Temporal modeling:
3D-LSTM / ConvLSTM/ E3D-LSTM.

Decoder:
Attention Enhanced RHCNet blocks.


## Dataset

Experiments were conducted on:

- CUHK Avenue dataset
- ShanghaiTech dataset
- UBnormal dataset


Dataset structure:
<img width="5480" height="1805" alt="Untitled Diagram" src="https://github.com/user-attachments/assets/4807dfdf-826d-4eaf-83d0-22742cfe714e" />


 
## Training

```bash
python train.py
