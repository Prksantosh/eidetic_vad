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
- ShanghiTech dataset
- Ubnormal dataset

Dataset structure:

Avenue/
    training_frames/
    testing_frames/

---

## Training

```bash
python train.py
