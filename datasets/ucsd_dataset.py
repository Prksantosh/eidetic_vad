import os
import glob
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class UCSDPed2(Dataset):
    """
    UCSD Ped2 Dataset Loader for Prediction-Based VAD

    Returns:
        input_sequence:  (T, C, H, W)
        target_frame:    (C, H, W)
        label (optional): 0/1 (test mode only)
    """

    def __init__(
        self,
        root_dir,
        sequence_length=8,
        split="train",
        resize=256,
        return_label=False
    ):
        super().__init__()

        self.root_dir = root_dir
        self.sequence_length = sequence_length
        self.split = split.lower()
        self.resize = resize
        self.return_label = return_label

        assert self.split in ["train", "test"]

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((resize, resize)),
        ])

        self.sequences = []
        self.gt_labels = []

        self._prepare_data()

    # ---------------------------------------------------
    # Prepare sequences
    # ---------------------------------------------------
    def _prepare_data(self):

        if self.split == "train":
            data_dir = os.path.join(self.root_dir, "Train")
        else:
            data_dir = os.path.join(self.root_dir, "Test")

        folders = sorted(os.listdir(data_dir))

        for folder in folders:

            folder_path = os.path.join(data_dir, folder)
            frame_paths = sorted(glob.glob(os.path.join(folder_path, "*.tif")))

            if len(frame_paths) < self.sequence_length + 1:
                continue

            # Load GT for test
            if self.split == "test" and self.return_label:
                gt_path = os.path.join(
                    self.root_dir,
                    "Test_gt",
                    f"{folder}_gt.npy"
                )
                gt = np.load(gt_path)
            else:
                gt = None

            # Sliding window
            for i in range(len(frame_paths) - self.sequence_length):

                input_seq = frame_paths[i:i + self.sequence_length]
                target = frame_paths[i + self.sequence_length]

                self.sequences.append((input_seq, target))

                if gt is not None:
                    self.gt_labels.append(gt[i + self.sequence_length])

    # ---------------------------------------------------
    def __len__(self):
        return len(self.sequences)

    # ---------------------------------------------------
    def __getitem__(self, idx):

        input_seq_paths, target_path = self.sequences[idx]

        input_frames = []

        for path in input_seq_paths:
            frame = cv2.imread(path, cv2.IMREAD_COLOR)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = self.transform(frame)
            input_frames.append(frame)

        input_frames = torch.stack(input_frames)  # (T, C, H, W)

        target_frame = cv2.imread(target_path, cv2.IMREAD_COLOR)
        target_frame = cv2.cvtColor(target_frame, cv2.COLOR_BGR2RGB)
        target_frame = self.transform(target_frame)

        if self.split == "test" and self.return_label:
            label = self.gt_labels[idx]
            return input_frames, target_frame, label


        return input_frames, target_frame
