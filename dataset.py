import torch
import os
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from skimage import io


class FaceLandmarksDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, csv_file, root, transform=None, idx_from_to=None):
        """
        Args:
            csv_file (string): Path to the csv_file with annotations.
            root (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
            idx_from_to (tuple, optional): Index for a part of the whole dataset that should be taken.
        """
        self.landmarks_frame = pd.read_csv(csv_file)
        if idx_from_to is not None:
            idx_from, idx_to = idx_from_to
            self.landmarks_frame = self.landmarks_frame[idx_from:idx_to]
        self.root = root
        self.transform = transform

    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root, self.landmarks_frame.iloc[idx, 0])
        image = io.imread(img_name)
        landmarks = self.landmarks_frame.iloc[idx, 1:]
        landmarks = np.array([landmarks])
        landmarks = landmarks.astype('float').reshape(-1, 2)
        sample = {'image': image, 'landmarks': landmarks}
        if self.transform:
            sample = self.transform(sample)
        return sample
