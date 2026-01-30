import torch
import os
from torch.utils.data import Dataset

from GeneratorDataset.CreateSpectogramm import audio_to_spectrogram


class ClapDataset(Dataset):
    def __init__(self, folder_positive, folder_negative):
        self.files = []
        self.labels = []

        for f in os.listdir(folder_positive):
            if f.endswith('.wav') and ("db" in f or "noise" in f):
                # if len(f)<15:
                    self.files.append(os.path.join(folder_positive, f))
                    self.labels.append(1.0)

        for f in os.listdir(folder_negative):
            if f.endswith('.wav') and ("db" in f or "noise" in f):
                # if len(f) < 14:
                    self.files.append(os.path.join(folder_negative, f))
                    self.labels.append(0.0)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        label = torch.tensor(self.labels[idx], dtype=torch.float32)

        spec = audio_to_spectrogram(path)
        x = torch.tensor(spec, dtype=torch.float32)
        x = x.unsqueeze(0)

        return x, label