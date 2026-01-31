import torch
import os
from torch.utils.data import Dataset

from GeneratorDataset.CreateSpectogramm import audio_to_spectrogram

ar = [f"Noise{x}.wav" for x in range(6,30)]

class ClapDataset(Dataset):
    def __init__(self, folder_positive=None, folder_negative=None):
        self.files = []
        self.labels = []

        if folder_positive is not None:
            for f in os.listdir(folder_positive):
                if f.endswith('.wav'):
                    self.files.append(os.path.join(folder_positive, f))
                    self.labels.append(1.0)

        if folder_negative is not None:
            for f in os.listdir(folder_negative):
                if f.endswith('.wav'):# and ("Noise" in f and "_" in f):
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