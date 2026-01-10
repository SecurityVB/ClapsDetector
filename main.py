import librosa
import numpy as np
import matplotlib.pyplot as plt
import librosa.display
from torch.utils.data import DataLoader

from GeneratorDataset.GenerateDatasetFiles import create_dataset
from GeneratorDataset.CreateSpectogramm import audio_to_spectrogram
from GeneratorDataset.GenerateDataset import *
from CNN import *
from checkfile import g

model = ClapCNN()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)


dataset = ClapDataset(
    folder_positive='Claps/Data_wav',
    folder_negative='NoClaps/Data_wav'
)

dataloader = DataLoader(
    dataset,
    batch_size=8,
    shuffle=True
)


model.train()

for x, y in dataloader:
    x = x.to(device)   # (B, 1, 64, T)
    y = y.to(device)   # (B,)

    optimizer.zero_grad()

    logits = model(x)      # (B, 1)
    logits = logits.squeeze(1)

    loss = criterion(logits, y)

    loss.backward()
    optimizer.step()

model.load_state_dict(torch.load("model_weights.pth"))
model.eval()

g(model)