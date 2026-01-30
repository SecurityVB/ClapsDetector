import librosa
import numpy as np
# import matplotlib.pyplot as plt
# import librosa.display
from torch.utils.data import DataLoader

from GeneratorDataset.CreateAugFiles import create_aug_files
from GeneratorDataset.CreateSpectogramm import audio_to_spectrogram
from GeneratorDataset.GenerateDataset import *
from CNN import *
from debugging import *


model = ClapCNN()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

pw = round(667/348, 3)
pos_weight = torch.tensor([pw]).to(device)

# criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
# optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)


dataset = ClapDataset(
    folder_positive='Claps/Data_wav',
    folder_negative='NoClaps/Data_wav'
)

dataloader = DataLoader(
    dataset,
    batch_size=8,
    shuffle=True
)

# model.train()
#
# for epoch in range(50):
#     for x, y in dataloader:
#         x = x.to(device)
#         y = y.to(device)
#
#         optimizer.zero_grad()
#
#         logits = model(x).squeeze(1)
#         loss = criterion(logits, y)
#
#         loss.backward()
#         optimizer.step()
#
#     if epoch % 5 == 0:
#         print(f"Epoch {epoch} | loss = {loss.item():.6f}")
#
# torch.save(model.state_dict(), "weights/new_weights/new_weights_aug.pth")

model.load_state_dict(torch.load("weights/new_weights/new_weights_aug.pth"))
model.eval()

find_optimal_threshold(model, device, dataloader)

prob_file(model)