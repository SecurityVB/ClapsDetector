from torch.utils.data import DataLoader
from GeneratorDataset.GenerateDataset import *
from CNN import *
from debugging import *
import os
from config import *

count_positive = len([f for f in os.listdir(path_claps_wav) if os.path.isfile(os.path.join(path_claps_wav, f))])
count_negative = len([f for f in os.listdir(path_Noclaps_wav) if os.path.isfile(os.path.join(path_Noclaps_wav, f))])


model = ClapCNN()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

model.load_state_dict(torch.load("weights/weights_2/weights.pth"))

pw = round(4/4, 3)
pos_weight = torch.tensor([pw]).to(device)

criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)


dataset = ClapDataset(
    folder_positive=path_claps_wav,
    folder_negative=path_Noclaps_wav
)

dataloader = DataLoader(
    dataset,
    batch_size=8,
    shuffle=True
)

model.train()

for epoch in range(50):
    for x, y in dataloader:
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()

        logits = model(x).squeeze(1)
        loss = criterion(logits, y)

        loss.backward()
        optimizer.step()

    if epoch % 5 == 0:
        print(f"Epoch {epoch} | loss = {loss.item():.6f}")

torch.save(model.state_dict(), "weights/weights_2/weights_aug.pth")