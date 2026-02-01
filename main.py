import numpy as np
import sounddevice as sd
import queue

import torch

from CNN import ClapCNN
from config import *
from GeneratorDataset.CreateSpectogramm import audio_array_to_spectrogram

audio_queue = queue.Queue()

def audio_callback(indata, frames, time, status):
    if status:
        print(status)
    audio_queue.put(indata.copy())

c=0

model = ClapCNN()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.load_state_dict(torch.load("weights/weights_2/weights_aug.pth"))
model.eval()

buffer = np.zeros(0, dtype=np.float32)
TARGET_SAMPLES = int(SR * DURATION)

with sd.InputStream(
    samplerate=SR,
    channels=1,
    dtype='float32',
    callback=audio_callback
):
    print("🎤 Слушаю микрофон...")

    while True:
        data = audio_queue.get()
        data = data.flatten()

        buffer = np.concatenate([buffer, data])

        while len(buffer) >= TARGET_SAMPLES:
            chunk = buffer[:TARGET_SAMPLES]
            buffer = buffer[TARGET_SAMPLES:]

            spec = audio_array_to_spectrogram(chunk)

            x = torch.tensor(spec, dtype=torch.float32)
            x = x.unsqueeze(0)
            x = x.unsqueeze(0)

            with torch.no_grad():
                logit = model(x)
                prob = torch.sigmoid(logit)

            threshold = 0.6
            is_two_claps = prob.item() > threshold
            print(f"Probability 2 claps {c}: {prob.item():.3f}")
            if is_two_claps:
                print("Two claps 👏")