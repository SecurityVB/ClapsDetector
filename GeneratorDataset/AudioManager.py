import os
from pydub import AudioSegment
import numpy as np

from config import *

def change_volume(audio, db_change):
    db_audio = audio + db_change
    return db_audio

def add_noise(audio, noise_level=0.01):
    samples = np.array(audio.get_array_of_samples())
    noise = np.random.randn(len(samples)) * (2 ** 15 - 1) * noise_level
    noisy_samples = samples + noise
    noisy_audio = audio._spawn(noisy_samples.astype(np.int16).tobytes())
    return noisy_audio

def save(name, audio, path_d):
    audio.export(path_d+name, format="wav")