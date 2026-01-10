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

def convert_audio_to_wav(file):
    audio = AudioSegment.from_file(path_m4a+file, format="m4a")
    new_path = path_wav + file.split(".")[0] + ".wav"
    audio.export(new_path, format="wav")
    audio2 = AudioSegment.from_file(new_path, format="wav")
    return audio2

def save(name, audio):
    audio.export(path_wav+name, format="wav")