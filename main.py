# import librosa
# import numpy as np
from pydub import AudioSegment
import os

from GenerateDataSet import *
from config import *


# SR = 22050
# N_FFT = 2048
# HOP = 512
# DURATION = 1.0
#
# def audio_to_spectrogram(path):
#     y, sr = librosa.load(path, sr=SR)
#
#     target_len = int(SR * DURATION)
#
#     # жёстко приводим к 1 секунде
#     if len(y) > target_len:
#         y = y[:target_len]
#     elif len(y) < target_len:
#         y = np.pad(y, (0, target_len - len(y)))
#
#     S = np.abs(librosa.stft(
#         y,
#         n_fft=N_FFT,
#         hop_length=HOP
#     ))
#
#     spec = librosa.amplitude_to_db(S, ref=np.max)
#
#     # нормализация — ОБЯЗАТЕЛЬНО
#     spec = (spec - spec.mean()) / spec.std()
#
#     return spec

for file in os.listdir(path_m4a):
    if file.endswith("m4a"):
        audio = AudioSegment.from_file(path_m4a+file, format="m4a")
        for db in range(-15, 16, 5):
            aug_audio = change_volume(audio, db)
            save(f'db_{db}.wav', aug_audio)
        for i in (0.01, 0.02, 0.03):
            aug_audio = add_noise(audio, i)
            save(f'Noisy_{i}.wav', aug_audio)
        for db in range(-15, 16, 5):
            if db != 0:
                for i in (0.01, 0.02, 0.03):
                    aug_audio = change_volume(add_noise(audio, i), db)
                    save(f'Noisy_{i}_db_{db}.wav', aug_audio)

# print(audio_to_spectrogram('Clap2.wav').shape)