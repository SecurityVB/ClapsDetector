import numpy as np
from scipy.signal import stft

from config import *


def hz_to_mel(hz):
    return 2595 * np.log10(1 + hz / 700)

def mel_to_hz(mel):
    return 700 * (10**(mel / 2595) - 1)

def mel_filterbank(sr, n_fft, n_mels, fmin, fmax):
    mel_min = hz_to_mel(fmin)
    mel_max = hz_to_mel(fmax)

    mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_points = mel_to_hz(mel_points)

    bins = np.floor((n_fft + 1) * hz_points / sr).astype(int)

    filterbank = np.zeros((n_mels, n_fft // 2 + 1))

    for i in range(1, n_mels + 1):
        left = bins[i - 1]
        center = bins[i]
        right = bins[i + 1]

        for j in range(left, center):
            filterbank[i - 1, j] = (j - left) / (center - left)
        for j in range(center, right):
            filterbank[i - 1, j] = (right - j) / (right - center)

    return filterbank

MEL_FILTER = mel_filterbank(SR, N_FFT, N_MELS, FMIN, FMAX)



def audio_array_to_spectrogram(y):
    target_len = int(SR * DURATION)

    if len(y) > target_len:
        y = y[:target_len]
    else:
        y = np.pad(y, (0, target_len - len(y)))

    _, _, Zxx = stft(
        y,
        fs=SR,
        nperseg=N_FFT,
        noverlap=N_FFT - HOP,
        window="hann",
        padded=False,
        boundary=None,
    )

    S = np.abs(Zxx) ** 2

    mel = np.dot(MEL_FILTER, S)

    mel = np.maximum(mel, 1e-10)
    mel_db = 10.0 * np.log10(mel)
    mel_db -= np.max(mel_db)

    mel_db = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-6)

    return mel_db