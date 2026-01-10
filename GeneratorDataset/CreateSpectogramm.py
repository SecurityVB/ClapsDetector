import librosa
import numpy as np
import matplotlib.pyplot as plt
import librosa.display


SR = 16000
DURATION = 1.0

N_FFT = 512
HOP = 128
N_MELS = 64

FMIN = 400
FMAX = 8000


def audio_to_spectrogram(path):
    y, _ = librosa.load(path, sr=SR, mono=True)
    target_len = int(SR * DURATION)

    if len(y) > target_len:
        y = y[:target_len]
    else:
        y = np.pad(y, (0, target_len - len(y)))

    mel = librosa.feature.melspectrogram(
        y=y,
        sr=SR,
        n_fft=N_FFT,
        hop_length=HOP,
        n_mels=N_MELS,
        fmin=FMIN,
        fmax=FMAX,
        power=2.0
    )

    mel_db = librosa.power_to_db(mel, ref=np.max, top_db=80)
    mel_db = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-6)

    return mel_db

# spec = audio_to_spectrogram('Claps/Data_wav/Clap1_db_-15.wav')
#
# plt.figure(figsize=(6, 4))
# librosa.display.specshow(
#     spec,
#     sr=SR,
#     hop_length=HOP,
#     x_axis="time",
#     y_axis="mel",
#     cmap="magma"
# )
#
# plt.colorbar(label="z-normalized dB")
# plt.title("Log-Mel Spectrogram")
# plt.tight_layout()
# plt.show()