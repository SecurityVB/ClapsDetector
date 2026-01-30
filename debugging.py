import torch
import numpy as np
from pydub import AudioSegment

from GeneratorDataset.CreateSpectogramm import audio_to_spectrogram


def prob_file(model, path_to_file='ClapExample.wav'):
    spec = audio_to_spectrogram(path_to_file)

    x = torch.tensor(spec, dtype=torch.float32)
    x = x.unsqueeze(0)
    x = x.unsqueeze(0)

    with torch.no_grad():
        logit = model(x)
        prob = torch.sigmoid(logit)

    threshold = 0.5
    is_two_claps = prob.item() > threshold

    print(f"Probability 2 claps: {prob.item():.3f}")
    print("Two claps?" , "yes" if is_two_claps else "no")


def prob_true_dataloader(model, device, val_loader):
    y_true = []
    y_prob = []

    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(device)
            logits = model(x).squeeze(1)
            probs = torch.sigmoid(logits)

            y_true.extend(y.cpu().numpy())
            y_prob.extend(probs.cpu().numpy())

    return (y_true, y_prob)


def precision_recall(y_true, y_prob, threshold):
    y_pred = (np.array(y_prob) > threshold).astype(int)
    y_true = np.array(y_true).astype(int)

    TP = np.sum((y_pred == 1) & (y_true == 1))
    FP = np.sum((y_pred == 1) & (y_true == 0))
    FN = np.sum((y_pred == 0) & (y_true == 1))

    precision = TP / (TP + FP + 1e-8)
    recall = TP / (TP + FN + 1e-8)

    return precision, recall


def get_statistics(y_true, y_prob, threshold):
    p, r = precision_recall(y_true, y_prob, threshold)
    print(f"thr={threshold:.2f} | precision={p:.3f} | recall={r:.3f}")


def find_optimal_threshold(model, device, dataloader):
    y_true_prob = prob_true_dataloader(model, device, dataloader)
    y_true = y_true_prob[0]
    y_prob = y_true_prob[1]
    for threshold in np.linspace(0.1, 0.9, 9):
        get_statistics(y_true, y_prob, threshold)