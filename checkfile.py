import torch

from GeneratorDataset.CreateSpectogramm import audio_to_spectrogram

def g(model):
    spec = audio_to_spectrogram('ClapExample.m4a')

    x = torch.tensor(spec, dtype=torch.float32)
    x = x.unsqueeze(0)
    x = x.unsqueeze(0)

    with torch.no_grad():
        logit = model(x)
        print(logit.shape)
        prob = torch.sigmoid(logit)

    threshold = 0.4
    is_two_claps = prob.item() > threshold

    print(f"Вероятность двух хлопков: {prob.item():.3f}")
    print("Два хлопка?" , "ДА" if is_two_claps else "НЕТ")