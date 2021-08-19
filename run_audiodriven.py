import torch
import torch.nn.functional as F
import numpy as np
import pytorch_lightning as pl

from models.models import AudioExpressionNet3

from glob import glob
from tqdm import tqdm

data_path = "/data/stars/user/rtouchen/AudioVisualGermanDataset512/"
T = 8

audio_features = sorted(glob(data_path + "/audio_features/*/"))

pretrained_model = AudioExpressionNet3.load_from_checkpoint('checkpoints/audiodriven/gitg0x5k/checkpoints/epoch=29-step=1679.ckpt')
pretrained_model.eval()

for idx in tqdm(range(len(audio_features))):

    audio_path = sorted(glob(audio_features[idx] + "*.npy"))
    audios = torch.stack([torch.tensor(np.load(p), dtype=torch.float32)
                            for p in audio_path])
    
    # audio_shape [nb_frames, 16, 29]

    A_files = sorted(glob(data_path + "/w512/*.npy"))
    A = torch.tensor(np.load(A_files[idx]), dtype=torch.float32)

    # A shape [nb_frames, 1, 20]

    # Pad audio features
    pad = T // 2
    audios = F.pad(audios, (0, 0, 0, 0, pad, pad - 1), 'constant', 0.)
    audios = audios.unfold(0, T, 1).permute(0, 3, 1, 2)

    # audio_shape after padding [nb_frames, T, 16, 29]

    predictions = []

    for audio in tqdm(audios, leave=False):     # audio shape : [T, 16, 29]
        input_audio = torch.unsqueeze(audio,0)  # [1, T, 16, 29]

        pred = pretrained_model(input_audio)    # [1, 1, 20]
        pred = pred.squeeze(0)                  # [1, 20]
        predictions.append(pred)

    predictions = torch.stack(predictions)      # [nb_frames, 1, 20]
    preds =  predictions.detach().cpu().numpy()
    np.save(data_path + f'/a512/{idx}.npy', preds)