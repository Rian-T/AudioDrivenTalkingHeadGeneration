from torch.utils.data import Dataset
import numpy as np
import torch 
import torch.nn.functional as F

from glob import glob

class AudioDataset(Dataset):
    """Audio dataset."""

    def __init__(self, data_path, T):
        """
        Args:
            data_path (string): Path to the dataset with audio features and A.
        """
        self.data_path = data_path
        self.audio_features = sorted(glob(self.data_path + "/audio_features/*/"))
        self.A = sorted(glob(self.data_path + "/w512/*.npy"))
        self.T = T

    def __len__(self):
        return len(self.audio_features)

    def __getitem__(self, idx):
        
        # Load audio features
        audio_path = sorted(glob(self.audio_features[idx] + "*.npy"))
        audios = torch.stack([torch.tensor(np.load(p), dtype=torch.float32)
                              for p in audio_path])

        # Pad audio features
        pad = self.T // 2
        audios = F.pad(audios, (0, 0, 0, 0, pad, pad - 1), 'constant', 0.)
        audios = audios.unfold(0, self.T, 1).permute(0, 3, 1, 2)

        #Load A
        A = torch.tensor(np.load(self.A[idx]), dtype=torch.float32)

        item = {'audio': audios, 'A': A}

        return item

if __name__ == '__main__':
    dataset = AudioDataset('/data/stars/user/rtouchen/AudioVisualGermanDataset512', 8)
    print(len(dataset))
    print('100th item audio shape : ' + str(dataset[100]['audio'].shape))
    print('100th item A shape : ' + str(dataset[100]['A'].shape))