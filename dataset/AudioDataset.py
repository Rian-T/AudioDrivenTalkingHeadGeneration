from torch.utils.data import Dataset
import numpy as np
import torch 
import torch.nn.functional as F
from tqdm import tqdm

from torch.nn.utils.rnn import pad_sequence

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

        self.audios = []
        self.labels = []

        print("Loading audio features")
        
        # Load audio features
        for idx in tqdm(range(len(self.audio_features))):

            audio_path = sorted(glob(self.audio_features[idx] + "*.npy"))
            audios = torch.stack([torch.tensor(np.load(p), dtype=torch.float32)
                                for p in audio_path])

            # Pad audio features
            pad = self.T // 2
            audios = F.pad(audios, (0, 0, 0, 0, pad, pad - 1), 'constant', 0.)
            audios = audios.unfold(0, self.T, 1).permute(0, 3, 1, 2)
            
            #Load A
            A = torch.tensor(np.load(self.A[idx]), dtype=torch.float32)

            self.audios.append(audios)
            self.labels.append(A)
        
        #self.audios = pad_sequence(self.audios)
        #self.labels = pad_sequence(self.labels)

    def __len__(self):
        return len(self.audio_features)

    def __getitem__(self, idx):
        
        return [self.audios[idx], self.labels[idx]]

if __name__ == '__main__':
    dataset = AudioDataset('/data/stars/user/rtouchen/AudioVisualGermanDataset512', 8)
    print(len(dataset))
    print('3rd item audio shape : ' + str(dataset[2][0].shape))
    print('3rd item A shape : ' + str(dataset[2][1].shape))