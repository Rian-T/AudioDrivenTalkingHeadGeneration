from torch.utils.data import Dataset
import numpy as np
import torch 
import torch.nn.functional as F
from tqdm import tqdm

import os.path

from glob import glob
from random import randrange


class AudioDataset(Dataset):
    """Audio dataset."""

    def __init__(self, data_path, T):
        """
        Args:
            data_path (string): Path to the dataset with audio features and A.
        """
        self.data_path = data_path
        self.T = T

        self.audio_features = sorted(glob(self.data_path + "/audio_features/*/"))
        self.A = sorted(glob(self.data_path + "/w512/*.npy"))


        self.audios = []
        self.labels = []

        print("Loading audio features")

        # Load audio features
        for idx in tqdm(range(len(self.audio_features))):
            
            if os.path.exists(self.audio_features[idx] + "audios.pt"):
                audios = torch.load(self.audio_features[idx] + "audios.pt")

            else: #Very slow
                audio_path = sorted(glob(self.audio_features[idx] + "*.npy"))
                audios = torch.stack([torch.tensor(np.load(p), dtype=torch.float32)[:, :32]
                                    for p in audio_path])
                
                torch.save(audios, self.audio_features[idx] + "audios.pt")

            #Load A
            A = torch.tensor(np.load(self.A[idx]), dtype=torch.float32)

            for x in range((len(audios)-self.T)//self.T):
                end = x*self.T + self.T
                self.audios.append(audios[end-self.T:end])
                self.labels.append(A[end-(self.T//2)])
        

    def __len__(self):
        return len(self.audios)

    def __getitem__(self, idx):
        
        return [self.audios[idx], self.labels[idx]]



class AudioDatasetLazy(Dataset): # Same as AudioDataset but loads files when needed rather than loading everything at init
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
        
        
        audio_path = sorted(glob(self.audio_features[idx] + "*.npy"))
        audios = torch.stack([torch.tensor(np.load(p), dtype=torch.float32)[:, :32]
                            for p in audio_path])

        end = randrange(self.T,len(audios))
        audios = audios[end-self.T:end]
        
        #Load A
        A = torch.tensor(np.load(self.A[idx]), dtype=torch.float32)
        A = A[end-(self.T//2)]

        return [audios, A]


if __name__ == '__main__':
    dataset = AudioDataset('/data/stars/user/rtouchen/AudioVisualGermanDataset512', 8)
    
    print(len(dataset))
    print('3rd item audio shape : ' + str(dataset[2][0].shape))
    print('3rd item A shape : ' + str(dataset[2][1].shape))