from torch.utils.data import Dataset
import torch
import numpy as np
from ast import literal_eval


class SongsDataset(Dataset):

    def __init__(self, songs_data):
        song_pitches = songs_data['pitches'].apply(literal_eval)
        song_keys = songs_data['key']
        self.pitches = song_pitches.tolist()
        self.keys = song_keys.tolist()

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        '''print(self.songs_data['key'].iloc[[idx]])
        pitches = self.songs_data.iloc[[idx]].at['pitches']
        print(pitches, type(pitches), len(pitches))
        keys = self.songs_data.iloc[[idx]].at['key']'''

        #sample = {'pitches': torch.tensor(self.pitches[idx]), 'keys': torch.tensor(self.keys[idx])}
        pitches = torch.tensor(self.pitches[idx])
        keys = torch.tensor(self.keys[idx])

        return pitches, keys
