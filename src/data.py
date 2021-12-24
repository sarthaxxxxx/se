import os
import sys
import numpy as np
from torch.utils.data import Dataset, DataLoader

sys.path.append('/media/sarthak/Data/SER/code/')

from utils.signal_preprocess import _read_slice_
from utils.cfg_loader import *


class DATA(Dataset):
    def __init__(self, cfg, clean_path, noisy_path):
        self.cfg = cfg
        self.clean_path, self.noisy_path = clean_path, noisy_path
        self.clean_files = [os.path.join(self.clean_path, file_path) for file_path in sorted(os.listdir(self.clean_path))]
        self.noisy_files = [os.path.join(self.noisy_path, file_path) for file_path in sorted(os.listdir(self.noisy_path))]

    def __len__(self):
        return len(self.clean_files)

    def __getitem__(self, index):
        print("PROCESSING WAV FILE {}/{} : {}".format(index, len(self.clean_files), self.clean_files[index]))
        clean_chunks = _read_slice_(self.clean_files[index], 
                                    self.cfg.window, 
                                    self.cfg.hop_per)
        noisy_chunks = _read_slice_(self.noisy_files[index], 
                                    self.cfg.window, 
                                    self.cfg.hop_per)
        assert len(clean_chunks) == len(noisy_chunks), "UNEQUAL LENGTHS OF CLEAN AND NOISY SPEECH CHUNKS!!!"
        return clean_chunks.copy(), noisy_chunks.copy()



def get_data_loader(cfg, clean_path, noisy_path):
    return DataLoader(DATA(cfg, clean_path, noisy_path), 
                      batch_size = cfg.batch_size, 
                      shuffle = True, 
                      num_workers = cfg.num_workers, 
                      drop_last = True, 
                      pin_memory = True)


if  __name__ == '__main__':
    config = '/media/sarthak/Data/SER/code/config.yaml'
    clean = '/media/sarthak/Data/SER/data/clean_testset/'
    noisy = '/media/sarthak/Data/SER/data/noisy_testset/'
    cfg = Configuration(config)
    data = DATA(cfg, clean, noisy)
    data.__getitem__(110)
