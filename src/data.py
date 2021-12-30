import os
import sys
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import train_test_split

sys.path.append('/media/sarthak/Data/SER/code/')

from utils.signal_preprocess import _read_slice_
from utils.custom_collate import custom_collate


class DATA(Dataset):
    """ Dataset class to load data from clean and noisy speech files and perform pre-processing steps.

    Parameters
    ----------
    cfg : main.Configuration 
        Configuration object containing all the parameters.
    clean_path : str
        Path to the clean speech files.
    noisy_path : str
        Path to the noisy speech files.
    
    Returns
    -------
    clean_data : list
        List of clean speech segments.
    noisy_data : list
        List of noisy speech segments.
    """
    def __init__(self, cfg, clean_path, noisy_path):
        self.cfg = cfg
        self.clean_path, self.noisy_path = clean_path, noisy_path
        self.clean_files = [os.path.join(self.clean_path, file_path) for file_path in sorted(os.listdir(self.clean_path))]
        self.noisy_files = [os.path.join(self.noisy_path, file_path) for file_path in sorted(os.listdir(self.noisy_path))]

    def __len__(self):
        return len(self.clean_files)

    def __getitem__(self, index):
        print("PROCESSING WAV FILE {}/{} : {}".format(index, len(self.clean_files), self.clean_files[index]))
        clean_data, noisy_data = [], []
        clean_data += _read_slice_(self.clean_files[index], 
                                   self.cfg.window, 
                                   self.cfg.tt_max)
        noisy_data += _read_slice_(self.noisy_files[index], 
                                   self.cfg.window, 
                                   self.cfg.tt_max)
        assert len(clean_data) == len(noisy_data), "UNEQUAL LENGTHS OF CLEAN AND NOISY SPEECH CHUNKS!!!"
        return clean_data, noisy_data



def get_data_loader(cfg, clean_path, noisy_path):
    """ Module to create data loader using a custom collate function. Train and validation data is created based on user config w/ a 70-30 split.

    Parameters
    ----------
    cfg : main.Configuration
        Configuration object containing all the parameters.
    clean_path : str
        Path to the clean speech files.
    noisy_path : str
        Path to the noisy speech files.
    
    Returns
    -------
    train_dl : torch.utils.data.DataLoader
        Data loader for training data.
    val_dl : torch.utils.data.DataLoader
        Data loader for validation data.
    """
    dataset = DATA(cfg, clean_path, noisy_path)    
    if cfg.val:
        train_idx, val_idx = train_test_split(list(range(len(dataset))), 
                                              test_size = 0.3, 
                                              random_state = cfg.seed, 
                                              shuffle = False)
        data = {}
        data['train'] = Subset(dataset, train_idx)
        data['val'] = Subset(dataset, val_idx)
        dataloaders = {x : DataLoader(data[x], 
                                      batch_size = cfg.batch_size, 
                                      shuffle = True, 
                                      num_workers = 1,
                                      drop_last = True,
                                      pin_memory = True,
                                      collate_fn = custom_collate) 
                                      for x in ['train', 'val']}
        
        return dataloaders['train'], dataloaders['val']
    else: return DataLoader(dataset,
                            batch_size = cfg.batch_size,
                            shuffle = True,
                            num_workers = 1,
                            drop_last = True,
                            pin_memory = True,
                            collate_fn = custom_collate)


