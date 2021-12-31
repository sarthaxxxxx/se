import os
import sys
import torch
from torch.utils.data import Dataset, DataLoader

sys.path.append('/media/sarthak/Data/SER/code/')

from utils.signal_preprocess import read_slice
from utils.custom_collate import train_collate_fn, test_collate_fn
from utils.cfg_loader import Configuration


class VALENTINIDataset(Dataset):
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
        clean_data += read_slice(self.clean_files[index], 
                                 self.cfg.window, 
                                 self.cfg.tt_max)
        noisy_data += read_slice(self.noisy_files[index], 
                                 self.cfg.window, 
                                 self.cfg.tt_max)
        assert len(clean_data) == len(noisy_data), "UNEQUAL LENGTHS OF CLEAN AND NOISY SPEECH CHUNKS!!!"
        return torch.tensor(clean_data), torch.tensor(noisy_data)



def get_data_loader(cfg, clean_path, noisy_path):
    """ Module to create data loader using a custom collate function.

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
    """
    dataset = VALENTINIDataset(cfg, clean_path, noisy_path)  
    dataset.__getitem__(0)  
    if cfg.train:  return DataLoader(dataset, 
                                     batch_size = cfg.batch_size, 
                                     shuffle = True, 
                                     collate_fn = train_collate_fn,
                                     num_workers = cfg.num_workers,
                                     drop_last = True,
                                     pin_memory= True,
                                     sampler = None)
    return DataLoader(dataset,
                      batch_size = 1,
                      shuffle = False,
                      num_workers = 0,
                      drop_last = False,
                      collate_fn = test_collate_fn)


if __name__ == '__main__':
    config = '/media/sarthak/Data/SER/code/config.yaml'
    clean_path = '/media/sarthak/Data/SER/data/clean_trainset/'
    noisy_path = '/media/sarthak/Data/SER/data/noisy_trainset/'
    cfg = Configuration(config)
    train_dl = get_data_loader(cfg, clean_path, noisy_path)
