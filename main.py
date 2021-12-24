import os
import time
import random
import argparse
import numpy as np
import matplotlib.pyplot as plt

import torch 
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import torch.backends.cudnn as cudnn

from utils import *
from src import *
from models.GAN_SE import DISC_CONV, GAN_SE, GENERATOR_AE


parser = argparse.ArgumentParser(description = 'CLI args for running scripts.')
parser.add_argument('--config', type = str, 
                    default = '/media/sarthak/Data/SER/code/config.yaml')
parser.add_argument('--train_clean', 
                    default = '/media/sarthak/Data/SER/data/clean_trainset/', type = str)
parser.add_argument('--train_noisy', 
                    default = '/media/sarthak/Data/SER/data/noisy_trainset/', type = str)
args = parser.parse_args()


def main(config, gpu, clean, noisy):
    assert isinstance(config, object), 'INVALID CONFIGURATION FILE!!!'
    cfg = Configuration(config)

    device = torch.cuda("cuda" if torch.cuda.is_available() else 'cpu')
    print(f"CUDA DEVICE: {device} IS BEING USED FOR TRAINING!!")

    if cfg.seed is not None:
        torch.manual_seed(cfg.seed)
        cudnn.deterministic = True
        cudnn.benchmark = False
        np.random.seed(cfg.seed)
        random.seed(cfg.seed)

    if not os.path.exists(cfg.gen_savepath):
        os.makedirs(cfg.gen_savepath)

    if not os.path.exists(cfg.ckpt):
        os.makedirs(cfg.ckpt)

    print("{} LOADING TRAINING DATA {}".format('-' * 10, '-' * 10))
    start_time = time.time()
    train_loader = get_data_loader(cfg, clean, noisy)
    assert len(train_loader) > 0 , "NO TRAINING DATA FOUND!!!"
    print("TOTAL PRE-PROCESSING TIME : {} s.".format(time.time() - start_time))

    if cfg.model == 'GAN_SE':
        print(f"INIT {cfg.model} MODEL!!")
        gen = GENERATOR_AE(cfg, device)
        disc = DISC_CONV(cfg, device)

        g_optim, d_optim = create_optimizer(cfg, gen, disc)
            
        for epoch in range(cfg.epochs):
            print("{} EPOCH {}/{} {}".format('-' * 10, epoch, cfg.epochs, '-' * 10))
            start_time = time.time()
            trainer(train_loader, cfg, epoch, device)
    else:
        raise ValueError("INVALID MODEL NAME!!")



if __name__ == '__main__':
    main(args.config, args.train_clean, args.train_noisy)