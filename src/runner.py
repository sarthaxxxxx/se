import torch 
import torch.nn as nn 
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import torch.backends.cudnn as cudnn

import numpy as np
from tqdm import tqdm 

def trainer(train_loader, cfg, epoch, device, val_loader = None):
    for idx, (clean, noise) in tqdm(enumerate(train_loader)):
        pass
