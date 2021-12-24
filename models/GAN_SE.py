import torch.nn as nn 
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np


def weights_init_normal(m):
    '''Takes in a module and initializes all linear layers with weight values taken from a Gaussian distribution.'''
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        n = m.in_features
        m.weight.data.normal_(0.0, 1.0 / np.sqrt(n))
        m.bias.data.fill_(0)


class GENERATOR_AE(nn.Module):
    def __init__(self, cfg, device):
        super(GENERATOR_AE, self).__init__()



class DISC_CONV(nn.Module):
    def  __init__(self, cfg, device):
        super(DISC_CONV, self).__init__()

