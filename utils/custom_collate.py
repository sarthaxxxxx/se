import torch 
import numpy as np


def custom_collate(data):
    """ Module to perform batching for variable-size inputs for PyTorch DataLoader.

    Parameters
    ----------
    data : list
        List of tuples containing clean and noisy speech chunks.
    
    Returns
    -------
    features_1 : torch.Tensor
        Tensor containing clean speech chunks (batch wise)
    features_2 : torch.Tensor
        Tensor containing noisy speech chunks (batch wise)
    """
    data_1, data_2 = zip(*data) # clean and noise batches respectively (unequal sizes)

    if type(data_1[0]).__name__ == 'ndarray' and type(data_2[0]).__name__ == 'ndarray':
        data_1 = torch.from_numpy(data_1)
        data_2 = torch.from_numpy(data_2)

    if isinstance(data_1[0], torch.Tensor) and isinstance(data_2[0], torch.Tensor):
        max_len_data = max([x.shape[1] for x in data_1]) # same max value for both clean and noise
        features_1 = torch.zeros((len(data_1), data_1[0].size(0), max_len_data))
        features_2 = torch.zeros((len(data_2), data_2[0].size(0), max_len_data))
        for idx in range(len(data_1)):
            features_1[idx] = torch.cat([data_1[idx], torch.zeros((data_1[idx].size(0), max_len_data - data_1[idx].size(1)))], dim = 1)
            features_2[idx] = torch.cat([data_2[idx], torch.zeros((data_2[idx].size(0), max_len_data - data_2[idx].size(1)))], dim = 1)
        return features_1, features_2

    else: 
        return ValueError("INPUT DATA TYPE NOT UNDERSTOOD!")
    
    