import torch 
import torch.nn.utils.rnn as rnn
    
def custom_collate(batch):
    """ Module to perform batching for variable-size inputs for PyTorch DataLoader.

    Parameters
    ----------
    batch : list
        List of tuples containing clean and noisy speech chunks.

    Returns
    -------
    pad_clean : torch.Tensor
        Tensor containing clean speech chunks (batch wise)
    pad_noise : torch.Tensor
        Tensor containing noisy speech chunks (batch wise)
    """

    data_1, data_2 = zip(*batch)
    clean_batch, noisy_batch = [torch.tensor(x) for x in data_1], \
                               [torch.tensor(x) for x in data_2]
    pad_clean, pad_noisy = rnn.pad_sequence(clean_batch, batch_first = True, padding_value = 0.), \
                           rnn.pad_sequence(noisy_batch, batch_first = True, padding_value = 0.)
    return pad_clean.permute(0, 2, 1, 3), pad_noisy.permute(0, 2, 1, 3)
