from numpy.core.numeric import zeros_like
import torch 
import torch.nn.utils.rnn as rnn
    
def train_collate_fn(batch):
    """ Module to perform batching for variable-size inputs for PyTorch DataLoader.

    Parameters
    ----------
    batch : list
        List of tuples containing clean and noisy speech chunks (torch.tensor)  .

    Returns
    -------
    clean_final: torch.Tensor
        Tensor containing clean speech chunks (batches, frames, time_steps, channels)
    noisy_final : torch.Tensor
        Tensor containing noisy speech chunks (batches, frames, time_steps, channels)
    """
    data_1, data_2 = zip(*batch)
    maxlen = max([x.size(0) for x in data_1])
    clean_batch, noisy_batch = [torch.cat((x, torch.zeros(maxlen - x.size(0), x.size(1), x.size(2)))) for x in data_1], \
                               [torch.cat((x, torch.zeros(maxlen - x.size(0), x.size(1), x.size(2)))) for x in data_2]
    clean_final, noisy_final = torch.stack(clean_batch, dim = 0), torch.stack(noisy_batch, dim = 0)
    return clean_final, noisy_final


def test_collate_fn(batch):
    """ Module to perform batching for test data.

    Parameters
    ----------
    batch : list
        List of tuples containing clean and noisy speech chunks.
    
    Returns
    -------
    batch : torch.Tensor
        Tensor containing clean and noisy speech chunks
    """
    return torch.tensor(batch)
