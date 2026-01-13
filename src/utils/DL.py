import torch
import numpy as np


def Data_iter(batch_size, features, labels):
    num_samples = len(features)
    indices = list(range(num_samples))
    np.random.shuffle(indices)

    for i in range(0, num_samples, batch_size):
        batch_indices = torch.tensor(
            indices[i: min(i+batch_size, num_samples)]
        )
        yield features[batch_indices], labels[batch_indices]

def try_gpu(i=0):
    if torch.cuda.device_count() >= i+1:
        return torch.device(f'cuda:{i}')
    else:
        return torch.device('cpu')
    