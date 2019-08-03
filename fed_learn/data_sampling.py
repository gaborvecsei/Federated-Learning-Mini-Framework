import numpy as np


def iid_data_indices(nb_clients: int, data_len: int):
    indices = np.arange(data_len)
    np.random.shuffle(indices)
    chunks = np.split(indices, nb_clients)
    return chunks


def non_iid_data_indices(nb_clients: int, data_len: int):
    raise NotImplementedError
