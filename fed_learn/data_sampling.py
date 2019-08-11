import random

import numpy as np


def iid_data_indices(nb_clients: int, data_len: int):
    indices = np.arange(data_len)
    np.random.shuffle(indices)
    chunks = np.array_split(indices, nb_clients)
    return chunks


def non_iid_data_indices(nb_clients: int, y_true: np.ndarray, nb_shards: int = 200):
    y_true = y_true.flatten()
    data_len = len(y_true)

    # Sorting data indices by labels
    indices = np.arange(data_len)
    indices = indices[y_true.argsort()]

    shards = np.array_split(indices, nb_shards)
    random.shuffle(shards)
    shards_for_users = np.array_split(shards, nb_clients)
    indices_for_users = [np.hstack(x) for x in shards_for_users]

    return indices_for_users
