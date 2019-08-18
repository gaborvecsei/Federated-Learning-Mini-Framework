import random
from typing import List

import numpy as np
from keras import utils

from fed_learn.fed_client import Client


def iid_data_indices(nb_clients: int, labels: np.ndarray):
    labels = labels.flatten()
    data_len = len(labels)
    indices = np.arange(data_len)
    np.random.shuffle(indices)
    chunks = np.array_split(indices, nb_clients)
    return chunks


def non_iid_data_indices(nb_clients: int, labels: np.ndarray, nb_shards: int = 200):
    labels = labels.flatten()
    data_len = len(labels)

    indices = np.arange(data_len)
    indices = indices[labels.argsort()]

    shards = np.array_split(indices, nb_shards)
    random.shuffle(shards)
    shards_for_users = np.array_split(shards, nb_clients)
    indices_for_users = [np.hstack(x) for x in shards_for_users]

    return indices_for_users


class BaseDataProcessor:
    def __init__(self):
        pass

    @staticmethod
    def pre_process(x: np.ndarray, y: np.ndarray, nb_classes: int):
        raise NotImplementedError


class CifarProcessor(BaseDataProcessor):
    def __init__(self):
        super().__init__()

    @staticmethod
    def pre_process(x: np.ndarray, y: np.ndarray, nb_classes: int):
        y = utils.to_categorical(y, nb_classes)
        x = x.astype(np.float32)
        x /= 255.0
        return x, y


class DataHandler:
    def __init__(self,
                 x_train: np.ndarray,
                 y_train: np.ndarray,
                 x_test: np.ndarray,
                 y_test: np.ndarray,
                 preprocessor: BaseDataProcessor,
                 only_debugging: bool = True):
        self._nb_classes = len(np.unique(y_train))
        self._preprocessor = preprocessor

        if only_debugging:
            x_train = x_train[:100]
            y_train = y_train[:100]
            x_test = x_test[:100]
            y_test = y_test[:100]

        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test

    def _sample(self, sampling_technique: str, nb_clients: int):
        if sampling_technique.lower() == "iid":
            sampler_fn = iid_data_indices
        else:
            sampler_fn = non_iid_data_indices
        client_data_indices = sampler_fn(nb_clients, self.y_train)
        return client_data_indices

    def preprocess(self, x, y):
        x, y = self._preprocessor.pre_process(x, y, self._nb_classes)
        return x, y

    def assign_data_to_clients(self, clients: List[Client], sampling_technique: str):
        sampled_data_indices = self._sample(sampling_technique, len(clients))
        for client, data_indices in zip(clients, sampled_data_indices):
            x = self.x_train[data_indices]
            y = self.y_train[data_indices]
            x, y = self.preprocess(x, y)
            client.receive_data(x, y)
