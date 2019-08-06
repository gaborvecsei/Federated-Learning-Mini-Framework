from typing import Callable

import numpy as np
from keras import datasets, utils, models

import fed_learn
from fed_learn.weight_summarizer import WeightSummarizer


class Server:
    def __init__(self, model_fn: Callable,
                 weight_summarizer: WeightSummarizer,
                 nb_clients: int = 100,
                 client_fraction: float = 0.2,
                 only_debugging: bool = True):
        self.nb_clients = nb_clients
        self.client_fraction = client_fraction
        self.weight_summarizer = weight_summarizer

        # Initialize the global model's weights
        self.model_fn = model_fn
        model = self.model_fn()
        self.global_test_metrics_dict = {k: [] for k in model.metrics_names}
        self.global_model_weights = model.get_weights()
        fed_learn.get_rid_of_the_models(model)

        self.global_train_losses = []
        self.epoch_losses = []

        (x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()

        if only_debugging:
            # TODO: remove me
            x_train = x_train[:100]
            y_train = y_train[:100]
            x_test = x_test[:100]
            y_test = y_test[:100]

        # TODO: separate preprocessor for the data transformations
        y_train = utils.to_categorical(y_train, len(np.unique(y_train)))
        x_train = x_train.astype(np.float32)
        x_train /= 255.0

        y_test = utils.to_categorical(y_test, len(np.unique(y_test)))
        x_test = x_test.astype(np.float32)
        x_test /= 255.0

        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test

        self.client_data_indices = None
        self.clients = []
        self.client_model_weights = []

        # Training parameters used by the clients
        self.client_train_params_dict = {"batch_size": 32,
                                         "epochs": 5,
                                         "verbose": 1,
                                         "shuffle": True}

    def _generate_data_indices(self):
        self.client_data_indices = fed_learn.iid_data_indices(self.nb_clients, len(self.x_train))

    def _get_data_indices_for_client(self, client: int):
        return self.client_data_indices[client]

    def _send_train_data_to_client(self, client):
        relevant_data_point_indices = self._get_data_indices_for_client(client.id)
        x = self.x_train[relevant_data_point_indices]
        y = self.y_train[relevant_data_point_indices]
        client.receive_data(x, y)
        return x, y

    def _create_model_with_updated_weights(self) -> models.Model:
        model = self.model_fn()
        fed_learn.models.set_model_weights(model, self.global_model_weights)
        return model

    def send_train_data(self):
        self._generate_data_indices()
        for c in self.clients:
            self._send_train_data_to_client(c)

    def send_model(self, client):
        client.receive_and_init_model(self.model_fn, self.global_model_weights)

    def init_for_new_epoch(self):
        # Reset the collected weights
        self.client_model_weights.clear()
        # Reset epoch losses
        self.epoch_losses.clear()

    def receive_results(self, client):
        client_weights = client.model.get_weights()
        self.client_model_weights.append(client_weights)
        client.reset_model()

    def create_clients(self):
        # Create all the clients
        for i in range(self.nb_clients):
            client = fed_learn.Client(i)
            self.clients.append(client)

    def summarize_weights(self):
        new_weights = self.weight_summarizer.process(self.client_model_weights)
        self.global_model_weights = new_weights

    def get_client_train_param_dict(self):
        return self.client_train_params_dict

    def update_client_train_params(self, param_dict: dict):
        self.client_train_params_dict.update(param_dict)

    def test_global_model(self):
        model = self._create_model_with_updated_weights()
        results = model.evaluate(self.x_test, self.y_test, batch_size=32, verbose=1)

        results_dict = dict(zip(model.metrics_names, results))
        for metric_name, value in results_dict.items():
            self.global_test_metrics_dict[metric_name].append(value)

        fed_learn.get_rid_of_the_models(model)

        return results_dict

    def select_clients(self):
        nb_clients_to_use = max(int(self.nb_clients * self.client_fraction), 1)
        client_indices = np.arange(self.nb_clients)
        np.random.shuffle(client_indices)
        selected_client_indices = client_indices[:nb_clients_to_use]
        return np.asarray(self.clients)[selected_client_indices]

    def save_model_weights(self, path: str):
        model = self._create_model_with_updated_weights()
        model.save_weights(str(path), overwrite=True)
        fed_learn.get_rid_of_the_models(model)

    def load_model_weights(self, path: str, by_name: bool = False):
        model = self._create_model_with_updated_weights()
        model.load_weights(str(path), by_name=by_name)
        self.global_model_weights = model.get_weights()
        fed_learn.get_rid_of_the_models(model)
