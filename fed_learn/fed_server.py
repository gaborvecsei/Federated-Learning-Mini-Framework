from typing import Callable

import numpy as np
from keras import datasets, utils

import fed_learn
from fed_learn.weight_summarizer import WeightSummarizer


class Server:
    def __init__(self, model_fn: Callable,
                 nb_clients: int,
                 weight_summarizer: WeightSummarizer,
                 only_debugging: bool = True):
        self.nb_clients = nb_clients
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

    def send_train_data(self, client):
        relevant_data_point_indices = self._get_data_indices_for_client(client.id)
        x = self.x_train[relevant_data_point_indices]
        y = self.y_train[relevant_data_point_indices]
        client.receive_data(x, y)
        return x, y

    def send_model(self, client):
        client.receive_and_init_model(self.model_fn, self.global_model_weights)

    def init_for_new_epoch(self):
        # Reset clients
        self.clients.clear()
        # Reset the collected weights
        self.client_model_weights.clear()
        # Reset epoch losses
        self.epoch_losses.clear()
        # Generate new data indices for the clients
        self._generate_data_indices()

    def receive_results(self, client):
        client_weights = client.model.get_weights()
        self.client_model_weights.append(client_weights)
        client.reset_model()

    def create_clients(self):
        # Create new ones
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
        model = self.model_fn()
        fed_learn.models.set_model_weights(model, self.global_model_weights)
        results = model.evaluate(self.x_test, self.y_test, batch_size=32, verbose=1)

        results_dict = dict(zip(model.metrics_names, results))
        for metric_name, value in results_dict.items():
            self.global_test_metrics_dict[metric_name].append(value)

        fed_learn.get_rid_of_the_models(model)

        return results_dict
