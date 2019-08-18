from typing import Callable

import numpy as np
from keras import models

import fed_learn
from fed_learn.weight_summarizer import WeightSummarizer


class Server:
    def __init__(self, model_fn: Callable,
                 weight_summarizer: WeightSummarizer,
                 nb_clients: int = 100,
                 client_fraction: float = 0.2):
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

        self.clients = []
        self.client_model_weights = []

        # Training parameters used by the clients
        self.client_train_params_dict = {"batch_size": 32,
                                         "epochs": 5,
                                         "verbose": 1,
                                         "shuffle": True}

    def _create_model_with_updated_weights(self) -> models.Model:
        model = self.model_fn()
        fed_learn.models.set_model_weights(model, self.global_model_weights)
        return model

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

    def test_global_model(self, x_test: np.ndarray, y_test: np.ndarray):
        model = self._create_model_with_updated_weights()
        results = model.evaluate(x_test, y_test, batch_size=32, verbose=1)

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
