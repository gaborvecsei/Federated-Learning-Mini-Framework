from typing import Callable

from keras import datasets

import fed_learn


class Server:
    def __init__(self, model_fn: Callable, nb_clients: int, weight_summarizer: fed_learn.WeightSummarizer):
        self.nb_clients = nb_clients
        self.weight_summarizer = weight_summarizer

        self.model_fn = model_fn
        model = self.model_fn()
        self.model_weights = model.get_weights()
        fed_learn.get_rid_of_the_models(model)

        (x_train, y_train), (_, _) = datasets.cifar10.load_data()

        self.x_train = x_train
        self.y_train = y_train

        self.client_data_indices = None
        self.clients = []

        self.client_model_weights = []

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
        client.receive_and_init_model(self.model_fn, self.model_weights)

    def init_for_new_epoch(self):
        self._generate_data_indices()

    def receive_results(self, client):
        client_weights = client.model.get_weights()
        self.client_model_weights.append(client_weights)
        client.reset_model()

    def create_clients(self):
        for i in range(self.nb_clients):
            client = fed_learn.Client(i)
            self.clients.append(client)

    def summarize_weights(self):
        new_weights = self.weight_summarizer.process(self.client_model_weights)
        self.model_weights = new_weights

    @staticmethod
    def get_client_train_param_dict():
        train_dict = {"batch_size": 32,
                      "epochs": 5,
                      "verbose": 1,
                      "shuffle": True}
        return train_dict
