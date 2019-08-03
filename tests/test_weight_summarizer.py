import fed_learn
import numpy as np
import unittest


class TestFedAvgAlgorithm(unittest.TestCase):

    def setUp(self) -> None:
        self.weight_summarizer = fed_learn.FedAvg()

        nb_clients = 3
        nb_weight_arrays = 6

        self.all_clients_weights = []

        for i in range(nb_clients):
            client_weight_arrays = []
            for k in range(nb_weight_arrays):
                rnd_weight_array = np.ones((8, 12))
                rnd_weight_array += i
                client_weight_arrays.append(rnd_weight_array)
            self.all_clients_weights.append(client_weight_arrays)

        self.avg_weights = self.weight_summarizer.process(self.all_clients_weights)

    def test_basic_averaging_mean(self):
        self.assertAlmostEqual(np.mean(self.avg_weights), 2.0)

    def test_basic_averaging_min(self):
        self.assertAlmostEqual(np.min(self.avg_weights), 2.0)

    def test_basic_averaging_max(self):
        self.assertAlmostEqual(np.max(self.avg_weights), 2.0)
