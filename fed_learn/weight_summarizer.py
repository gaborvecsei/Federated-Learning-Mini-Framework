from typing import List, Optional

import numpy as np


class WeightSummarizer:
    def __init__(self):
        pass

    def process(self,
                client_weight_list: List[List[np.ndarray]],
                global_weights: Optional[List[np.ndarray]] = None) -> List[np.ndarray]:
        raise NotImplementedError()


class FedAvg(WeightSummarizer):
    def __init__(self, nu: float = 1.0):
        """
        Federated Averaging

        :param nu: Controls the summarized client join model fraction to the global model
        """

        super().__init__()
        self.nu = nu

    def process(self,
                client_weight_list: List[List[np.ndarray]],
                global_weights_per_layer: Optional[List[np.ndarray]] = None) -> List[np.ndarray]:
        nb_clients = len(client_weight_list)
        weights_average = [np.zeros_like(w) for w in client_weight_list[0]]

        for layer_index in range(len(weights_average)):
            w = weights_average[layer_index]
            if global_weights_per_layer is not None:
                global_weight_mtx = global_weights_per_layer[layer_index]
            else:
                global_weight_mtx = np.zeros_like(w)
            for client_weight_index in range(nb_clients):
                client_weight_mtx = client_weight_list[client_weight_index][layer_index]

                # TODO: this step should be done at client side (client should send the difference of the weights)
                client_weight_diff_mtx = client_weight_mtx - global_weight_mtx

                w += client_weight_diff_mtx
            weights_average[layer_index] = (self.nu * w / nb_clients) + global_weight_mtx
        return weights_average


class BitSummarizer(WeightSummarizer):
    def __init__(self):
        super().__init__()

    def process(self,
                client_weight_list: List[List[np.ndarray]],
                global_weights: Optional[List[np.ndarray]] = None) -> List[np.ndarray]:
        nb_clients = len(client_weight_list)
        weights_average = [np.zeros_like(w) for w in client_weight_list[0]]

        for layer_index in range(len(weights_average)):
            w = weights_average[layer_index]
            if global_weights is not None:
                global_weight_mtx = global_weights[layer_index]
            else:
                global_weight_mtx = np.zeros_like(w)
            for client_weight_index in range(nb_clients):
                client_weight_mtx = client_weight_list[client_weight_index][layer_index]

                client_weight_diff_mtx = client_weight_mtx - global_weight_mtx
                client_1_bit_mtx = np.zeros(client_weight_diff_mtx.shape, dtype=np.int8)
                client_1_bit_mtx[client_weight_diff_mtx > 0] = 1
                client_1_bit_mtx[client_weight_diff_mtx < 0] = -1
                client_update_mtx = np.random.uniform(0.0, 1.0, client_1_bit_mtx.shape) * client_1_bit_mtx

                w += client_update_mtx
            weights_average[layer_index] = (w / nb_clients) + global_weight_mtx
        return weights_average


class TwoBitSummarizer(WeightSummarizer):
    """
    The two bits are used per layer in the following way:
    - 0b00 --> no significant change
    - 0b01 --> values decreased
    - 0b10 --> values increased
    """

    def __init__(self, epsilon: float = 1e-3):
        super().__init__()
        self.epsilon = epsilon

    def process(self,
                client_weight_list: List[List[np.ndarray]],
                global_weights: Optional[List[np.ndarray]] = None) -> List[np.ndarray]:
        nb_clients = len(client_weight_list)
        weights_average = [np.zeros_like(w) for w in client_weight_list[0]]

        for layer_index in range(len(weights_average)):
            w = weights_average[layer_index]
            global_weight_mtx = global_weights[layer_index]
            for client_weight_index in range(nb_clients):
                client_weight_mtx = client_weight_list[client_weight_index][layer_index]
                client_weight_diff_mtx = client_weight_mtx - global_weight_mtx

                change_avg = np.mean(client_weight_diff_mtx)
                # TODO: make use of this
                change_std = np.std(client_weight_diff_mtx)

                print(change_avg)

                if (change_avg <= self.epsilon) and (change_avg >= -1 * self.epsilon):
                    # Stationary
                    client_update_mtx = np.zeros_like(client_weight_mtx)
                else:
                    client_update_mtx = np.random.uniform(0.0, 1.0, client_weight_mtx.shape)
                    if change_avg >= 0:
                        # Increased
                        pass
                    else:
                        # Decreased
                        client_update_mtx *= -1
                        pass

                w += client_update_mtx
            weights_average[layer_index] = (w / nb_clients) + global_weight_mtx
        return weights_average
