from typing import List

import numpy as np


class WeightSummarizer:
    def __init__(self):
        pass

    def process(self, client_weight_list: List[List[np.ndarray]]) -> List[np.ndarray]:
        raise NotImplementedError()


class FedAvg(WeightSummarizer):
    def __init__(self):
        super().__init__()

    def process(self, client_weight_list: List[List[np.ndarray]]) -> List[np.ndarray]:
        weights_average = [np.zeros_like(w) for w in client_weight_list[0]]

        for i in range(len(weights_average)):
            w = weights_average[i]
            for k in range(len(client_weight_list)):
                w += client_weight_list[k][i]
            w /= len(client_weight_list)
        return weights_average
