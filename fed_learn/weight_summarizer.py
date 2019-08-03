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
        # TODO: implement simple averaging
        return client_weight_list[0]
