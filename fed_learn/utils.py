import gc
import os
from typing import List

from keras import backend as K

import fed_learn


def get_rid_of_the_models(model=None):
    """
    This function clears the TF session from the model.
    This is needed as TF/Keras models are not automatically cleared, and the memory will be overloaded
    """

    K.clear_session()
    if model is not None:
        del model
    gc.collect()


def print_selected_clients(clients: List[fed_learn.fed_client.Client]):
    client_ids = [c.id for c in clients]
    print("Selected clients for epoch: {0}".format("| ".join(map(str, client_ids))))


def set_working_GPU(gpu_ids: str):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids
