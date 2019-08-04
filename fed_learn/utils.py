import gc
from typing import List

from keras import backend as K

import fed_learn


def get_rid_of_the_models(model=None):
    K.clear_session()
    if model is not None:
        del model
    gc.collect()


def print_selected_clients(clients: List[fed_learn.fed_client.Client]):
    client_ids = [c.id for c in clients]
    print("Selected clients for epoch: {0}".format("| ".join(map(str, client_ids))))
