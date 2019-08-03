from keras import backend as K
import gc


def get_rid_of_the_models(model=None):
    # TODO: somehow this does not free up the GPU memory (after a while you will get OOM) (tested on Windows 10...)
    K.clear_session()
    if model is not None:
        del model
    gc.collect()
