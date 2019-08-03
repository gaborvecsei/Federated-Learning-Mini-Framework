from keras import backend as K


def get_rid_of_the_models(model=None):
    if model is not None:
        del model
    K.clear_session()
