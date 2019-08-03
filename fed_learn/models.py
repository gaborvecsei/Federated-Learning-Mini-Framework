from keras import backend as K
from keras import optimizers, losses, models
from keras.applications.vgg16 import VGG16


def create_model(input_shape: tuple,
                 nb_classes: int,
                 optimizer=optimizers.Adam(lr=0.001),
                 loss=losses.categorical_crossentropy):
    model = VGG16(input_shape=input_shape,
                  classes=nb_classes,
                  weights='imagenet',
                  include_top=False)
    model.compile(optimizer, loss, metrics=["accuracy"])
    return model


def set_model_weights(model: models.Model, weight_list):
    for i, symbolic_weights in enumerate(model.weights):
        weight_values = weight_list[i]
        K.set_value(symbolic_weights, weight_values)
