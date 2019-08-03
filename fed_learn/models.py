from keras import backend as K
from keras import optimizers, losses, models, layers
from keras.applications.vgg16 import VGG16


def create_model(input_shape: tuple, nb_classes: int, init_with_imagenet: bool = False):
    weights = None
    if init_with_imagenet:
        weights = "imagenet"

    model = VGG16(input_shape=input_shape,
                  classes=nb_classes,
                  weights=weights,
                  include_top=False)
    # "Shallow" VGG for Cifar10
    x = model.get_layer('block3_pool').output
    x = layers.Flatten(name='Flatten')(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dense(nb_classes)(x)
    x = layers.Softmax()(x)
    model = models.Model(model.input, x)

    loss = losses.categorical_crossentropy
    optimizer = optimizers.Adam(lr=0.001)

    model.compile(optimizer, loss, metrics=["accuracy"])
    return model


def set_model_weights(model: models.Model, weight_list):
    for i, symbolic_weights in enumerate(model.weights):
        weight_values = weight_list[i]
        K.set_value(symbolic_weights, weight_values)
