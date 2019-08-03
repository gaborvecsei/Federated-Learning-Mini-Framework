import numpy as np
from keras import datasets
from keras import utils

import fed_learn

model = fed_learn.create_model((32, 32, 3), 10)

(x_train, y_train), (_, _) = datasets.cifar10.load_data()
y_train = utils.to_categorical(y_train, len(np.unique(y_train)))

model.fit(x_train, y_train, batch_size=32, epochs=20, verbose=1)
