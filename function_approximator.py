import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class FunctionApproximator():
    def __init__(self, nn_dims):
        self.nn_dims = nn_dims

    def build_network(self, lrate=0.01, opt=keras.optimizers.SGD, loss=keras.losses.categorical_crossentropy, act="relu"):
        model = keras.models.Sequential()

        for i, node_count in enumerate(self.nn_dims):
            model.add(layers.Dense(node_count, activation=act,
                                   name="Layer{}".format(i)))

        model.compile(optimizer=opt(lr=lrate), loss=loss, metrics=[
                      keras.metrics.categorical_accuracy])
        return model


if __name__ == "__main__":
    func_app = FunctionApproximator([16, 128, 128, 64, 1])
    model = func_app.build_network()
    x = tf.ones((1, 16))
    y = model(x)
    model.summary()
