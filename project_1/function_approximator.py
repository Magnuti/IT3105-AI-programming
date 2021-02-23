import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers


class FunctionApproximator():
    def __init__(self, nn_dims):
        self.nn_dims = nn_dims

    def build_network(self, lrate=0.01, opt=keras.optimizers.SGD,
                      loss=keras.losses.categorical_crossentropy, act="relu"):
        model = keras.models.Sequential()

        for i, node_count in enumerate(self.nn_dims):
            if i == 0:
                model.add(layers.Input(shape=(1, node_count),
                                       name="Layer_{}".format(i)))
            elif i == len(self.nn_dims) - 1:
                # We should take activation functions as config input
                model.add(layers.Dense(
                    node_count, activation="sigmoid", name="Layer_{}".format(i)))
            else:
                model.add(layers.Dense(node_count, activation=act,
                                       name="Layer_{}".format(i)))

        model.compile(optimizer=opt(lr=lrate), loss=loss, metrics=[
                      keras.metrics.categorical_accuracy])

        self.model = model

    def decay_eligibilities(self, discount_factor, eligibility_decay):
        for i, eligibility in enumerate(self.eligibilities):
            self.eligibilities[i] = tf.multiply(
                eligibility, discount_factor * eligibility_decay)

    def reset_eligibilities(self):
        self.eligibilities = None

    def fit(self, feature, learning_rate, TD_error):
        # params are the trainable variables, not all variables.
        params = self.model.trainable_weights
        features = tf.convert_to_tensor([feature])

        with tf.GradientTape() as tape:
            # Do not move the line below up above the "with"-block, it will crash unexpectedly
            predictions = self.model(features)

            # gradient for all params <[tensorarray]>
            gradients = tape.gradient(predictions, params)

            # Initialize eligibilities to zero if non-existent
            if(self.eligibilities is None):
                self.eligibilities = []
                for i, gradient in enumerate(gradients):
                    self.eligibilities.append(
                        tf.zeros(shape=(gradients[i].get_shape()), dtype=tf.float32))

            # Update eligibilities
            for i, gradient in enumerate(gradients):
                self.eligibilities[i] = tf.add(
                    self.eligibilities[i], gradient)

            # Find new "gradients": (TD_error * e_i)
            # The learning rate is integrated in the model
            for i, gradient in enumerate(gradients):
                # Negative because the gradient descent equation is normally minus, so minus*minus=plus
                gradients[i] = - tf.multiply(self.eligibilities[i], TD_error)

            # Adjust weights with the new gradients
            self.model.optimizer.apply_gradients(zip(gradients, params))
