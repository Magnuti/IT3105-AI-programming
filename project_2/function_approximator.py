import os  # nopep8
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # nopep8
from argument_parser import Arguments
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow import keras
import numpy as np

# tf.debugging.set_log_device_placement(True)
# tf.config.set_visible_devices([], 'GPU')
# my_devices = tf.config.list_physical_devices(device_type='CPU')
# tf.config.set_visible_devices([], 'CPU')


class ANET:
    """
    This class houses the neural network(s) used for the actor (and possibly
    the critic).
    """

    def __init__(self, neurons_per_layer, activation_functions, optimizer, learning_rate):
        optimizer = keras.optimizers.get(optimizer.value)
        if learning_rate is not None:
            optimizer.learning_rate.assign(learning_rate)

        # Build the model
        model = keras.models.Sequential()
        for i, node_count in enumerate(neurons_per_layer):
            if i == 0:
                model.add(layers.Input(shape=node_count))
            else:
                model.add(layers.Dense(
                    node_count, activation=activation_functions[i - 1].value, name="Layer_{}".format(i)))

        # We use MSE since Keras categorical_crossentropy takes in one-hot encoded targets
        model.compile(optimizer=optimizer, loss="mse", metrics=[
                      tf.keras.metrics.MeanSquaredError()])

        self.model = model
        # self.model.summary()

    def cache_model_params(self):
        self.params_per_layer = []
        for layer in self.model.layers:
            params = layer.get_weights()
            af = layer.get_config()["activation"]
            activation_function = tf.keras.activations.get(af)
            self.params_per_layer.append(
                (params[0], params[1], activation_function))

    def forward(self, features):
        # print(features.shape)
        # if len(features.shape) == 1:
        # Reshape from (k, ) to (1, k) since that means a batch size of 1
        # features = features.reshape((1, features.shape[0]))
        features = np.expand_dims(features, axis=0)
        # print(features.shape)

        # with tf.device('/CPU:0'):
        # return self.model(features)
        # keras_output = self.model(features)

        x = features
        for i, layer in enumerate(self.model.layers):
            # print("Layer", i)
            # params = layer.get_weights()
            params = self.params_per_layer[i]
            weights = params[0]
            # print("Input", x.shape, type(x))
            # print("Weight", weights.shape, type(weights))
            bias = params[1]
            # print("Bias", bias.shape, type(bias))
            # af = layer.get_config()["activation"]
            # activation_function = tf.keras.activations.get(af)
            activation_function = params[2]
            # print(activation_function)
            z = np.matmul(x, weights) + bias
            z = tf.convert_to_tensor(z)  # ? can we avoid this?
            # print(type(z))
            x = activation_function(z)

        # print(keras_output)
        # print(x)

        # exit()
        return x

    def fit(self, x, y, batch_size, epochs, verbose=0):
        return self.model.fit(
            x, y, batch_size=batch_size, epochs=epochs, verbose=verbose)

    def __get_file_path(self, episode, save_path):
        return save_path.joinpath("anet_episode_{}".format(episode))

    def save_model(self, episode, save_path):
        self.model.save(self.__get_file_path(episode, save_path))

    def load_model(self, episode, save_path):
        self.model = keras.models.load_model(
            self.__get_file_path(episode, save_path))

    def load_model_path_known(self, file_path):
        self.model = keras.models.load_model(file_path)


if __name__ == "__main__":
    args = Arguments()
    args.parse_arguments()
    anet = ANET(args.neurons_per_layer, args.activation_functions,
                args.optimizer, args.learning_rate)
    anet.model.summary()
