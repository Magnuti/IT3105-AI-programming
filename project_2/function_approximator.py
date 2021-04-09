from tensorflow import keras
from tensorflow.keras import layers

from argument_parser import Arguments


class ANET:
    """
    This class houses the neural network(s) used for the actor (and possibly
    the critic).
    """

    def __init__(self, neurons_per_layer, activation_functions):
        self.neurons_per_layer = neurons_per_layer
        self.activation_functions = activation_functions

        self.build_network()

    def build_network(self, lrate=0.01, opt=keras.optimizers.SGD,
                      loss=keras.losses.categorical_crossentropy):
        # TODO take loss as config param
        model = keras.models.Sequential()

        for i, node_count in enumerate(self.neurons_per_layer):
            if i == 0:
                model.add(layers.Input(shape=node_count))
            else:
                model.add(layers.Dense(
                    node_count, activation=self.activation_functions[i - 1].value, name="Layer_{}".format(i)))

        model.compile(optimizer=opt(lr=lrate), loss=loss, metrics=[
                      keras.metrics.categorical_accuracy])

        self.model = model

    def forward(self, features):
        if len(features.shape) == 1:
            # Reshape from (k, ) to (1, k) since that means a batch size of 1
            features = features.reshape((1, features.shape[0]))
        return self.model(features)

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
    anet = ANET(args.neurons_per_layer, args.activation_functions)
    anet.model.summary()
