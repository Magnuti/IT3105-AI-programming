import tensorflow as tf
from tensorflow import keras
import pathlib
import numpy as np


class TournamentOfProgressivePolicies:
    def __init__(self):
        self.save_path = pathlib.Path("saved_models")
        self.save_path.mkdir(exist_ok=True)

    def __get_file_path(self, episode):
        return self.save_path.joinpath("anet_episode_{}".format(episode))

    def save_ANET(self, anet, episode):
        anet.save(self.__get_file_path(episode))

    def load_ANET(self, episode):
        return keras.models.load_model(self.__get_file_path(episode))


if __name__ == "__main__":
    inputs = keras.Input(shape=(32,))
    outputs = keras.layers.Dense(1)(inputs)
    model = keras.Model(inputs, outputs)
    model.compile(optimizer="adam", loss="mean_squared_error")

    test_input = np.random.random((128, 32))
    test_target = np.random.random((128, 1))
    model.fit(test_input, test_target)

    inputs = keras.Input(shape=(32,))
    outputs = keras.layers.Dense(1)(inputs)
    model50 = keras.Model(inputs, outputs)
    model50.compile(optimizer="adam", loss="mean_squared_error")

    test_input = np.random.random((128, 32))
    test_target = np.random.random((128, 1))
    model50.fit(test_input, test_target)

    topp = TournamentOfProgressivePolicies()
    topp.save_ANET(model, 0)

    topp.save_ANET(model50, 50)

    reconstructed_model = topp.load_ANET(0)

    np.testing.assert_allclose(
        model.predict(test_input), reconstructed_model.predict(test_input)
    )

    reconstructed_model.fit(test_input, test_target)

    reconstructed_model50 = topp.load_ANET(50)

    np.testing.assert_allclose(
        model50.predict(test_input), reconstructed_model50.predict(test_input)
    )

    reconstructed_model50.fit(test_input, test_target)
