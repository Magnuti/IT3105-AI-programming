import yaml

from constants import GameType, ActivationFunction, Optimizer, EpsilonDecayFunction


class Arguments:
    def parse_arguments(self):
        with open("config.yaml", "r", encoding="utf-8") as f:
            config_data = yaml.safe_load(f)

        self.game_type = GameType(config_data["game_type"])
        self.nim_N = config_data["nim_N"]
        self.nim_K = config_data["nim_K"]
        self.board_size = config_data["board_size"]
        self.episodes = config_data["episodes"]
        self.simulations = config_data["simulations"]
        self.learning_rate = config_data["learning_rate"]
        self.epsilon = config_data["epsilon"]
        self.epsilon_decay_function = EpsilonDecayFunction(
            config_data["epsilon_decay_function"])
        self.epsilon_decay = config_data["epsilon_decay"]
        self.visualize = config_data["visualize"]
        self.visualization_games = config_data["visualization_games"]

        self.neurons_per_layer = config_data["neurons_per_hidden_layer"]
        # Add the input/output dimensions
        # One output neuron for each action
        if self.game_type == GameType.NIM:
            # One input neuron for each state we can be in (11 for a 10 piece
            # game since we can have [0-10] pieces) + the ID of the player (two bits)
            self.neurons_per_layer.insert(0, self.nim_N + 3)
            self.neurons_per_layer.append(self.nim_K)

            if self.visualize:
                self.visualize = False
                print(
                    "Warning: visualize was set to false since we do not visualize the game of Nim.")
        elif self.game_type == GameType.HEX:
            # A NxN board has N^2 cells, each cell is represented as two bits,
            # we also need 2 bits to represent the current player.
            self.neurons_per_layer.insert(0, self.board_size ** 2 * 2 + 2)
            self.neurons_per_layer.append(self.board_size ** 2)
        else:
            raise NotImplementedError()

        activation_functions = config_data["activation_functions"]
        self.activation_functions = []
        for af in activation_functions:
            self.activation_functions.append(ActivationFunction(af))

        self.optimizer = Optimizer(config_data["optimizer"])
        self.replay_buffer_selection_size = config_data["replay_buffer_selection_size"]
        self.mini_batch_size = config_data["mini_batch_size"]
        self.epochs = config_data["epochs"]
        self.games_to_save = config_data["games_to_save"]
        self.games_between_agents = config_data["games_between_agents"]

    def __str__(self):
        x = "Arguments: {\n"
        for key, value in self.__dict__.items():
            x += "\t{}: {}\n".format(key, value)
        x += "}"
        return x


if __name__ == "__main__":
    arguments = Arguments()
    arguments.parse_arguments()
    print(arguments)
