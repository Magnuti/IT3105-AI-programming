import yaml

from constants import GameType, ActivationFunction, Optimizer


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

        self.neurons_per_layer = config_data["neurons_per_hidden_layer"]
        # Add the input/output dimensions
        # One input neuron for each state we can be in (11 for a 10 piece
        # game since we can have [0-10] pieces) + the ID of the player (two bits)
        # One output neuron for each action
        # + 2 to input dimension because we need two bits to represent the current player
        if self.game_type == GameType.NIM:
            self.neurons_per_layer.insert(0, self.nim_N + 3)
            self.neurons_per_layer.append(self.nim_K)
        elif self.game_type == GameType.HEX:
            self.neurons_per_layer.insert(0, self.board_size ** 2 + 2)
            self.neurons_per_layer.append(self.board_size ** 2)
        else:
            raise NotImplementedError()

        activation_functions = config_data["activation_functions"]
        self.activation_functions = []
        for af in activation_functions:
            self.activation_functions.append(ActivationFunction(af))

        self.optimizer = Optimizer(config_data["optimizer"])
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
