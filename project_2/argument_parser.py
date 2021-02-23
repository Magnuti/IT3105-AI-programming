import yaml

from constants import ActivationFunction, Optimizer


class Arguments:
    def parse_arguments(self):
        with open("config.yaml", "r", encoding="utf-8") as f:
            config_data = yaml.safe_load(f)

        self.board_size = config_data["board_size"]
        self.episodes = config_data["episodes"]
        self.search_games_per_actual_move = config_data["search_games_per_actual_move"]
        self.simulations = config_data["simulations"]
        self.learning_rate = config_data["learning_rate"]
        self.neurons_per_layer = config_data["neurons_per_layer"]

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
