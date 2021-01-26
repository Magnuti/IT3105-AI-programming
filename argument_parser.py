import yaml

from constants import BoardType, CriticType


class Arguments:
    def parse_arguments(self):
        with open("config.yaml", "r", encoding="utf-8") as f:
            config_data = yaml.safe_load(f)

        board_type = config_data["board_type"]
        board_size = config_data["board_size"]
        open_cell_positions = config_data["open_cell_positions"]
        episodes = config_data["episodes"]
        critic_type = config_data["critic_type"]
        nn_dims = config_data["nn_dims"]
        learning_rate_critic = config_data["learning_rate_critic"]
        learning_rate_actor = config_data["learning_rate_actor"]
        eligibility_decay_critic = config_data["eligibility_decay_critic"]
        eligibility_decay_actor = config_data["eligibility_decay_actor"]
        discount_factor_critic = config_data["discount_factor_critic"]
        discount_factor_actor = config_data["discount_factor_actor"]
        epsilon = config_data["epsilon"]
        visualize = config_data["visualize"]
        frame_time = config_data["frame_time"]

        if(board_type == BoardType.Triangle.value):
            raise NotImplementedError()
        elif(board_type == BoardType.Diamond.value):
            if(board_size == 4 and len(open_cell_positions) == 1 and (open_cell_positions[0] == [1, 1] or open_cell_positions[0] == [2, 2])):
                print(
                    "WARNING: A diamond board of size 4 can only be solved with center positions [1, 2] or [2, 1], not [1, 1] or [2, 2].")
        else:
            raise NotImplementedError()

        if(critic_type == CriticType.NEURAL_NETWORK and len(nn_dims) < 2):
            raise ValueError(
                "nn_dims must have a valid neural network structure.")

        if(board_type == BoardType.Triangle.value):
            raise NotImplementedError()
        if(board_type == BoardType.Diamond.value):
            for (x, y) in open_cell_positions:
                if(x >= board_size or y >= board_size):
                    raise ValueError(
                        "Cell position {0} is too large for a board of size {1}x{1}".format([x, y], board_size))
        else:
            raise NotImplementedError()

        if(discount_factor_critic * eligibility_decay_critic >= 1.0):
            raise ValueError(
                "γλ > 1 is not allowed for critic (i.e. discount factor * eligibility decay must be below 1)")

        if(discount_factor_actor * eligibility_decay_actor >= 1.0):
            raise ValueError(
                "γλ > 1 is not allowed for actor (i.e. discount factor * eligibility decay must be below 1)")

        self.board_type = BoardType(board_type)
        self.board_size = board_size
        self.episodes = episodes
        self.critic_type = CriticType(critic_type)
        self.nn_dims = nn_dims
        self.learning_rate_critic = learning_rate_critic
        self.learning_rate_actor = learning_rate_actor
        self.eligibility_decay_critic = eligibility_decay_critic
        self.eligibility_decay_actor = eligibility_decay_actor
        self.discount_factor_critic = discount_factor_critic
        self.discount_factor_actor = discount_factor_actor
        self.epsilon = epsilon
        self.visualize = visualize
        self.frame_time = frame_time

        # Calculate 1D indexes from 2D [x, y] positions
        if(self.board_type == BoardType.Triangle):
            raise NotImplementedError()
        if(self.board_type == BoardType.Diamond):
            indexes = []
            for (y, x) in open_cell_positions:
                indexes.append(y * self.board_size + x)

            self.open_cell_positions = indexes
        else:
            raise NotImplementedError()

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
