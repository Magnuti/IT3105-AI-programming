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
        nn_dim = config_data["nn_dim"]
        learning_rate_critic = config_data["learning_rate_critic"]
        learning_rate_actor = config_data["learning_rate_actor"]
        eligibility_decay_critic = config_data["eligibility_decay_critic"]
        eligibility_decay_actor = config_data["eligibility_decay_actor"]
        discount_factor_critic = config_data["discount_factor_critic"]
        discount_factor_actor = config_data["discount_factor_actor"]
        epsilon = config_data["epsilon"]
        visualize = config_data["visualize"]
        visualize_training_episodes = config_data["visualize_training_episodes"]
        frame_time = config_data["frame_time"]

        if(board_type == BoardType.Triangle.value):
            pass
        elif(board_type == BoardType.Diamond.value):
            if(board_size == 4 and len(open_cell_positions) == 1 and ([1, 1] in open_cell_positions or [2, 2] in open_cell_positions)):
                print(
                    "WARNING: A diamond board of size 4 can only be solved with center positions [1, 2] or [2, 1], not [1, 1] or [2, 2].")
        else:
            raise NotImplementedError()

        if(critic_type == CriticType.NEURAL_NETWORK and len(nn_dim) < 2):
            raise ValueError(
                "nn_dim must have a valid neural network structure.")

        if (board_type == BoardType.Triangle.value):
            pass
            # TODO: implement bounds check for triangle
            # raise NotImplementedError()
        elif (board_type == BoardType.Diamond.value):
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
        self.nn_dim = nn_dim
        self.learning_rate_critic = learning_rate_critic
        self.learning_rate_actor = learning_rate_actor
        self.eligibility_decay_critic = eligibility_decay_critic
        self.eligibility_decay_actor = eligibility_decay_actor
        self.discount_factor_critic = discount_factor_critic
        self.discount_factor_actor = discount_factor_actor
        self.epsilon = epsilon
        self.visualize = visualize
        self.visualize_training_episodes = visualize_training_episodes
        self.frame_time = frame_time

        indexes = []
        # Calculate 1D indexes from 2D [x, y] positions
        if(self.board_type == BoardType.Triangle):
            for (y, x) in open_cell_positions:
                cells_before_this_cell = 0
                for i in range(y + 1):
                    if i < y:
                        cells_before_this_cell += i + 1
                    else:
                        cells_before_this_cell += x + 1
                indexes.append(cells_before_this_cell - 1)
        elif(self.board_type == BoardType.Diamond):
            for (y, x) in open_cell_positions:
                indexes.append(y * self.board_size + x)
        else:
            raise NotImplementedError()

        self.open_cell_positions = indexes

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
