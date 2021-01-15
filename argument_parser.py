import argparse


class Arguments():

    # For example
    # python .\argument_parser.py --board triangle --size 3 --cell_positions 2 3 4 --episodes 10 --critic table

    def parse_arguments(self):
        # Triangle/diamond
        # Board size
        # Open cells position, can be multiple
        # Number of episodes
        # Table or NN for critic
        # Dimension of critic's NN, number of layers and neurons in each layer (e.g. (5, 10, 1) is 3 layers with 5, 10 and 1 neurons)
        # Learning rate for actor
        # Learning rate for critic
        # Eligibility decay for actor
        # Eligibility decay for critic
        # Discount factor for actor
        # Discount factor for critic
        # Initial value of epsilon
        # Display active/deactive
        # Delay between frames for visualization

        self.episodes = 1
        self.learning_rate_critic = 1.0
        self.learning_rate_actor = 1.0
        self.eligibility_decay_critic = 1.0
        self.eligibility_decay_actor = 1.0
        self.discound_factor_critic = 1.0
        self.discound_factor_actor = 1.0
        self.epsilon = 1.0
        self.visualization = False
        self.frame_time = 1.0

        parser = argparse.ArgumentParser()
        parser.add_argument("-b", "--board", help="triangle or diamond board", type=str,
                            choices=["triangle", "diamond"], required=True)
        parser.add_argument("--size", help="Board size",
                            type=int, required=True)
        parser.add_argument(
            "--cell_positions", help="List of open cell positions on the form 2 4 5 ...", type=int, required=True, nargs="*")
        parser.add_argument(
            "--episodes", help="Number of episodes", type=int, required=True)
        parser.add_argument("--critic", help="Type of critic",
                            choices=["table", "nn"], required=True)
        parser.add_argument(
            "--nn_dim", help="List of dimension of the critic's NN on the form 10 15 1", type=int, nargs="*")
        parser.add_argument(
            "--lr_critic", help="Learning rate critic", type=float, default=self.learning_rate_critic)
        parser.add_argument(
            "--lr_actor", help="Learning rate actor", type=float, default=self.learning_rate_actor)
        parser.add_argument(
            "--ed_critic", help="Eligibility decay critic", type=float, default=self.eligibility_decay_critic)
        parser.add_argument(
            "--ed_actor", help="Eligibility decay actor", type=float, default=self.eligibility_decay_actor)
        parser.add_argument("--epsilon", help="Epsilon value",
                            type=float, default=self.epsilon)

        args = parser.parse_args()

        if(args.critic == "nn" and not args.nn_dim):
            parser.error("--nn_dim is required when --critic = nn")

        self.board = args.board
        self.board_size = args.size
        self.cell_positions = args.cell_positions
        self.episodes = args.episodes
        self.critic_type = args.critic
        self.nn_dim = args.nn_dim
        self.learning_rate_critic = args.lr_critic
        self.learning_rate_actor = args.lr_actor
        self.eligibility_decay_critic = args.ed_critic
        self.eligibility_decay_actor = args.ed_actor
        self.epsilon = args.epsilon

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
