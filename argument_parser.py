import argparse

from constants import BoardType, CriticType


class Arguments:

    # For example
    # python .\argument_parser.py --board diamond --size 3 --cell_positions 2 3 4 --episodes 10 --critic table

    def parse_arguments(self):
        # Triangle/diamond
        # Board size (4-8 for triangle, 3-6 for diamond)
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
        self.learning_rate_critic = 0.1
        self.learning_rate_actor = 0.1
        self.eligibility_decay_critic = 0.9
        self.eligibility_decay_actor = 0.9
        self.discount_factor_critic = 0.9
        self.discount_factor_actor = 0.9
        self.epsilon = 0.5
        self.frame_time = 1.0

        parser = argparse.ArgumentParser()
        parser.add_argument("-b", "--board", help="triangle or diamond board", type=str,
                            choices=[e.value for e in BoardType], required=True)
        parser.add_argument("--size", help="Board size",
                            type=int, required=True)
        parser.add_argument(
            "--cell_positions", help="List of open cell positions on the form 2 4 5 ...", type=int, required=True, nargs="*")
        parser.add_argument(
            "--episodes", help="Number of episodes", type=int, required=True)
        parser.add_argument("--critic", help="Type of critic",
                            choices=[e.value for e in CriticType], required=True)
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
        parser.add_argument("--df_critic", help="Discount factor critic",
                            type=float, default=self.discount_factor_critic)
        parser.add_argument("--df_actor", help="Discount factor actor",
                            type=float, default=self.discount_factor_actor)
        parser.add_argument("--epsilon", help="Epsilon value",
                            type=float, default=self.epsilon)
        parser.add_argument(
            "--visualize", help="Set the --visualization flag to enable visualization", action="store_true")
        parser.add_argument(
            "--frame_time", help="The frame rate for the visualization", type=float, default=self.frame_time)

        args = parser.parse_args()

        if(args.board == BoardType.Triangle.value):
            if(args.size < 4 or args.size > 8):
                parser.error("Triangle boards must be of size [4-8]")
        elif(args.board == BoardType.Diamond.value):
            if(args.size < 3 or args.size > 6):
                parser.error("Diamond boards must be of size [3-6]")
            if(args.size == 4 and len(args.cell_positions) == 1 and (args.cell_positions[0] == 5 or args.cell_positions[0] == 10)):
                print(
                    "WARNING: A diamond board of size 4 can only be solved with center positions 6 or 9, not 5 or 10.")
        else:
            raise NotImplementedError()

        if(args.critic == CriticType.NEURAL_NETWORK and not args.nn_dim):
            parser.error("--nn_dim is required when --critic = nn")

        for x in args.cell_positions:
            if(x >= args.size**2):
                parser.error(
                    "Cell position {0} is too large for a board of size {1}x{1}".format(x, args.size))

        if(args.df_critic * args.ed_critic >= 1.0):
            parser.error(
                "γλ > 1 is not allowed for critic (i.e. discount factor * eligibility decay must be below 1)")

        if(args.df_actor * args.ed_actor >= 1.0):
            parser.error(
                "γλ > 1 is not allowed for actor (i.e. discount factor * eligibility decay must be below 1)")

        self.board = BoardType(args.board)
        self.board_size = args.size
        self.cell_positions = args.cell_positions
        self.episodes = args.episodes
        self.critic_type = CriticType(args.critic)
        self.nn_dim = args.nn_dim
        self.learning_rate_critic = args.lr_critic
        self.learning_rate_actor = args.lr_actor
        self.eligibility_decay_critic = args.ed_critic
        self.eligibility_decay_actor = args.ed_actor
        self.discount_factor_critic = args.df_critic
        self.discount_factor_actor = args.df_actor
        self.epsilon = args.epsilon
        self.visualize = args.visualize
        self.frame_time = args.frame_time

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
