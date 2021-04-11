import tensorflow as tf
from tensorflow import keras
import numpy as np

from visualization import visualize_board, keep_board_visualization_visible
from sim_world import SimWorldNim, SimWorldHex
from argument_parser import Arguments
from function_approximator import ANET


class TournamentOfProgressivePolicies:
    def __init__(self, args, sim_world, model_save_path):
        self.args = args
        self.sim_world = sim_world
        self.model_save_path = model_save_path

    def round_robin_tournament(self, games_between_agents):
        """
        Plays a round-robin tournament between the save agents, such that every
        agent plays every other agent in one series of games_between_agents
        games.

        Args:
            games_between_agents: int
                how many games to play between each agent
        """

        anets = {}
        victories_per_anet = {}
        for path in self.model_save_path.iterdir():
            anet = ANET(self.args.neurons_per_layer,
                        self.args.activation_functions)
            anet.load_model_path_known(path)
            anets[path.name] = anet
            victories_per_anet[path.name] = 0

        played_matches = set()
        for anet_0_name, anet_0 in anets.items():
            for anet_1_name, anet_1 in anets.items():
                if anet_0_name == anet_1_name:
                    continue

                match_id_1 = (anet_0_name, anet_1_name)
                match_id_2 = (anet_1_name, anet_0_name)

                if match_id_1 in played_matches or match_id_2 in played_matches:
                    # Skip mathces that have already been played between agents
                    continue

                played_matches.add(match_id_1)
                played_matches.add(match_id_2)

                starting_player = 1
                for game in range(games_between_agents):
                    # print("\nPlaying a game between {} and {}".format(
                    # anet_0_name, anet_1_name))
                    starting_player = 1 - starting_player  # Alternate between 0 and 1
                    # Play a game between model_0 and model_1
                    self.sim_world.reset_game(starting_player)

                    gameover, reward = self.sim_world.get_gameover_and_reward()
                    while not gameover:
                        # Batch size is 1 so we get the output by indexing [0]

                        # TODO try to alternate who is player 0 and 1 between the ANETs as well
                        if self.sim_world.current_player_array_to_id():
                            # Player 1's turn
                            output_propabilities = anet_1.forward(
                                self.sim_world.get_game_state()).numpy()[0]
                        else:
                            # Player 0's turn
                            output_propabilities = anet_0.forward(
                                self.sim_world.get_game_state()).numpy()[0]

                        child_states = self.sim_world.get_child_states()

                        # Set illegal actions to 0 probability
                        for i, state in enumerate(child_states):
                            if state is None:
                                output_propabilities[i] = 0.0

                        # Normalize the new probabilities
                        output_propabilities /= sum(output_propabilities)

                        # Make greedy choice
                        move_index = np.argmax(output_propabilities)
                        self.sim_world.pick_move(child_states[move_index])

                        gameover, reward = self.sim_world.get_gameover_and_reward()

                        # visualize_board(self.sim_world.graph, list(
                        #     map(lambda x: x.status, self.sim_world.cells)), 0)
                        # self.sim_world.print_current_game_state()

                    if reward == 1:
                        victories_per_anet[anet_1_name] += 1
                        # print("Black (player 1, player 2 in project spec) wins")
                    else:
                        # print("Red (player 0, player 1 in project spec) wins")
                        victories_per_anet[anet_0_name] += 1

                    # visualize_board(self.sim_world.graph, list(
                    #     map(lambda x: x.status, self.sim_world.cells)), 0)
                    # keep_board_visualization_visible()

        for key, value in victories_per_anet.items():
            print(key, "won", value, "times")


if __name__ == "__main__":
    args = Arguments()
    args.parse_arguments()
    anet = ANET(args.neurons_per_layer, args.activation_functions)
    print("Input shape:", anet.model.input_shape)
    anet.model.summary()

    # TODO game_type
    sim_world = SimWorldHex(args.board_size)

    topp = TournamentOfProgressivePolicies(args, sim_world)

    # Test save
    anet_0 = ANET(args.neurons_per_layer, args.activation_functions)
    anet_50 = ANET(args.neurons_per_layer, args.activation_functions)
    anet_100 = ANET(args.neurons_per_layer, args.activation_functions)

    anet_0.save_model(0)
    anet_50.save_model(50)
    anet_100.save_model(100)

    # Test load

    # anet_0 = ANET(args.neurons_per_layer, args.activation_functions)
    # anet_50 = ANET(args.neurons_per_layer, args.activation_functions)
    # anet_100 = ANET(args.neurons_per_layer, args.activation_functions)

    # anet_0.load_model(0)
    # anet_50.load_model(50)
    # anet_100.load_model(100)

    topp.round_robin_tournament(25)
