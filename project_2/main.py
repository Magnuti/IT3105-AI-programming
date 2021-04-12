import pathlib
import shutil
import numpy as np

from argument_parser import Arguments
from sim_world import SimWorldNim, SimWorldHex
from constants import GameType
from rl_agent import RL_agent
from topp import TournamentOfProgressivePolicies
from function_approximator import ANET
from visualization import visualize_board_manually

if __name__ == "__main__":
    args = Arguments()
    args.parse_arguments()

    # TODO: init RL_agent and play
    if args.game_type == GameType.NIM:
        sim_world = SimWorldNim(args.nim_N, args.nim_K)
        # print(reward)
    elif args.game_type == GameType.HEX:
        sim_world = SimWorldHex(args.board_size, args.visualize)
    else:
        raise NotImplementedError()

    model_save_path = pathlib.Path("saved_models")
    best_model_save_path = pathlib.Path("best_model")

    # if(model_save_path.exists()):
    #     # Remove all saved models so we start of with a clean folder
    #     shutil.rmtree(model_save_path)
    # model_save_path.mkdir(exist_ok=True)

    # if(best_model_save_path.exists()):
    #     # Remove all saved models so we start of with a clean folder
    #     shutil.rmtree(best_model_save_path)
    # best_model_save_path.mkdir(exist_ok=True)

    # _RL_agent = RL_agent(sim_world, args, model_save_path,
    #                      best_model_save_path)
    # _RL_agent.play()

    topp = TournamentOfProgressivePolicies(args, sim_world, model_save_path)
    topp.round_robin_tournament(args.games_between_agents)

    # Best vs. random
    victories = 0
    anet = ANET(args.neurons_per_layer, args.activation_functions)
    anet.load_model_path_known(
        best_model_save_path.joinpath("anet_episode_best_model"))

    frame_time = 0.1
    games = 100
    starting_player = 1
    print("Our ANET player is black")
    for game in range(games):
        starting_player = 1 - starting_player
        sim_world.reset_game(starting_player)

        graph_list = []
        state_status_list_list = []

        gameover, reward = sim_world.get_gameover_and_reward()
        move_count = 0
        while not gameover:
            move_count += 1
            # Batch size is 1 so we get the output by indexing [0]

            if sim_world.current_player_array_to_id():
                # Player 1's turn
                output_propabilities = anet.forward(
                    sim_world.get_game_state()).numpy()[0]
                child_states = sim_world.get_child_states()

                # Set illegal actions to 0 probability
                for i, state in enumerate(child_states):
                    if state is None:
                        output_propabilities[i] = 0.0

                # Normalize the new probabilities
                output_propabilities /= sum(output_propabilities)

                # Make greedy choice
                move_index = np.argmax(output_propabilities)
                sim_world.pick_move(child_states[move_index])
            else:
                child_states = sim_world.get_child_states()
                legal_child_states = []
                for i in range(len(child_states)):
                    if child_states[i] is not None:
                        legal_child_states.append(child_states[i])

                # For simplicity we just select a random legal action here
                legal_action_index = np.random.randint(
                    0, len(legal_child_states))
                next_state = legal_child_states[legal_action_index]
                sim_world.pick_move(next_state)

            gameover, reward = sim_world.get_gameover_and_reward()

            graph_list.append(sim_world.graph)
            state_status_list_list.append(
                list(map(lambda x: x.status, sim_world.cells)))

        # TODO try to set best player as min as well
        if reward == 1:
            victories += 1

        if args.visualize:
            visualize_board_manually(graph_list, state_status_list_list)

    print("won {} out of {} games".format(victories, games))
