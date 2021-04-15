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

    if args.game_type == GameType.NIM:
        sim_world = SimWorldNim(args.nim_N, args.nim_K)
    elif args.game_type == GameType.HEX:
        sim_world = SimWorldHex(args.board_size, args.visualize)
    else:
        raise NotImplementedError()

    model_save_path = pathlib.Path("saved_models")
    best_model_save_path = pathlib.Path("best_model")

    # TODO temporarily
    train = True
    # train = False

    if train:
        if(model_save_path.exists()):
            # Remove all saved models so we start of with a clean folder
            shutil.rmtree(model_save_path)
        model_save_path.mkdir(exist_ok=True)

        if(best_model_save_path.exists()):
            # Remove all saved models so we start of with a clean folder
            shutil.rmtree(best_model_save_path)
        best_model_save_path.mkdir(exist_ok=True)

        _RL_agent = RL_agent(sim_world, args, model_save_path,
                             best_model_save_path)
        _RL_agent.play()

    topp = TournamentOfProgressivePolicies(args, sim_world, model_save_path)
    topp.round_robin_tournament(args.games_between_agents)
