from argument_parser import Arguments
from sim_world import SimWorldNim
from mcts import MonteCarloTreeSearch
from constants import GameType

if __name__ == "__main__":
    args = Arguments()
    args.parse_arguments()

    # TODO: init RL_agent and play
    if args.game_type == GameType.NIM:
        sim_world = SimWorldNim(args.nim_N, args.nim_K)
        # print(reward)
    elif args.game_type == GameType.HEX:
        raise NotImplementedError()
    else:
        raise NotImplementedError()
