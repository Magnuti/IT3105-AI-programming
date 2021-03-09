from argument_parser import Arguments
from sim_world import SimWorldNim
from mcts import MonteCarloTreeSearch
from constants import GameType

if __name__ == "__main__":
    args = Arguments()
    args.parse_arguments()

    if args.game_type == GameType.NIM:
        sim_world = SimWorldNim(args.nim_N, args.nim_K)
        mcts = MonteCarloTreeSearch(None, None, sim_world, args)
        reward = mcts.leaf_eval(sim_world.get_game_state())
        print(reward)
    elif args.game_type == GameType.HEX:
        raise NotImplementedError()
    else:
        raise NotImplementedError()
