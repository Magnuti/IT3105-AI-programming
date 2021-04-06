from argument_parser import Arguments
from sim_world import SimWorldNim, SimWorldHex
from constants import GameType
from rl_agent import RL_agent

if __name__ == "__main__":
    args = Arguments()
    args.parse_arguments()

    # TODO: init RL_agent and play
    if args.game_type == GameType.NIM:
        sim_world = SimWorldNim(args.nim_N, args.nim_K)
        # print(reward)
    elif args.game_type == GameType.HEX:
        sim_world = SimWorldHex(args.board_size)
    else:
        raise NotImplementedError()

    _RL_agent = RL_agent(sim_world, args)
    _RL_agent.play()
