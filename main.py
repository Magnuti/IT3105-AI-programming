from argument_parser import Arguments
from sim_world import SimWorld
from rl_agent import RL_agent
from visualization import visualize_board

if __name__ == "__main__":
    args = Arguments()
    args.parse_arguments()
    sim_world = SimWorld(args.board, args.cell_positions, args.board_size)
    visualize_board(sim_world.board_type, sim_world.current_state)
    child_states = sim_world.find_child_states()
    for c in child_states:
        visualize_board(sim_world.board_type, c)
