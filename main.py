from argument_parser import Arguments
from sim_world import SimWorld, coordinates_to_1D_index
from rl_agent import RL_agent
from visualization import visualize_board

if __name__ == "__main__":
    args = Arguments()
    args.parse_arguments()

    coordinates_to_1D_index(args.open_cell_positions,
                            args.board_type, args.board_size)

    sim_world = SimWorld(
        args.board_type, args.open_cell_positions, args.board_size)
    rl_agent = RL_agent(sim_world, args.episodes, args.critic_type, args.learning_rate_critic, args.learning_rate_actor, args.eligibility_decay_critic,
                        args.eligibility_decay_actor, args.discount_factor_critic, args.discount_factor_actor, args.epsilon, args.visualize)
    rl_agent.play()
