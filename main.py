from argument_parser import Arguments
from sim_world import SimWorld
from rl_agent import RL_agent
from visualization import visualize_board

if __name__ == "__main__":
    args = Arguments()
    args.parse_arguments()

    sim_world = SimWorld(
        args.board_type, args.open_cell_positions, args.board_size)
    rl_agent = RL_agent(sim_world, args.episodes, args.critic_type, args.nn_dims, args.learning_rate_critic, args.learning_rate_actor, args.eligibility_decay_critic,
                        args.eligibility_decay_actor, args.discount_factor_critic, args.discount_factor_actor, args.epsilon, args.epsilon_decay, args.visualize)
    rl_agent.play()
