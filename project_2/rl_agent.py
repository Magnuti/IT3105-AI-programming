import numpy as np
from mcts import MonteCarloTreeSearch
from function_approximator import ANET
from constants import EpsilonDecayFunction
# TODO
# from visualization import visualize_board, plot_performance
import time
import random


class RL_agent:
    """
    This class houses the actor (and the critic, if we are using one).
    """

    def __init__(self, sim_world, args):
        self.sim_world = sim_world
        # The reason I set this as its own attribute is that it's changed during the runs
        self.epsilon = args.epsilon  # Decays over time
        self.args = args

        if args.epsilon_decay_function == EpsilonDecayFunction.LINEAR:
            self.epsilon_decay = self.epsilon / self.args.episodes  # Linear decay
        else:
            self.epsilon_decay = args.epsilon_decay

        self.actor = Actor(self.sim_world, args)

    def play(self):
        # TODO Think it's helpful to just plot this to see how it's manifesting
        epsilon_history = []

        for episode in range(self.args.episodes):
            if(episode % 100 == 0 or episode == self.args.episodes - 1):
                print("--- Episode {} ---".format(episode))

            # self.critic.new_episode()
            # TODO is this needed ?
            # self.actor.new_episode()
            # SAP_list_in_current_episode.clear()

            # EPSILON UPDATE
            if self.args.epsilon_decay_function == EpsilonDecayFunction.EXPONENTIAL:
                self.epsilon *= self.epsilon_decay
            elif self.args.epsilon_decay_function == EpsilonDecayFunction.REVERSED_SIGMOID:
                # Horizontally flipped Sigmoid
                self.epsilon = 1 / \
                    (1+np.exp((episode-(self.args.episodes/2))/(self.args.episodes*0.08)))
            elif self.args.epsilon_decay_function == EpsilonDecayFunction.LINEAR:
                self.epsilon -= self.epsilon_decay
            else:
                raise NotImplementedError()
            if(episode == self.args.episodes - 1):
                self.epsilon = 0  # Target policy for last run

            # TODO: (?) need child_states with visualization like in project 1
            # self.successor_states, self.successor_states_with_visualization = self.sim_world.find_child_states()
            # self.child_states = self.sim_world.get_child_states()

            # ! TODO Reset MCTS tree for the new actual-game-episode

            # First action
            # TODO epsilon handled correctly in actor?
            self.actor.pick_next_actual_action(self.epsilon)

            gameover = self.sim_world.get_gameover_and_reward()[0]

            # For each step of the episode: do another move
            while not gameover:
                # TODO
                # new_state_with_visualization = self.successor_states_with_visualization[action]

                # TODO don't think we need child states here, but the visualization part should be implemented somewhere in the code
                # self.child_states, self.successor_states_with_visualization = self.sim_world.get_child_states()

                self.actor.pick_next_actual_action(self.epsilon)
                # TODO should reward be fetched here too?
                gameover, reward = self.sim_world.get_gameover_and_reward()

                # # visualize current episode if it's in visualize_training_episodes or last episode
                # if (self.args.visualize and (episode in self.args.visualize_training_episodes or episode == self.args.episodes - 1)):
                #     # TODO
                #     # visualize_board(self.sim_world.graph, new_state_with_visualization, episode=episode)
                #     # time.sleep(self.args.frame_time)
                #     pass

            epsilon_history.append(self.epsilon)

            if episode < self.args.episodes - 1:
                self.sim_world.reset_game()
                self.actor.train_ANET()


class Actor:
    def __init__(self, sim_world, args):
        # since it refers to same object as RL-agent, why not just store it here
        self.sim_world = sim_world
        # TODO pass in learning rate to ANET ?
        self.ANET = ANET(args.neurons_per_layer, args.activation_functions)
        # TODO we need to pass in c, which should probably be decaying
        temp_c = 0.7
        self.MCTS = MonteCarloTreeSearch(
            root_state=sim_world.get_game_state(), c=temp_c, simworld=sim_world, ANET=self.ANET, args=args)
        self.replay_buffer = []
        self.args = args

    # assuming that the current state is already picked in simworld
    def pick_next_actual_action(self, epsilon):
        # TODO is it right that the actor only consults the MCTS for next actual move?
        next_state, train_case = self.MCTS.search_next_actual_move(epsilon)
        self.replay_buffer.append(train_case)
        self.sim_world.pick_move(next_state)

    def train_ANET(self, plot=False):
        batch = random.choices(self.replay_buffer, k=self.args.mini_batch_size)
        x = []
        y = []
        for c in batch:
            x.append(c[0])
            y.append(c[1])

        x = np.array(x)
        y = np.array(y)
        # batch_size None, since we are already serving only 1 mini-batch
        history = self.ANET.fit(x, y, batch_size=None, epochs=self.args.epochs)

        if plot:
            # TODO plot training graph (?), but atm there is no validation set
            pass
