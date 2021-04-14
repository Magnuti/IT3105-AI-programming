import numpy as np
import time
import random

from mcts import MonteCarloTreeSearch
from function_approximator import ANET
from constants import EpsilonDecayFunction


class RL_agent:
    """
    This class houses the actor (and the critic, if we are using one).
    """

    def __init__(self, sim_world, args, model_save_path, best_model_save_path):
        self.sim_world = sim_world
        # The reason I set this as its own attribute is that it's changed during the runs
        self.epsilon = args.epsilon  # Decays over time
        self.args = args
        self.model_save_path = model_save_path
        self.best_model_save_path = best_model_save_path

        if self.args.epsilon_decay_function == EpsilonDecayFunction.EXPONENTIAL:
            self.epsilon_decay = args.epsilon_decay
        elif self.args.epsilon_decay_function == EpsilonDecayFunction.REVERSED_SIGMOID:
            self.epsilon_decay = args.epsilon_decay
        elif self.args.epsilon_decay_function == EpsilonDecayFunction.LINEAR:
            self.epsilon_decay = self.epsilon / self.args.episodes
        else:
            raise NotImplementedError()

        print("Epsilon decay:", self.epsilon_decay)

        self.actor = Actor(self.sim_world, args)

    def play(self):
        # TODO Think it's helpful to just plot this to see how it's manifesting
        epsilon_history = []

        starting_player = 1
        self.sim_world.reset_game(starting_player)

        episode_save = self.args.episodes // (self.args.games_to_save - 1)
        episode_save_interval = [
            i * episode_save for i in range(self.args.games_to_save)]

        # Deals with non-dividible numbers
        episode_save_interval[-1] = self.args.episodes - 1
        print("Saving every {}th episode".format(episode_save_interval))

        for episode in range(self.args.episodes):
            start = time.time()
            starting_player = 1 - starting_player  # Alternate between 0 and 1
            self.sim_world.reset_game(starting_player)

            last_episode = episode == self.args.episodes - 1
            if(episode % 1 == 0 or last_episode):
                print("--- Episode {} ---".format(episode))
            if episode in episode_save_interval:
                print("Saving episode", episode)
                self.actor.ANET.save_model(episode, self.model_save_path)

                if last_episode:
                    self.actor.ANET.save_model(
                        "best_model", self.best_model_save_path)

            # Epsilon update
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

            if last_episode:
                self.epsilon = 0  # Target policy for last run

            # TODO: remove
            print("\tepsilon:", self.epsilon)

            # First action
            self.actor.pick_next_actual_action(self.epsilon)

            gameover = self.sim_world.get_gameover_and_reward(
                visualization=last_episode)[0]

            # For each step of the episode: do another move
            while not gameover:

                self.actor.pick_next_actual_action(self.epsilon)
                gameover, _ = self.sim_world.get_gameover_and_reward(
                    visualization=last_episode)

            self.actor.start_new_game()

            epsilon_history.append(self.epsilon)

            train_start = time.time()
            self.actor.train_ANET()
            train_used = time.time() - train_start
            print("\tTraining took {} seconds".format(train_used))

            used = time.time() - start
            print("\tThis episode took {} seconds".format(used))


class Actor:
    def __init__(self, sim_world, args):
        # since it refers to same object as RL-agent, why not just store it here
        self.sim_world = sim_world
        # TODO pass in learning rate to ANET ?
        self.ANET = ANET(args.neurons_per_layer, args.activation_functions,
                         args.optimizer, args.learning_rate)
        self.ANET.cache_model_params()
        # TODO we need to pass in explore_constant, which should probably be decaying
        temp_explore_constant = 1
        self.MCTS = MonteCarloTreeSearch(
            explore_constant=temp_explore_constant, simworld=sim_world, ANET=self.ANET, args=args)
        self.replay_buffer = []
        self.args = args

    # assuming that the current state is already picked in simworld
    def pick_next_actual_action(self, epsilon):
        next_state, train_case = self.MCTS.search_next_actual_move(epsilon)
        self.replay_buffer.append(train_case)
        self.sim_world.pick_move(next_state)

    def start_new_game(self):
        self.MCTS.start_new_game()

    def train_ANET(self, plot=False):
        if len(self.replay_buffer) > 256:
            # TODO may take 256 as config param
            # Drop old training cases when the replay buffer reaches a size of 256
            from_index = len(self.replay_buffer) - 256
            self.replay_buffer = self.replay_buffer[from_index:]

        if len(self.replay_buffer) <= self.args.replay_buffer_selection_size:
            # Select the entire replay buffer it is isn't filled up enough
            # TODO may be an idea to skip training until it is somewhat filled up
            # TODO so the beginning cases are not overtrained on
            selection = self.replay_buffer
        else:
            sample_size = self.args.replay_buffer_selection_size
            # Random choice without replacement
            selection = random.sample(self.replay_buffer, k=sample_size)

        # It may be an idea to split replay_buffer into replay_buffer_x and
        # replay_buffer_y so we can skip this part
        x = np.empty((len(selection), len(selection[0][0])))
        y = np.empty((len(selection), len(selection[0][1])))

        for i, train_case in enumerate(selection):
            x[i] = train_case[0]
            y[i] = train_case[1]

        history = self.ANET.fit(
            x, y, batch_size=self.args.mini_batch_size, epochs=self.args.epochs)

        self.ANET.cache_model_params()
