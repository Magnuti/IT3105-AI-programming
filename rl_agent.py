import time
import tensorflow as tf
import numpy as np
from collections import defaultdict
from enum import Enum
import random

from function_approximator import FunctionApproximator
from visualization import visualize_board, plot_performance
from constants import CriticType, StateStatus
from sim_world import SimWorld

# --- Terminology ---
# state == board
# SAP = a (state, action) tuple


def make_SAP_hashable(SAP):
    '''
    Returns a tuple(tuple(integers), integer)

    Input tuple(list: integers, int)
    '''
    return (tuple(SAP[0]), SAP[1])


def make_state_hashable(state):
    return tuple(state)


class CriticFuncApp:
    def __init__(self, nn_dims, discount_factor, learning_rate, eligibility_decay):
        self.func_app = FunctionApproximator(nn_dims)
        self.func_app.build_network(lrate=learning_rate)

        self.nn_dims = nn_dims
        self.discount_factor = discount_factor  # γ
        self.learning_rate = learning_rate
        self.eligibility_decay = eligibility_decay  # aka. trace decay λ

    def new_episode(self):
        self.func_app.reset_eligibilities()

    def update_TD_error(self, state, new_state, reward):
        state = tf.convert_to_tensor([state])
        new_state = tf.convert_to_tensor([new_state])
        TD_error_tensor = reward + self.discount_factor * \
            self.func_app.model(new_state) - self.func_app.model(state)
        self.TD_error = TD_error_tensor.numpy()[0, 0]

    def fit_model(self, state):
        self.func_app.fit(state, self.learning_rate, self.TD_error)

    def decay_all_eligibilities(self):
        self.func_app.decay_eligibilities(
            self.discount_factor, self.eligibility_decay)


class CriticTable:
    # This critic is state-based, not state-action-pair-based
    # so, we use V(s) instead of Q(s,a)

    def __init__(self, discount_factor, learning_rate, eligibility_decay):
        # V[s] --> V(s) value of state s, initialized with random values in the interval [0-1)
        self.V = defaultdict(lambda: random.random())

        self.discount_factor = discount_factor  # γ
        self.learning_rate = learning_rate
        self.eligibility_decay = eligibility_decay  # aka. trace decay λ

        self.eligibilities = defaultdict(lambda: 0)

    def new_episode(self):
        # Reset eligibilities
        for key in self.eligibilities.keys():
            self.eligibilities[key] = 0

    def update_TD_error(self, state, new_state, reward):
        self.TD_error = reward + self.discount_factor * \
            self.V[make_state_hashable(new_state)] - \
            self.V[make_state_hashable(state)]

    def set_state_eligibility(self, state, value=1.0):
        self.eligibilities[make_state_hashable(state)] = value

    def update_state_value(self, state):
        self.V[make_state_hashable(state)] += self.learning_rate * \
            self.TD_error * self.eligibilities[make_state_hashable(state)]

    def decay_state_eligibility(self, state):
        self.eligibilities[make_state_hashable(
            state)] *= self.discount_factor * self.eligibility_decay


class Actor:
    # TODO "...the actor MAY consult the critic to get the values of all child states of s..."
    # TODO should it ask everytime or only sometimes? Ask about this

    def __init__(self, learning_rate, discount_factor, eligibility_decay):
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.eligibility_decay = eligibility_decay  # aka. trace decay λ

        # Π(s): the action recommended by the actor when the system is in state s
        # Π(s,a): the actor’s quantitative evaluation of the desirability of choosing action a when in state s
        # So, Π(s) = argmax Π(s,a). I.e. the action from state s that yields the best Π(s,a) value

        # Dict with values for pi(SAP), maps state-action-pairs to values
        # Pi[(s, a)] --> Pi(s, a)
        self.pi = defaultdict(lambda: 0)
        self.eligibilities = defaultdict(lambda: 0)

    def new_episode(self):
        # Reset eligibilities
        for key in self.eligibilities.keys():
            self.eligibilities[key] = 0

    def get_best_action(self, current_state, number_of_child_states):
        '''
        Greedy
        '''
        # Π(s), the one with the highest value of Π(s)
        # Greedy manner
        values = []
        for i in range(number_of_child_states):
            SAP = (current_state, i)
            values.append(self.pi[make_SAP_hashable(SAP)])
        return values.index(max(values))

    def get_next_action(self, current_state, number_of_child_states, epsilon):
        '''
        Epsilon-greedy
        '''

        if number_of_child_states == 0:
            return None

        if random.random() < epsilon:
            # Make random choice
            # ? Including the best action yes?
            return random.randrange(number_of_child_states)
        else:
            # Make greedy choice
            return self.get_best_action(current_state, number_of_child_states)

    def set_SAP_eligibility(self, SAP, value=1.0):
        self.eligibilities[make_SAP_hashable(SAP)] = value

    def update_SAP_policy(self, SAP, TD_error):
        self.pi[make_SAP_hashable(SAP)] += self.learning_rate * \
            TD_error * self.eligibilities[make_SAP_hashable(SAP)]

    def decay_SAP_eligibility(self, SAP):
        self.eligibilities[make_SAP_hashable(
            SAP)] *= self.discount_factor * self.eligibility_decay


class RL_agent:
    def __init__(self, sim_world, episodes, critic_type, nn_dims, learning_rate_critic,
                 learning_rate_actor, eligibility_decay_critic, eligibility_decay_actor,
                 discount_factor_critic, discount_factor_actor, epsilon, epsilon_decay,
                 visualize, visualize_training_episodes, frame_time):
        self.episodes = episodes
        self.critic_type = critic_type
        self.nn_dims = nn_dims
        self.sim_world = sim_world
        self.epsilon = epsilon  # Decays over time
        self.epsilon_decay = epsilon_decay
        self.visualize = visualize
        self.visualize_training_episodes = visualize_training_episodes
        self.frame_time = frame_time

        if(critic_type == CriticType.TABLE):
            self.critic = CriticTable(
                learning_rate_critic, eligibility_decay_critic, discount_factor_critic)
        elif(critic_type == CriticType.NEURAL_NETWORK):
            self.critic = CriticFuncApp(
                nn_dims, learning_rate_critic, eligibility_decay_critic, discount_factor_critic)
        else:
            raise NotImplementedError()

        self.actor = Actor(learning_rate_actor,
                           eligibility_decay_actor, discount_factor_actor)

    def play(self):
        # Previous state-action-pairs in this episode
        SAP_list_in_current_episode = []
        remaining_pegs_list = []
        epsilon_history = []

        for episode in range(self.episodes):
            if(episode % 100 == 0 or episode == self.episodes - 1):
                print("--- Episode {} ---".format(episode))
            self.critic.new_episode()
            self.actor.new_episode()
            SAP_list_in_current_episode.clear()

            self.successor_states, self.successor_states_with_visualization = self.sim_world.find_child_states()

            # Init state and action
            state = self.sim_world.get_current_state_statuses()

            # TODO: is it possible that we miss the actual "best" move, by choosing the best initial action according to the agent?
            action = self.actor.get_best_action(
                state, len(self.successor_states))
            _, state_status = self.sim_world.get_reward_and_state_status(
                len(self.successor_states))

            # TODO plot how epsilon decays over time
            self.epsilon = self.epsilon * self.epsilon_decay
            if(episode == self.episodes - 1):
                self.epsilon = 0  # Target policy for last run

            # For each step of the episode
            while state_status == StateStatus.IN_PROGRESS:
                new_state = self.successor_states[action]
                new_state_with_visualization = self.successor_states_with_visualization[action]
                self.sim_world.pick_new_state(new_state)
                self.successor_states, self.successor_states_with_visualization = self.sim_world.find_child_states()
                reward, state_status = self.sim_world.get_reward_and_state_status(
                    len(self.successor_states))

                new_action = self.actor.get_next_action(
                    new_state, len(self.successor_states), self.epsilon)

                SAP_list_in_current_episode.append((state, action))

                self.actor.set_SAP_eligibility((state, action))

                self.critic.update_TD_error(state, new_state, reward)

                if(self.critic_type == CriticType.TABLE):
                    self.critic.set_state_eligibility(state)
                elif(self.critic_type == CriticType.NEURAL_NETWORK):
                    self.critic.fit_model(state)
                    self.critic.decay_all_eligibilities()
                else:
                    raise NotImplementedError()

                # For SAP so far in this episode
                for (s, a) in SAP_list_in_current_episode:
                    # TODO restrict to only those that have eligibilites > 0 for performance gain

                    if(self.critic_type == CriticType.TABLE):
                        self.critic.update_state_value(s)
                        self.critic.decay_state_eligibility(s)

                    self.actor.update_SAP_policy(
                        (s, a), self.critic.TD_error)
                    self.actor.decay_SAP_eligibility((s, a))

                state = new_state
                action = new_action

                # visualize current game if it's in visualize_training_episodes or this is last episode
                if (self.visualize and (episode in self.visualize_training_episodes or episode == self.episodes - 1)):
                    visualize_board(self.sim_world.graph,
                                    new_state_with_visualization, episode=episode)
                    time.sleep(self.frame_time)

            remaining_pegs_list.append(self.sim_world.get_remaining_pegs())
            epsilon_history.append(self.epsilon)

            # Assuming all boards look the same for now
            self.sim_world.reset_board()

        if(self.visualize):
            plot_performance(remaining_pegs_list, epsilon_history)


if __name__ == "__main__":
    pass
