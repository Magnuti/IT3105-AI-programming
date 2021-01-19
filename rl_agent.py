from collections import defaultdict
from enum import Enum
import random

from sim_world import SimWorld
from constants import CriticType, StateStatus
from visualization import visualize_board, plot_performance


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


class Critic:
    def __init__(self, critic_type, discount_factor, learning_rate, eligibility_decay):
        if(critic_type == CriticType.TABLE):
            # ? Should we use V(s) or Q(s, a) ?
            # ? I think V(s) as mentioned on page 6
            # V[s] --> V(s) value of state s, initialized with random values in the interval [0-1)
            self.V = defaultdict(lambda: random.random())
        elif(critic_type == CriticType.NEURAL_NETWORK):
            raise NotImplementedError()
        else:
            raise NotImplementedError()

        self.critic_type = critic_type  # Table or nn
        # self.state = state
        # self.successor_states = successor_states
        self.discount_factor = discount_factor  # γ
        self.learning_rate = learning_rate

        self.eligibility_decay = eligibility_decay  # ? aka. trace decay λ
        # self.TD_error = 0

        self.eligibilities = defaultdict(lambda: 0)

    def update_TD_error(self, state, new_state, reward):
        self.TD_error = reward + self.discount_factor * \
            self.V[make_state_hashable(new_state)] - \
            self.V[make_state_hashable(state)]

    def set_eligibility_by_state(self, state, value=1.0):
        self.eligibilities[make_state_hashable(state)] = value

    def new_episode(self):
        # Reset eligibilities
        for key in self.eligibilities.keys():
            self.eligibilities[key] = 0

    def update_V_by_state(self, state):
        self.V[make_state_hashable(state)] += learning_rate * \
            self.TD_error * self.eligibilities[make_state_hashable(state)]

    def update_eligibilities_by_state(self, state):
        self.eligibilities[make_state_hashable(
            state)] = self.discount_factor * self.eligibility_decay * self.eligibilities[make_state_hashable(state)]


class Actor:
    # TODO "...the actor MAY consult the critic to get the values of all child states of s..."
    # TODO should it ask everytime or only sometimes? Ask about this

    def __init__(self, learning_rate, discount_factor, eligibility_decay, epsilon):
        # self.state = state
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.eligibility_decay = eligibility_decay  # ? aka. trace decay λ
        # TODO decrement epsilon over time and set to 0 on final lap to make behavior policy = target policy?
        self.epsilon = epsilon

        # Π(s): the action recommended by the actor when the system is in state s
        # Π(s,a): the actor’s quantitative evaluation of the desirability of choosing action a when in state s
        # So, Π(s) = argmax Π(s,a). I.e. the action from state s that yields the best Π(s,a) value

        # Dict with values for pi(SAP), maps state-action-pairs to values
        # Policy(s) --> Π(s) # ! pi(s) or pi(SAP) ???
        # Pi[(s, a)] --> Pi(s, a)
        self.pi = defaultdict(lambda: 0)
        self.eligibilities = defaultdict(lambda: 0)

        # self.new_episode()

    def new_episode(self):
        # Reset eligibilities
        for key in self.eligibilities.keys():
            self.eligibilities[key] = 0

    def set_TD_error(self, TD_error):
        self.TD_error = TD_error

    def get_best_action_of_child_states(self, child_states):
        '''
        Greedy
        '''
        # Π(s), the one with the highest value of Π(s)
        # Greedy manner, but since we use epsilon-greedy manner we don't need this yet
        raise NotImplementedError()

    def get_next_action(self, current_state, number_of_child_states):
        '''
        Epsilon-greedy
        '''
        if(random.random() < self.epsilon):
            # Make random choice
            # ? Including the best action yes?
            return random.randrange(number_of_child_states)
        else:
            # Make greedy choice
            values = []
            for i in range(number_of_child_states):
                SAP = (current_state, i)
                values.append(self.pi[make_SAP_hashable(SAP)])
            return values.index(max(values))

    def set_eligibility_by_SAP(self, SAP, value=1.0):
        self.eligibilities[make_SAP_hashable(SAP)] = value

    def update_pi_by_SAP(self, SAP, TD_error):
        self.pi += self.learning_rate * TD_error * \
            self.eligibilites[make_SAP_hashable(SAP)]

    def update_eligibilities_by_SAP(self, SAP, trace_decay):
        self.eligibilities[make_SAP_hashable(
            SAP)] = self.discount_factor * trace_decay * self.eligibilities[make_SAP_hashable(SAP)]


class RL_agent:
    def __init__(self, sim_world, episodes, critic_type, learning_rate_critic, learning_rate_actor, eligibility_decay_critic, eligibility_decay_actor, discount_factor_critic, discount_factor_actor, epsilon, visualize):
        self.episodes = episodes
        self.critic_type = critic_type
        self.sim_world = sim_world
        self.visualize = visualize

        self.critic = Critic(critic_type, learning_rate_critic,
                             eligibility_decay_critic, discount_factor_critic)
        self.actor = Actor(learning_rate_actor,
                           eligibility_decay_actor, discount_factor_actor, epsilon)

        # Previous state-action-pairs in this episode
        self.SAP_list_in_current_episode = []

    def play(self):
        remaining_pegs_list = []

        # ? State-based critic, should we use SAP instead?
        for episode in range(self.episodes):
            print("--- Episode {} ---".format(episode))
            self.critic.new_episode()
            self.actor.new_episode()
            self.SAP_list_in_current_episode.clear()

            self.successor_states, self.successor_states_with_visualization = self.sim_world.find_child_states()

            # Init state and action
            state = self.sim_world.current_state
            action = self.actor.get_next_action(
                self.sim_world.current_state, len(self.successor_states))
            _, state_status = self.sim_world.get_reward_and_state_status()

            # For each step of the episode
            while state_status == StateStatus.IN_PROGRESS:
                new_state = self.successor_states[action]
                new_state_with_visualization = self.successor_states_with_visualization[action]
                self.sim_world.pick_new_state(new_state)
                self.successor_states, self.successor_states_with_visualization = self.sim_world.find_child_states()
                reward, state_status = self.sim_world.get_reward_and_state_status()
                self.actor.set_eligibility_by_SAP((state, action))

                self.critic.update_TD_error(state, new_state, reward)
                self.critic.set_eligibility_by_state(state)

                # For SAP in this episode
                for (s, a) in self.SAP_list_in_current_episode:
                    # TODO restrict to only those that have eligibilites > 0 for performance gain
                    self.critic.update_V_by_state(s)
                    self.critic.update_eligibilities_by_state(s)

                    self.actor.update_pi_by_SAP((s, a), self.critic.TD_error)
                    self.actor.update_eligibilities_by_SAP((s, a))

                if(state_status == StateStatus.IN_PROGRESS):
                    # Line 2. in the algorithm, but a' is not needed until here, so we place it here instead of futher up
                    new_action = self.actor.get_next_action(
                        self.sim_world.current_state, len(self.successor_states))

                    state = new_state
                    action = new_action

                if(self.visualize and episode == self.episodes - 1):
                    # TODO create automatic visualization animation with given frame rate by args
                    visualize_board(self.sim_world.board_type,
                                    new_state_with_visualization)

            remaining_pegs_list.append(self.sim_world.get_remaining_pegs())

            # Assuming all boards look the same for now
            self.sim_world.reset_board()

        if(self.visualize):
            plot_performance(remaining_pegs_list)


if __name__ == "__main__":
    pass
