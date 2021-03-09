import numpy as np
import random
from function_approximator import ANET


class MonteCarloTreeSearch:
    '''
    node_representation:
        {s: state, c: [node], p: node|None, Q: {a: int}, N_a: {a: int}, N_s: int}
        s = state
        c = children
        p = parent node
        Q{a: int} = monte-carlo value for action a
        N_a{a: int} = visit count for action a
        N_s = visit count for this state (node)
    tree_representation
        {'state': node}
        nodes: a dict of all nodes in tree (hashed by state)

    '''

    # TODO as of now, simworld sent into MCTS needs to be a copy of the simworld with actual_board
    def __init__(self, root_state, c, simworld, args):
        # TODO CYT
        self.root = {'s': root_state, 'c': [],
                     'p': None, 'Q': {}, 'N_a': {}, 'N_s': 0}
        # TODO from Jonas: I use a dict of nodes as tree-repr instead of a set of state-hashes,
        # because I assume it will be handy to be able to quickly get a node given it's state.
        # But this may not be true. In that case I would repr tree as [(set of states in tree), root_node]
        # self.tree = {tuple(self.root['s']): self.root}
        self.c = c
        self.simworld = simworld
        self.ANET = ANET(args.neurons_per_layer, args.activation_functions)

    def simulate(self):
        self.tree_search()

    def tree_search(self):
        t = 0
        # TODO check exact simworld method
        while not self.simworld.gameOver():
            # TODO check exact simworld method
            # TODO: (?) need hashed_state to search for it in keys below
            state = self.simworld.get_state()
            # TODO CYT, but first check that it's not a bottleneck
            if state not in self.tree['nodes'].keys():
                pass

    def node_expand(self):
        ...

    def leaf_eval(self, leaf_node_state, epsilon):
        """
        Estimates the value of a leaf node by doing a rollout simulation using
        the default policy from the leaf node’s state to a final state.

        If we use rollout this value is the reward that is given when the
        simulated game is finished (e.g., -1 or 1). If we are using a critic,
        however, this value is an evaluation value (e.g. 0.547).

        Args:
            leaf_node_state: np.ndarray
                The state of the board with the first two indices being player bits.
            epsilon: float
                Select the best action with a probability of 1-epsilon, and a random
                action with probability epsilon. The random action will have
                probability > 0, since we cannot pick illegal actions.
        Returns:
            float: the value of the leaf node.
        """

        saved_gamed_state = np.copy(leaf_node_state)

        gameover, reward = self.simworld.get_gameover_and_reward()
        while not gameover:
            # Batch size is 1 so we get the output by indexing [0]
            output_propabilities = self.ANET.forward(
                leaf_node_state).numpy()[0]

            child_states = self.simworld.get_child_states()

            # Set illegal actions to 0 probability
            for i, state in enumerate(child_states):
                if state is None:
                    output_propabilities[i] = 0.0

            # Normalize the new probabilities
            output_propabilities /= sum(output_propabilities)

            if random.random() < epsilon:
                # Make random choice (including the best action)
                legal_child_states = []
                for state in child_states:
                    if state is not None:
                        legal_child_states.append(state)

                choice = random.choice(legal_child_states)
                self.simworld.pick_move(choice)
            else:
                # Make greedy choice
                move_index = np.argmax(output_propabilities)
                self.simworld.pick_move(child_states[move_index])

            gameover, reward = self.simworld.get_gameover_and_reward()

        self.simworld.set_state_and_player(saved_gamed_state)

        return reward

    def backprop(self):
        ...

    def make_node(self, state, parent):
        # TODO: is it right to init N_s as 0 here?
        return {'s': state, 'c': [], 'p': parent, 'Q': {}, 'N_a': {}, 'N_s': 0}

    def insert_node(self, node):
        # attach child to it's parent
        node['p']['c'].append(node)
        self.tree[self.get_hashed_state(node)] = node

    def get_hashed_state(self, node):
        return tuple(node['s'])
