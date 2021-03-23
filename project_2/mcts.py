import numpy as np
import random
from function_approximator import ANET
import math


class MonteCarloTreeSearch:
    '''
    # TODO: don't know if float is the optimal var-type for the Q and E value (in terms of num decimals)
    node_representation:
        {s: state, c: [node], p: node|None, Q_a: {
            a: float}, N_a: {a: int}, N_s: int, E_a: {a: float}}
        s = state
        c = children
        p = parent node
        Q_a = monte-carlo value for action a (outgoing)
        N_a = visit count for action a (outgoing)
        N_s = visit count for this state (node)
        E_a = sum of all final-state rewards this SAP has been involved in (so far) (outgoing)
    tree_representation
        {'state': node}

    '''

    def __init__(self, root_state, c, simworld, args):
        '''
        args:
            simworld:  actual_simworld, which state is reset at end of search_next_actual_move()
        '''

        # TODO CYT
        self.root = {'s': root_state, 'c': [],
                     'p': None, 'Q_a': {}, 'N_a': {}, 'N_s': 0, 'E_a': {}}
        # TODO from Jonas: I use a dict of nodes as tree-repr instead of a set of state-hashes,
        # because I assume it will be handy to be able to quickly get a node given it's state.
        # We do lookup in hash_state in tree_search.while_loop.. which seems beneficial!
        # If not needed to lookup the node, I would repr tree as [(set of states in tree), root_node].. but why would we need a list of all the states in tree at all?
        self.tree = {self.get_hashed_state(root_state): self.root}
        self.c = c
        self.simworld = simworld
        self.simulations = args.simulations
        self.ANET = ANET(args.neurons_per_layer, args.activation_functions)

    def search_next_actual_move(self):
        for s in range(self.simulations):
            self.simulate()
        # reset simworld to the root_state (actual_state before search)
        self.simworld.pick_move(self.root['s'])

        # based on all the sims, choose recommended actual_move
        children = self.root['c']
        move_num = self.tree_select_move(self.root, len(children))
        return children[move_num]

    def simulate(self):
        self.simworld.pick_move(self.root['s'])
        leaf_node = self.tree_search()
        # TODO: no copy of simworld is needed in leaf_eval, since we reset the state on top of this func
        # z = leaf-eval(self.simworld)
        # backprop (start from leaf and climb to top)

    # using UCT algorithm
    def tree_search(self):
        previous_node = None
        while not self.simworld.get_gameover_and_reward()[0]:
            state = self.simworld.get_game_state()
            hash_state = self.get_hashed_state(state)

            # TODO CYT, but first check that it's not a bottleneck
            node = self.tree[hash_state]

            # if node is leaf
            if not len(node['c']):
                # find child states and insert into tree
                self.node_expand(node)
                # return the original leaf-node
                return node

            a = self.tree_select_move(node, len(node['c']))
            self.simworld.pick_move(node['c'][a])
            previous_node = node

        # Do one last evaluation (rollout/critic) of the parent of the game_over_state,
        # this way we may discover a better move than tree_policy chose
        return previous_node

    def node_expand(self, parent_node):
        child_states = self.simworld.get_child_states(parent_node['s'])
        skipped_moves = 0
        for i in range(len(child_states)):
            if not child_states[i]:
                # if it's an illegal move
                skipped_moves += 1
                continue
            node = self.make_node(child_states[i], parent_node=parent_node)
            # attach child to it's parent
            node['p']['c'].append(node)
            # init N(s,a), Q(s,a), E(s,a) counters on parent, for this action
            actual_action_index = i-skipped_moves
            node['N_a'][actual_action_index] = 0
            node['Q_a'][actual_action_index] = 0
            node['E_a'][actual_action_index] = 0
            # attach child to tree
            self.tree[self.get_hashed_state(node['s'])] = node

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

        return reward

    def backprop(self, leaf_node, z):
        def climb_and_update(node, node_child):
            # Upd N(s)
            node['N_s'] += 1
            # which action/child we climbed up from <int>
            a = node['c'].index(node_child)
            # Upd N(s,a), E, Q(s, a)
            node['N_a'][a] += 1
            node['E_a'][a] += z
            node['Q_a'][a] = node['E_a'][a] / node['N_a'][a]

            if node['p']:
                climb_and_update(node['p'], node)
        if leaf_node['p']:
            climb_and_update(leaf_node['p'], leaf_node)

    def make_node(self, state, parent_node):
        return {'s': state, 'p': parent_node, 'c': [], 'Q_a': {}, 'N_a': {}, 'N_s': 0, 'E_a': {}}

    def get_hashed_state(self, state):
        return tuple(state)

    # TODO: be ready to explain this on demo!
    def tree_select_move(self, node, num_child_states):
        if self.black_to_play(node):
            # Get the greedy best-action coice for player 1
            # the math here would be better to do in np, but need to be explicit to be able to cythonize
            values = [(node['Q_a'][a] + self.c*math.sqrt(math.log(node['N_s']
                                                                  ) / node['N_a'][a])) for a in range(num_child_states)]
            action_chosen = values.index(max(values))
        else:
            values = [(node['Q_a'][a] - self.c*math.sqrt(math.log(node['N_s']
                                                                  ) / node['N_a'][a])) for a in range(num_child_states)]
            action_chosen = values.index(min(values))
        return action_chosen

    def black_to_play(self, node):
        # Differences between player ID [0, 1] and [1, 0]
        if node['s'][0] == 0:
            return True
        return False
