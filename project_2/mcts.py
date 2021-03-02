import math


class MonteCarloTreeSearch:
    '''
    node_representation:
        {s: state, c: [node], p: node|None, Q: {
            a: int}, N_a: {a: int}, N_s: int}
        s = state
        c = children
        p = parent node
        Q{a: int} = monte-carlo value for action a
        N_a{a: int} = visit count for action a
        N_s = visit count for this state (node)
    tree_representation
        {'state': node}

    '''

    def __init__(self, root_state, c, simworld, simulations):
        '''
        args:
            simworld:  actual_simworld, which state is reset at end of search_next_actual_move()
        '''

        # TODO CYT
        self.root = {'s': root_state, 'c': [],
                     'p': None, 'Q': {}, 'N_a': {}, 'N_s': 0}
        # TODO from Jonas: I use a dict of nodes as tree-repr instead of a set of state-hashes,
        # because I assume it will be handy to be able to quickly get a node given it's state.
        # We do lookup in hash_state in tree_search.while_loop.. which seems beneficial!
        # If not needed to lookup the node, I would repr tree as [(set of states in tree), root_node].. but why would we need a list of all the states in tree at all?
        self.tree = {self.get_hashed_state(root_state): self.root}
        self.c = c
        self.simworld = simworld
        self.simulations = simulations

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
        # TODO check exact simworld method
        while not self.simworld.gameOver():
            # TODO check exact simworld method
            state = self.simworld.get_state()
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
        # TODO: check implementation
        child_states = self.simworld.get_child_states(parent_node['s'])
        for i in range(len(child_states)):
            node = self.make_node(child_states[i], parent_node=parent_node)
            # attach child to it's parent
            node['p']['c'].append(node)
            # attach child to tree
            self.tree[self.get_hashed_state(node['s'])] = node

    def leaf_eval(self):
        ...

    def backprop(self):
        ...

    def make_node(self, state, parent_node):
        return {'s': state, 'p': parent_node, 'c': [], 'Q': {}, 'N_a': {}, 'N_s': 0}

    def get_hashed_state(self, state):
        return tuple(state)

    # TODO: be ready to explain this on demo!
    def tree_select_move(self, node, num_child_states):
        if self.black_to_play(node):
            # the math here would be better to do in np, but need to be explicit to be able to cythonize
            values = [(node['Q'][a] + self.c*math.sqrt(math.log(node['N_s']
                                                                ) / node['N_a'][a])) for a in range(num_child_states)]
            action_chosen = values.index(max(values))
        else:
            values = [(node['Q'][a] - self.c*math.sqrt(math.log(node['N_s']
                                                                ) / node['N_a'][a])) for a in range(num_child_states)]
            action_chosen = values.index(min(values))
        return action_chosen

    def black_to_play(self, node):
        # TODO check actual player_to_move representation, and make sure this repr is viable for both nim and hex
        if node['s'][-1] == 0:
            return True
        return False
