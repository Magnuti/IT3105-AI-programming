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
    def __init__(self, root_state, c, simworld):
        # TODO CYT
        self.root = {'s': root_state, 'c': [],
                     'p': None, 'Q': {}, 'N_a': {}, 'N_s': 0}
        # TODO from Jonas: I use a dict of nodes as tree-repr instead of a set of state-hashes,
        # because I assume it will be handy to be able to quickly get a node given it's state.
        # But this may not be true. In that case I would repr tree as [(set of states in tree), root_node]
        self.tree = {tuple(self.root['s']): self.root}
        self.c = c
        self.simworld

    def simulate(self):
        self.tree_search()

    def tree_search(self):
        t = 0
        # TODO check exact simworld method
        while not self.simworld.gameOver():
            # TODO check exact simworld method
            # TODO: (?) need hashed_state to search for it in keys below
            state = simworld.get_state()
            # TODO CYT, but first check that it's not a bottleneck
            if state not in self.tree['nodes'].keys():
                pass

    def node_expand(self):
        ...

    def leaf_eval(self):
        ...

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
