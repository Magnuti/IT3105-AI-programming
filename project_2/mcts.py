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

    '''

    def __init__(self, root_state):
        # TODO CYT
        root = {'s': root_state, 'c': [],
                'p': None, 'Q': {}, 'N_a': {}, 'N_s': 0}

    def tree_search(self):
        ...

    def node_expand(self):
        ...

    def leaf_eval(self):
        ...

    def backprop(self):
        ...

    def make_node(self, state, parent):
        # TODO: is it right to init N_s as 0 here?
        return {'s': state, 'c': [], 'p': parent, 'Q': {}, 'N_a': {}, 'N_s': 0}
