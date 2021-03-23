import numpy as np
import random
import math


# TODO CYT: would it make sense to inherit from Node-class, so this class understands the node-attribs? (if that's how inheritance even works)
class MonteCarloTreeSearch:
    '''
    tree_representation
        {'state': node}

    '''

    def __init__(self, root_state, explore_constant, simworld, ANET, args):
        '''
        args:
            simworld:  actual_simworld, which state is reset at end of search_next_actual_move()
        '''

        # TODO CYT
        self.explore_constant = explore_constant
        self.simworld = simworld
        self.simulations = args.simulations
        self.ANET = ANET
        self.num_childstates = args.neurons_per_layer[len(
            args.neurons_per_layer) - 1]
        self.root = None
        self.tree = {self.get_hashed_state(root_state): self.root}

    def search_next_actual_move(self, epsilon):
        if not self.root:
            self.root = self.make_node(self.simworld.get_game_state(), None)
        else:
            self.root = self.prune_tree(self.simworld.get_game_state())

        # TODO prune tree according to new root
        for _ in range(self.simulations):
            self.simulate(epsilon)
        # reset simworld to the root_state (actual_state before search)
        self.simworld.pick_move(self.root.state)

        # based on all the sims, choose recommended actual_move
        children = self.root.children
        move_num = self.tree_select_move(self.root, len(children))

        # Make training case for root_node
        target_dist = np.array([self.root.action_visit[a]
                                for a in range(len(children))])
        target_norm = np.sum(target_dist)
        target_dist /= target_norm
        return children[move_num].state, [self.root.state, target_dist]

    def simulate(self, epsilon):
        self.simworld.pick_move(self.root.state)
        leaf_node = self.tree_search()
        self.simworld.pick_move(leaf_node.state)
        z = self.leaf_eval(epsilon)
        self.backprop(leaf_node, z)

    # using UCT algorithm
    def tree_search(self):
        previous_node = None
        # This only gets set to False if a gameover node that already exists in the tree is picked. This pre-existing node would have children (it's expanded) but all childs are None
        continue_search = True
        while continue_search:
            state = self.simworld.get_game_state()
            hash_state = self.get_hashed_state(state)

            # TODO CYT, but first check that it's not a bottleneck
            node = self.tree[hash_state]

            # if node is leaf (means unexpanded (which DOESN'T inform anything about it being a gameover))
            if not node.children:
                # find child states and insert into tree
                self.node_expand(node)
                # return the original leaf-node
                return node

            previous_node = node
            if not node.gameover:
                a = self.tree_select_move(node, len(node.children))
                self.simworld.pick_move(node.children[a].state)
            else:
                continue_search = False

        # Do one last evaluation (rollout/critic) of the parent of the game_over_state,
        # this way we may discover a better move than tree_policy chose
        return previous_node

    def node_expand(self, parent_node):
        self.simworld.pick_move(parent_node.state)
        child_states = self.simworld.get_child_states()
        has_legal_child = False

        # TODO CYT: type is [Node|None], this is a fixed size array
        children = [None]*self.num_childstates
        for i in range(len(child_states)):
            # this mean the child_state is legal (for illegal children, the related position in children[] is already None)
            if child_states[i]:
                has_legal_child = True
                node = self.make_node(child_states[i], parent_node=parent_node)
                # attach child to our lookup-tree
                self.tree[self.get_hashed_state(node.state)] = node
                # attach child to it's parent
                children[i] = node
        if not has_legal_child:
            parent_node.gameover = True
        parent_node.children = children

    def leaf_eval(self, epsilon):
        """
        Estimates the value of a leaf node by doing a rollout simulation using
        the default policy from the leaf node’s state to a final state.

        If we use rollout this value is the reward that is given when the
        simulated game is finished (e.g., -1 or 1). If we are using a critic,
        however, this value is an evaluation value (e.g. 0.547).

        Args:
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
                self.simworld.get_game_state()).numpy()[0]

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
            node.visit += 1
            # which action/child we climbed up from <int>
            a = node.children.index(node_child)
            # Upd N(s,a), E, Q(s, a)
            # E_a is accumulated reward for action (over all searches in this tree)
            node.action_visit[a] += 1
            node.action_cumreward[a] += z
            node.action_value[a] = node.action_cumreward[a] / \
                node.action_visit[a]

            if node.parent:
                climb_and_update(node.parent, node)
        if leaf_node.parent:
            climb_and_update(leaf_node.parent, leaf_node)

    def make_node(self, state, parent_node):
        return Node(state, parent_node, self.num_childstates)

    def get_hashed_state(self, state):
        return tuple(state)

    # TODO: be ready to explain this on demo!
    def tree_select_move(self, node, num_child_states):
        if self.black_to_play(node):
            # Get the greedy best-action coice for player 1
            # the math here would be better to do in np, but need to be explicit to be able to cythonize
            values = []
            # TODO CYT: not sure how math.inf translates to Cython
            for a in range(num_child_states):
                if not node.children[a]:
                    values.append(-math.inf)
                    continue
                values.append(node.action_value[a] + self.explore_constant *
                              math.sqrt(math.log(node.visit) / node.action_visit[a]))
            action_chosen = values.index(max(values))
        else:
            # get best-action for player 0
            values = []
            for a in range(num_child_states):
                if not node.children[a]:
                    values.append(math.inf)
                    continue
                values.append(node.action_value[a] - self.explore_constant *
                              math.sqrt(math.log(node.visit) / node.action_visit[a]))
            action_chosen = values.index(min(values))
        return action_chosen

    def black_to_play(self, node):
        # Differences between player ID [0, 1] and [1, 0]
        if node.state[0] == 0:
            return True
        return False

    def prune_tree(self, root_state):
        hash_state = self.get_hashed_state(root_state)
        # This is our new root_node, we want to prune all that exists above this
        new_root_node = self.tree[hash_state]

        def dive_and_burn(node):
            if not node.gameover:
                for child in range(len(node.children)):
                    # burning all legal children
                    if node.children[child]:
                        dive_and_burn(node.children[child])
            self.tree.pop(self.get_hashed_state(node.state))
            node.children = None

        # detach this new root_node from it's parent
        a = new_root_node.parent.children.index(new_root_node)
        new_root_node.parent.children[a] = None

        real_root = new_root_node.parent
        while real_root.parent:
            real_root = real_root.parent
        dive_and_burn(real_root)

        new_root_node.parent = None
        return new_root_node


class Node():
    '''
    node_representation:
        state = state
        parent = parent node
        children = children (fixed size array)
        action_value = monte-carlo value for action a(outgoing)
        action_visit = visit count for action a(outgoing)
        action_cumreward = sum of all final - state rewards this SAP has been involved in (so far)(outgoing)
        visit = visit count for this state(node)
    '''

    def __init__(self, state, parent_node, num_childstates):
        self.state = state
        self.parent = parent_node
        # TODO CYT: this becomes a fixed size array of Node|None when then node is expanded
        self.children = None

        # TODO CYT: arrays<float> filled with zero need to be initilized with a range that sets all spots to 0 think (look it up)
        self.action_value = [0] * num_childstates
        self.action_visit = [0] * num_childstates
        self.action_cumreward = [0] * num_childstates
        self.visit = [0] * num_childstates
        self.gameover = False
