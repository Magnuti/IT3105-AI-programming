import numpy as np
import random
import math
import time


range_36 = range(36)


class MonteCarloTreeSearch:
    '''
    tree_representation
        {'state': node}

    '''

    def __init__(self, simworld, ANET, args):
        '''
        args:
            simworld:  actual_simworld, which state is reset at end of search_next_actual_move()
        '''

        self.explore_constant = args.explore_constant
        self.simworld = simworld
        self.simulations = args.simulations
        self.ANET = ANET
        self.num_childstates = args.neurons_per_layer[len(
            args.neurons_per_layer) - 1]

        self.start_new_game()
        self.args = args

    def start_new_game(self):
        self.root = None
        self.tree = {}

    def search_next_actual_move(self, epsilon):
        """
        Return
            next_state
            train_case: tuple with root state and target distribution which is
            to be added into the replay buffer.
        """

        start = time.time()

        # Set root + prune
        if not self.root:
            # This only happens at beginning of an episode, when the tree is empty and the root is not set
            self.root = self.make_node(self.simworld.get_game_state())
            self.tree[self.get_hashed_state(self.root.state)] = self.root
        else:
            self.root = self.tree[self.get_hashed_state(
                self.simworld.get_game_state())]
            # Remove unused states from self.tree (prune)
            self.prune_tree()

        # Run simulations
        for _ in range(self.simulations):
            if time.time() - start > self.args.sim_timelimit:
                break
            leaf_node, tree_search_path = self.tree_search()
            self.simworld.pick_move(leaf_node.state)
            z = self.leaf_eval(epsilon)
            if len(tree_search_path):
                self.backprop(leaf_node, z, tree_search_path)

            # Reset simworld to the root state before the simulation started
            self.simworld.pick_move(self.root.state)

        # Based on all the simulations, choose recommended actual_move
        # which is the one with the highest visit count
        move_num = np.argmax(self.root.action_visit)

        # Make training case for root_node
        target_dist = np.array(self.root.action_visit, dtype=float)
        target_norm = np.sum(target_dist)
        target_dist /= target_norm

        # print('Simulations took: ', time.time() - start)

        return self.root.children[move_num].state, (self.root.state, target_dist)

    # using UCT algorithm
    def tree_search(self):
        previous_node = None
        # This only gets set to False if a gameover-node that already exists in the tree is picked.
        # This pre-existing gameover-node is expanded already, but no children are attached to it
        continue_search = True
        tree_search_path = []
        while continue_search:
            state = self.simworld.get_game_state()
            hash_state = self.get_hashed_state(state)

            node = self.tree[hash_state]

            # if node is leaf (means unexpanded (which DOESN'T inform anything about it being a gameover))
            if not node.children:
                # find child states and insert into tree
                self.node_expand(node)
                # return the original leaf-node
                return node, tree_search_path

            tree_search_path.append(node)

            previous_node = node
            if not node.gameover:
                a = self.tree_select_move(node, len(node.children))
                self.simworld.pick_move(node.children[a].state)
            else:
                continue_search = False

        # Do one last evaluation (rollout/critic) of the parent of the game_over_state,
        # this way we may discover a better move than tree_policy chose
        # also remove the gameover_node from tree_search_path
        tree_search_path.pop(len(tree_search_path) - 1)
        return previous_node, tree_search_path

    def tree_select_move(self, node, num_child_states):
        if self.black_to_play(node):
            # Get the best-action choice for player 1
            values = []
            for a in range(num_child_states):
                if node.children[a] is None:
                    values.append(-math.inf)
                    continue

                # gives higher explore bonus if the action has less action_visit
                utc = self.explore_constant * \
                    math.sqrt(math.log(node.visit) /
                              (1 + node.action_visit[a]))
                values.append(node.action_value[a] + utc)
            action_chosen = values.index(max(values))
        else:
            # get best-action for player 0
            values = []
            for a in range(num_child_states):
                if node.children[a] is None:
                    values.append(math.inf)
                    continue

                # gives higher explore bonus if the action has less action_visit
                utc = self.explore_constant * \
                    math.sqrt(math.log(node.visit) /
                              (1 + node.action_visit[a]))
                values.append(node.action_value[a] - utc)
            action_chosen = values.index(min(values))

        return action_chosen

    def node_expand(self, parent_node):
        """
        Expands the children of a given node.
        """
        self.simworld.pick_move(parent_node.state)
        if self.simworld.get_gameover_and_reward()[0]:
            parent_node.gameover = True
            # No need to find the children of a game over state
            return

        child_states = self.simworld.get_child_states()

        # TODO CYT: type is [Node|None], this is a fixed size array
        children = [None]*self.num_childstates
        for i in range(len(child_states)):
            # this mean the child_state is legal (for illegal children, the related position in children[] is already None)
            if child_states[i] is not None:
                hash_state = self.get_hashed_state(child_states[i])
                if hash_state in self.tree:
                    node = self.tree[hash_state]
                else:
                    # attach child_node to our lookup-tree if not exists
                    node = self.make_node(child_states[i])
                    self.tree[hash_state] = node
                # attach child to it's parent
                children[i] = node

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

            child_states = self.simworld.get_child_states()

            if random.random() < epsilon:
                if self.args.rollout_explore == 0:
                    # Make completely random choice (including the best action)
                    legal_child_states = []
                    for state in child_states:
                        if state is not None:
                            legal_child_states.append(state)
                    choice = random.choice(legal_child_states)
                    self.simworld.pick_move(choice)
                else:
                    # Weighted choice
                    output_propabilities = self.ANET.forward(
                        self.simworld.get_game_state()).numpy()[0]
                    # Set illegal actions to 0 probability
                    for i, state in enumerate(child_states):
                        if state is None:
                            output_propabilities[i] = 0.0
                    # Normalize the new probabilities
                    output_propabilities /= sum(output_propabilities)

                    # TODO: hardcoded for size 6 board
                    move_index = random.choices(
                        range_36, weights=output_propabilities, k=1)[0]
                    self.simworld.pick_move(child_states[move_index])

            else:
                # Make greedy choice

                # Batch size is 1 so we get the output by indexing [0]
                output_propabilities = self.ANET.forward(
                    self.simworld.get_game_state()).numpy()[0]
                # Set illegal actions to 0 probability
                for i, state in enumerate(child_states):
                    if state is None:
                        output_propabilities[i] = 0.0

                # Normalize the new probabilities
                output_propabilities /= sum(output_propabilities)
                move_index = np.argmax(output_propabilities)
                self.simworld.pick_move(child_states[move_index])

            gameover, reward = self.simworld.get_gameover_and_reward()

        return reward

    def backprop(self, leaf_node, z, tree_search_path):
        child_node = leaf_node
        for parent_node in reversed(tree_search_path):
            # Update N(s)
            parent_node.visit += 1

            a = parent_node.children.index(child_node)  # Action index

            # Update N(s,a), E(s, a), Q(s, a)
            # E_a is accumulated reward for action (over all searches in this tree)
            parent_node.action_visit[a] += 1
            parent_node.action_cumreward[a] += z
            # Nodes with high visits will have high cumreward. By dividing we try to answer: "how much did action a contribute?"
            parent_node.action_value[a] = parent_node.action_cumreward[a] / \
                parent_node.action_visit[a]

            child_node = parent_node

    def make_node(self, state):
        return Node(state, self.num_childstates)

    def get_hashed_state(self, state):
        return tuple(state)

    def black_to_play(self, node):
        # Differences between player ID [0, 1] and [1, 0]
        if node.state[0] == 0:
            return True
        return False

    def prune_tree(self):
        new_tree = {}
        new_tree[self.get_hashed_state(self.root.state)] = self.root

        def add_children(node):
            tree = {}
            if node.children:
                for child in node.children:
                    if child is not None:
                        tree[self.get_hashed_state(child.state)] = child
                        # Merge the two dicts
                        tree.update(add_children(child))
            return tree

        # Merge the two dicts
        new_tree.update(add_children(self.root))
        self.tree = new_tree


class Node():
    '''
    node_representation:
        state = state
        children = children (fixed size array)
        action_value = "Q(s,a)" - monte-carlo value for action a(outgoing)
        action_visit = "N(s,a)" - visit count for action a(outgoing)
        action_cumreward = "e_t" sum of all final - state rewards this SAP has been involved in (so far)(outgoing)
        visit = "N(s)" -visit count for this state(node)
    '''

    def __init__(self, state, num_childstates):
        self.state = state
        self.children = None
        self.action_value = [0] * num_childstates
        self.action_cumreward = [0] * num_childstates
        self.action_visit = [0] * num_childstates
        self.visit = 1
        self.gameover = False

    def __str__(self):
        x = type(self).__name__ + ": {\n"
        for key, value in self.__dict__.items():
            x += "\t{}: {}\n".format(key, value)
        x += "}"
        return x
