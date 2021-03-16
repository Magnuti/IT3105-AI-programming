import numpy as np


class SimWorldInterface:
    """
    This class houses all the game logic of a game.
    E.g., generating initial board states, successor board states, legal moves;
    and recognizing final states and the winning player
    """

    def __init__(self):
        self.reset_game()

    def reset_game(self):
        raise NotImplementedError()

    def get_game_state(self):
        """
        Returns the state of the board with the first two indices being player bits.
        """
        # Player 1 = [0, 1], player 0 = [1, 0]
        state = [0, 1] if self.current_player else [1, 0]
        state.extend(self.state)
        return np.array(state, dtype=int)

    def get_child_states(self):
        raise NotImplementedError()

    def pick_move(self, next_state):
        self.state = next_state[2:]
        self.current_player = 1 - self.current_player  # Flip between 0 and 1

    def set_state_and_player(self, state):
        raise NotImplementedError()

    def get_gameover_and_reward(self):
        raise NotImplementedError()

    def print_current_game_state(self):
        game_state = self.get_game_state()
        # print("Current player:", game_state[0:2])
        print("Game state:\n\t", game_state)
        print("Child states:")
        for child_state in self.get_child_states():
            print("\t", child_state)
        gameover, reward = self.get_gameover_and_reward()
        print("Gameover:", gameover)
        print("Reward:", reward)
        print("")


class SimWorldNim(SimWorldInterface):
    def __init__(self, N, K):
        """
        Args
            N: int
                The number of pieces on the board
            K: int
                The maximum number that a player can take off the board on their turn.
                The minimum pieces to remove is always 1.
        """
        self.N = N
        self.K = K
        super().__init__()

    def reset_game(self):
        self.current_player = 0
        self.state = np.zeros(self.N, dtype=int)
        self.state[-1] = 1

    def __get_remaining_pieces(self):
        return np.where(self.state == 1)[0][0]

    def get_child_states(self):
        """
        Returns a list of size K of all possible states where the child states are not None
            For example, [[some state], [some state], None, None] where None means that the action
            is not legal. We need to pass down None values because we need to scale these illegal
            action's probabilities down to 0.
        """
        child_states = []
        for i in range(1, self.K + 1):
            if i > self.__get_remaining_pieces():
                child_states.append(None)
            else:
                child_state = [0, 1] if not self.current_player else [1, 0]
                child_state.extend(np.roll(self.state, -i))
                child_states.append(np.array(child_state, dtype=int))

        return child_states

    def set_state_and_player(self, state):
        """
        Used to reset the board to the state it was before the rollout simulation started.
        """
        self.current_player = state[:2]
        self.state = state[2:]

    def get_gameover_and_reward(self):
        if self.__get_remaining_pieces() > 0:
            return False, 0

        # Player 1 wants to maximize, while player 0 wants to minimize
        reward = 1 if self.current_player else -1
        return True, reward


class SimWorldHex(SimWorldInterface):
    def __init__(self, board_size):
        """
        Args
            board_size: int
        """
        self.board_size = board_size
        self.neighbor_indices_list = self.__init_neighbor_indices(
            self.board_size)
        # Each player has a list of disjoint sets. When A and B are in the
        # same set we have a winner.
        # self.neighbor_sets = {0: [{"A"}, {"B"}], 1: [{"A"}, {"B"}]}
        self.neighbor_sets = {0: [{"R1"}, {"R2"}], 1: [{"L1"}, {"L2"}]}
        super().__init__()

    def __init_neighbor_indices(self, board_size):
        neighbor_indices_list = []
        cell_count = board_size**2

        for i in range(cell_count):
            current_row = i // board_size
            neighbor_indices = []

            # Top
            k = i - board_size
            if(k >= 0):
                neighbor_indices.append(k)
            else:
                neighbor_indices.append("L1")

            # Top-right
            k = i - board_size + 1
            if(k >= 0 and (k // board_size) == current_row - 1):
                neighbor_indices.append(k)
            else:
                neighbor_indices.append(None)

            # Left
            k = i - 1
            if(k >= 0 and (k // board_size) == current_row):
                neighbor_indices.append(k)
            else:
                neighbor_indices.append("R1")

            # Right
            k = i + 1
            if(k < cell_count and (k // board_size) == current_row):
                neighbor_indices.append(k)
            else:
                neighbor_indices.append("R2")

            # Bottom-left
            k = i + board_size - 1
            if(k < cell_count and (k // board_size) == current_row + 1):
                neighbor_indices.append(k)
            else:
                neighbor_indices.append(None)

            # Bottom
            k = i + board_size
            if(k < cell_count):
                neighbor_indices.append(k)
            else:
                neighbor_indices.append("L2")

            neighbor_indices_list.append(neighbor_indices)

        return neighbor_indices_list

    def reset_game(self):
        self.current_player = 0
        # Each cell is represented as two bits [0, 0] = empty, [1, 0] = filled by
        # player 0, and [0, 1] = filled by player 1
        self.state = np.zeros(self.board_size ** 2 * 2, dtype=int)

    def get_child_states(self):
        """
        Returns a list of size baord_size of all possible states where the child states are not None
            For example, [[some state], [some state], None, None] where None means that the action
            is not legal. We need to pass down None values because we need to scale these illegal
            action's probabilities down to 0.
        """
        child_states = []
        player_cells = np.array(
            [0, 1]) if not self.current_player else np.array([1, 0], dtype=int)
        for i in range(0, len(self.state), 2):
            if np.array_equal(self.state[i: i + 2], np.array([0, 0], dtype=int)):
                # Empty cell
                child_state = np.empty(2 + len(self.state), dtype=int)
                child_state[:2] = player_cells
                child_state[2:] = self.state
                child_state[2 + i: 4 + i] = player_cells
            elif np.array_equal(self.state[i: i + 2], np.array([1, 0], dtype=int)):
                # Player 0's cell
                child_state = None
            elif np.array_equal(self.state[i: i + 2], np.array([0, 1], dtype=int)):
                # Player 1's cell
                child_state = None
            else:
                raise ValueError("Illegal value in self.state", self.state)

            child_states.append(child_state)

        return child_states

    def pick_move(self, next_state):
        for i in range(0, len(next_state[2:]), 2):
            if not np.array_equal(next_state[2+i:4+i], self.state[i: 2+i]):
                cell_index = i // 2
                break

        super().pick_move(next_state)
        player_cached = 1 - self.current_player  # Flip between 0 and 1

        to_merge = [{cell_index}]
        self.neighbor_sets[player_cached].append(to_merge[0])
        for neighbour_index in self.neighbor_indices_list[cell_index]:
            for neighbor_set in self.neighbor_sets[player_cached]:
                if neighbour_index in neighbor_set:
                    neighbor_set.add(cell_index)
                    to_merge.append(neighbor_set)

        new_set = set()
        for i in range(len(to_merge)):
            new_set = new_set.union(to_merge[i])

        for neighbor_set in to_merge:
            # The if statement avoids duplicate removes which happens if
            # the inserted cell has more than one neighbour.
            if neighbor_set in self.neighbor_sets[player_cached]:
                self.neighbor_sets[player_cached].remove(neighbor_set)

        self.neighbor_sets[player_cached].append(new_set)

    def set_state_and_player(self, state):
        raise NotImplementedError()

    def get_gameover_and_reward(self):
        player_cached = 1 - self.current_player  # Flip between 0 and 1

        print(self.neighbor_sets[player_cached])

        for neighbor_set in self.neighbor_sets[player_cached]:
            # TODO look over the rewards 1 vs -1 depending on which player is on which side
            if "R1" in neighbor_set and "R2" in neighbor_set:
                return True, 1
            elif "L1" in neighbor_set and "L2" in neighbor_set:
                return True, -1

        return False, 0


if __name__ == "__main__":
    # sim_world = SimWorldNim(10, 5)
    sim_world = SimWorldHex(3)

    gameover, reward = sim_world.get_gameover_and_reward()
    while not gameover:
        sim_world.print_current_game_state()
        child_states = sim_world.get_child_states()
        legal_child_states = []
        for i in range(len(child_states)):
            if child_states[i] is not None:
                legal_child_states.append(child_states[i])

        move_index = np.random.randint(0, len(legal_child_states))
        sim_world.pick_move(legal_child_states[move_index])
        gameover, reward = sim_world.get_gameover_and_reward()

    sim_world.print_current_game_state()
