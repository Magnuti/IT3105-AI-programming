import numpy as np
import networkx as nx

from visualization import visualize_board
from constants import BoardCell

"""
Player 1 = [0, 1], player 0 = [1, 0]
Player 1 wants to maximize the reward, while player 0 wants to minimize.
"""


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
        state = [0, 1] if self.current_player else [1, 0]
        state.extend(self.state)
        return np.array(state, dtype=int)

    def get_child_states(self):
        raise NotImplementedError()

    def pick_move(self, next_state):
        self.state = next_state[2:]

    def next_player(self):
        self.current_player = 1 - self.current_player  # Flip between 0 and 1

    def get_gameover_and_reward(self):
        raise NotImplementedError()

    def print_current_game_state(self):
        game_state = self.get_game_state()
        print("Current player:", self.current_player)
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
        self.current_player = 1
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
                child_state = [0, 1] if self.current_player else [1, 0]
                child_state.extend(np.roll(self.state, -i))
                child_states.append(np.array(child_state, dtype=int))

        return child_states

    def get_gameover_and_reward(self):
        if self.__get_remaining_pieces() > 0:
            return False, 0

        reward = 1 if self.current_player else -1
        return True, reward


class SimWorldHex(SimWorldInterface):
    class Cell:
        """
        Used to build the graph of the game state which can be passed to the
        networkx library for visual representations.
        """

        def __str__(self):
            return 'Cell_' + str(self.pos)

        def __repr__(self):
            return 'Cell_' + str(self.pos)

        def __init__(self, status, pos):
            self.status = status
            self.neighbor_indices = []
            self.pos = pos

    def __init__(self, board_size):
        """
        Args
            board_size: int
        """
        self.board_size = board_size
        self.neighbor_indices_list, self.cells = self.__init_neighbor_indices(
            self.board_size)
        # Each player has a list of disjoint sets. When A and B are in the
        # same set we have a winner.
        super().__init__()
        # Player 0 is red (R1 and R2), while player 1 is black (B1 and B2)
        self.neighbor_sets = {0: [{"R1"}, {"R2"}], 1: [{"B1"}, {"B2"}]}
        self.graph = self.__init_graph(self.cells)

    def __init_neighbor_indices(self, board_size):
        cells = []

        neighbor_indices_list = []
        cell_count = board_size**2

        for i in range(cell_count):
            cell = self.Cell(status=BoardCell.EMPTY_CELL,
                             pos=self.__index_to_coordinate(i, board_size))
            current_row = cell.pos[0]
            neighbor_indices = []  # Holds R1, R2, L1, L2 as well

            # Top
            k = i - board_size
            if(k >= 0):
                neighbor_indices.append(k)
                cell.neighbor_indices.append(k)
            else:
                neighbor_indices.append("R1")
                cell.neighbor_indices.append(None)

            # Top-right
            k = i - board_size + 1
            if(k >= 0 and (k // board_size) == current_row - 1):
                neighbor_indices.append(k)
                cell.neighbor_indices.append(k)
            else:
                neighbor_indices.append(None)
                cell.neighbor_indices.append(None)

            # Left
            k = i - 1
            if(k >= 0 and (k // board_size) == current_row):
                neighbor_indices.append(k)
                cell.neighbor_indices.append(k)
            else:
                neighbor_indices.append("B1")
                cell.neighbor_indices.append(None)

            # Right
            k = i + 1
            if(k < cell_count and (k // board_size) == current_row):
                neighbor_indices.append(k)
                cell.neighbor_indices.append(k)
            else:
                neighbor_indices.append("B2")
                cell.neighbor_indices.append(None)

            # Bottom-left
            k = i + board_size - 1
            if(k < cell_count and (k // board_size) == current_row + 1):
                neighbor_indices.append(k)
                cell.neighbor_indices.append(k)
            else:
                neighbor_indices.append(None)
                cell.neighbor_indices.append(None)

            # Bottom
            k = i + board_size
            if(k < cell_count):
                neighbor_indices.append(k)
                cell.neighbor_indices.append(k)
            else:
                neighbor_indices.append("R2")
                cell.neighbor_indices.append(None)

            neighbor_indices_list.append(neighbor_indices)
            cells.append(cell)

        return neighbor_indices_list, cells

    def __init_graph(self, cells):
        G = nx.empty_graph(len(cells))

        # Add all edges and connect the Cell to the graph_node
        for i, cell in enumerate(cells):
            G.nodes[i]['data'] = cell
            for neighbor_index in cell.neighbor_indices:
                if neighbor_index is not None:
                    G.add_edge(i, neighbor_index)

        # Set node-positions for plotting
        plot_pos_dict = {}
        for node_key in G.nodes():
            cell = G.nodes[node_key]['data']
            # This line is inspired by the TA Mathias
            plot_pos_dict[node_key] = (-cell.pos[0] + cell.pos[1], -
                                       cell.pos[0] - cell.pos[1])

        # Store the plot_pos_dict on the graph-object
        G.graph['plot_pos_dict'] = plot_pos_dict

        return G

    def __index_to_coordinate(self, index, board_size):
        count = 0
        for y in range(board_size):
            for x in range(board_size):
                if index == count:
                    return (y, x)
                count += 1
        raise IndexError(
            f"index '{index}' is not within the bounds given by board_size")

    def reset_game(self):
        self.current_player = 1
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
            [0, 1]) if self.current_player else np.array([1, 0], dtype=int)
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

        to_merge = [{cell_index}]
        self.neighbor_sets[self.current_player].append(to_merge[0])
        for neighbour_index in self.neighbor_indices_list[cell_index]:
            for neighbor_set in self.neighbor_sets[self.current_player]:
                if neighbour_index in neighbor_set:
                    neighbor_set.add(cell_index)
                    to_merge.append(neighbor_set)

        new_set = set()
        for merge_set in to_merge:
            new_set = new_set.union(merge_set)

        for neighbor_set in to_merge:
            # The if statement avoids duplicate removes which happens if
            # the inserted cell has more than one neighbour.
            if neighbor_set in self.neighbor_sets[self.current_player]:
                self.neighbor_sets[self.current_player].remove(neighbor_set)

        self.neighbor_sets[self.current_player].append(new_set)

        super().pick_move(next_state)

    def get_gameover_and_reward(self):
        if self.current_player:
            for neighbor_set in self.neighbor_sets[self.current_player]:
                if "B1" in neighbor_set and "B2" in neighbor_set:
                    self.winner_set = neighbor_set
                    # Black wins, player 1
                    return True, 1
        else:
            for neighbor_set in self.neighbor_sets[self.current_player]:
                if "R1" in neighbor_set and "R2" in neighbor_set:
                    self.winner_set = neighbor_set
                    # Red wins, player 0
                    return True, -1

        return False, 0

    def update_graph_statuses(self):
        for i in range(0, len(self.state), 2):
            cell_index = i // 2
            cell_state = self.state[i: i + 2]
            if np.array_equal(cell_state, np.array([0, 0], dtype=int)):
                # Empty cell
                status = BoardCell.EMPTY_CELL
            elif np.array_equal(cell_state, np.array([1, 0], dtype=int)):
                # Player 0's cell
                if cell_index in self.winner_set:
                    status = BoardCell.PLAYER_0_CELL_PART_OF_WINNING_PATH
                else:
                    status = BoardCell.PLAYER_0_CELL
            elif np.array_equal(cell_state, np.array([0, 1], dtype=int)):
                # Player 1's cell
                if cell_index in self.winner_set:
                    status = BoardCell.PLAYER_1_CELL_PART_OF_WINNING_PATH
                else:
                    status = BoardCell.PLAYER_1_CELL
            else:
                raise ValueError("Illegal value in self.state", self.state)

            self.cells[cell_index].status = status


if __name__ == "__main__":
    # sim_world = SimWorldNim(10, 5)
    sim_world = SimWorldHex(10)

    gameover, reward = sim_world.get_gameover_and_reward()
    while not gameover:
        # sim_world.print_current_game_state()
        sim_world.next_player()
        child_states = sim_world.get_child_states()
        legal_child_states = []
        for i in range(len(child_states)):
            if child_states[i] is not None:
                legal_child_states.append(child_states[i])

        move_index = np.random.randint(0, len(legal_child_states))
        sim_world.pick_move(legal_child_states[move_index])
        gameover, reward = sim_world.get_gameover_and_reward()

    # sim_world.print_current_game_state()
    if reward == 1:
        print("Player 1 (black) wins")
    else:
        print("Player 0 (red) wins")

    sim_world.update_graph_statuses()
    visualize_board(sim_world.graph, list(
        map(lambda x: x.status, sim_world.cells)), 0)
