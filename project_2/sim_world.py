import numpy as np
import networkx as nx

from visualization import visualize_board, keep_board_visualization_visible
from constants import BoardCell
from disjoint_set import DisjointSet

"""
Player 1 = [0, 1], player 0 = [1, 0]
Player 1 wants to maximize the reward, while player 0 wants to minimize.

The project specification uses player 1 and 2, so we just increment them.

For the Hex game, player 0 is red, while player 1 is black
"""


class SimWorldInterface:
    """
    This class houses all the game logic of a game.
    E.g., generating initial board states, successor board states, legal moves;
    and recognizing final states and the winning player
    """

    def reset_game(self, starting_player):
        self.current_player = self.player_id_to_array(starting_player)

    def get_game_state(self):
        """
        Returns the state of the board with the first two indices being player bits.
        """
        return np.concatenate((self.current_player, self.state))

    def get_child_states(self):
        raise NotImplementedError()

    def pick_move(self, next_player_and_next_state):
        self.current_player = next_player_and_next_state[:2]
        self.state = next_player_and_next_state[2:]

    def get_gameover_and_reward(self):
        raise NotImplementedError()

    def current_player_array_to_id(self):
        if np.array_equal(self.current_player, np.array([0, 1], dtype=int)):
            return 1
        elif np.array_equal(self.current_player, np.array([1, 0], dtype=int)):
            return 0
        else:
            raise ValueError("Illegal player:", self.current_player)

    # TODO some of these calls should be made into constant since they don't change
    def player_id_to_array(self, player_id):
        if player_id == 0:
            return np.array([1, 0], dtype=int)
        elif player_id == 1:
            return np.array([0, 1], dtype=int)
        else:
            raise ValueError("Illegal player:", player_id)

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

    def reset_game(self, starting_player):
        super().reset_game(starting_player)
        # +1 because we must include the zero piece
        self.state = np.zeros(self.N + 1, dtype=int)
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
                next_player = np.roll(self.current_player, 1)
                child_states.append(np.concatenate(
                    (next_player, np.roll(self.state, -i))))

        return child_states

    def get_gameover_and_reward(self):
        if self.__get_remaining_pieces() > 0:
            return False, 0

        # Reversed because we look at the previous player, not the current one
        if self.current_player_array_to_id() == 0:
            return True, 1
        elif self.current_player_array_to_id() == 1:
            return True, -1
        else:
            raise ValueError("Illegal player:", self.current_player)


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

    def reset_game(self, starting_player):
        super().reset_game(starting_player)
        # Each cell is represented as two bits [0, 0] = empty, [1, 0] = filled by
        # player 0, and [0, 1] = filled by player 1
        self.state = np.zeros(self.board_size ** 2 * 2, dtype=int)
        # TODO set all cell status to emtpy instead of rebuilding
        # TODO this will save time in the TOPP
        self.graph = self.__init_graph(self.cells)
        self.winner_set = set()

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

    def get_child_states(self):
        """
        Returns a list of size board_size ** 2 + 2 of all possible states where
            the child states are not None. The first two elements represent the
            player ID, while the next element pairs represent cell values.
            For example, [[100010], [101000], None, None] where None
            means that the action is not legal. We need to pass down None
            values because we need to scale these illegal action's
            probabilities down to 0.
            We see that we can place the value 10 in two locations: [2:4] and
            [4:6], since the current player is 10
        """
        child_states = []
        for i in range(0, len(self.state), 2):
            if np.array_equal(self.state[i: i + 2], np.array([0, 0], dtype=int)):
                # Empty cell
                # The format is [next_player...current_player...]
                # where we set current_player to some board cell, and the next
                # player to make a move is represented by the first two bits.
                child_state = np.empty(2 + len(self.state), dtype=int)
                # Next player
                child_state[:2] = np.roll(self.current_player, 1)
                child_state[2:] = self.state
                child_state[2 + i: 4 + i] = self.current_player
            elif np.array_equal(self.state[i: i + 2], self.player_id_to_array(0)):
                # Player 0's cell
                child_state = None
            elif np.array_equal(self.state[i: i + 2], self.player_id_to_array(1)):
                # Player 1's cell
                child_state = None
            else:
                raise ValueError("Illegal value in self.state", self.state)

            child_states.append(child_state)

        return child_states

    def get_gameover_and_reward(self):
        # Player 0 is red (R1 and R2), while player 1 is black (B1 and B2)
        player_0_cells = {"R1", "R2"}
        player_1_cells = {"B1", "B2"}

        for i in range(0, len(self.state), 2):
            cell_index = i // 2
            cell_state = self.state[i: i + 2]

            if np.array_equal(cell_state, np.array([0, 0], dtype=int)):
                # Empty cell
                continue
            elif np.array_equal(cell_state, self.player_id_to_array(0)):
                # Player 0's cell
                player_0_cells.add(cell_index)
            elif np.array_equal(cell_state, self.player_id_to_array(1)):
                # Player 1's cell
                player_1_cells.add(cell_index)
            else:
                raise ValueError("Illegal value in self.state", self.state)

        disjoint_set_player_0 = DisjointSet(player_0_cells)
        disjoint_set_player_1 = DisjointSet(player_1_cells)

        for cell in player_0_cells:
            if cell == "R1" or cell == "R2":
                continue
            for neighbor_cell in self.neighbor_indices_list[cell]:
                if neighbor_cell in player_0_cells:
                    disjoint_set_player_0.union(cell, neighbor_cell)

        game_over = False
        reward = 0

        # Find out if they are in the same set
        if disjoint_set_player_0.find("R1") == disjoint_set_player_0.find("R2"):
            # Red wins, player 0
            self.winner_set = player_0_cells
            game_over = True
            reward = -1

        for cell in player_1_cells:
            if cell == "B1" or cell == "B2":
                continue
            for neighbor_cell in self.neighbor_indices_list[cell]:
                if neighbor_cell in player_1_cells:
                    disjoint_set_player_1.union(cell, neighbor_cell)

        # Find out if they are in the same set
        if disjoint_set_player_1.find("B1") == disjoint_set_player_1.find("B2"):
            # Black wins, player 1
            self.winner_set = player_1_cells
            game_over = True
            reward = 1

        # # TODO only do this if visualize is set to true or something like that
        self.__update_graph_statuses(game_over)

        return game_over, reward

    def __update_graph_statuses(self, game_over=False):
        for i in range(0, len(self.state), 2):
            cell_index = i // 2
            cell_state = self.state[i: i + 2]
            if np.array_equal(cell_state, np.array([0, 0], dtype=int)):
                # Empty cell
                status = BoardCell.EMPTY_CELL

            elif np.array_equal(cell_state, self.player_id_to_array(0)):
                # Player 0's cell
                if cell_index in self.winner_set and game_over:
                    status = BoardCell.PLAYER_0_CELL_PART_OF_WINNING_PATH
                else:
                    status = BoardCell.PLAYER_0_CELL
            elif np.array_equal(cell_state, self.player_id_to_array(1)):
                # Player 1's cell
                if cell_index in self.winner_set and game_over:
                    status = BoardCell.PLAYER_1_CELL_PART_OF_WINNING_PATH
                else:
                    status = BoardCell.PLAYER_1_CELL
            else:
                raise ValueError("Illegal value in self.state", self.state)

            self.cells[cell_index].status = status


if __name__ == "__main__":
    # sim_world = SimWorldNim(10, 5)
    sim_world = SimWorldHex(10)
    sim_world.reset_game(0)

    gameover, reward = sim_world.get_gameover_and_reward()
    while not gameover:
        # sim_world.print_current_game_state()
        child_states = sim_world.get_child_states()
        legal_child_states = []
        for i in range(len(child_states)):
            if child_states[i] is not None:
                legal_child_states.append(child_states[i])

        # For simplicity we just select a random legal action here
        legal_action_index = np.random.randint(0, len(legal_child_states))
        next_state = legal_child_states[legal_action_index]

        sim_world.pick_move(next_state)
        gameover, reward = sim_world.get_gameover_and_reward()

        visualize_board(sim_world.graph, list(
            map(lambda x: x.status, sim_world.cells)), 0)

    # sim_world.print_current_game_state()
    if reward == 1:
        print("Black (player 1, player 2 in project spec) wins")
    else:
        print("Red (player 0, player 1 in project spec) wins")

    visualize_board(sim_world.graph, list(
        map(lambda x: x.status, sim_world.cells)), 0)
    keep_board_visualization_visible()
