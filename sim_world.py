from constants import *
from argument_parser import Arguments
from visualization import visualize_board
import networkx as nx
import copy


class Cell:
    def __str__(self):
        return 'Cell_' + str(self.pos)

    def __repr__(self):
        return 'Cell_' + str(self.pos)

    def __init__(self, status, pos):
        self.status = status
        self.neighbor_indices = []
        self.pos = pos


class SimWorld:
    def __init__(self, board_type, open_cell_positions, board_size):
        self.board_type = board_type
        self.board_size = board_size
        self.open_cell_positions = open_cell_positions

        # A list of cells
        self.current_state = self.__init_neighbor_cells(
            board_type, board_size, open_cell_positions)
        self.__init_board(board_type, board_size)

        # TODO check if init board has valid IN_PROGRESS status
        self.current_state_status = StateStatus.IN_PROGRESS

    def __init_board(self, board_type, board_size):
        if(board_type == BoardType.Triangle):
            self.graph = self.__init_triangle_board(board_size)
        elif (board_type == BoardType.Diamond):
            self.graph = self.__init_diamond_board(board_size)
        else:
            raise NotImplementedError()

    def __init_triangle_board(self, board_size):
        G = nx.empty_graph(int((board_size * (board_size + 1)) / 2))
        G = self.graph_add_connections(G)

        # Set node-positions for plotting
        plot_pos_dict = {}
        for node_key in G.nodes():
            cell = G.nodes[node_key]['data']
            # This line is inspired by the TA Mathias
            plot_pos_dict[node_key] = (-10 * cell.pos[0] + 20 * cell.pos[1], - 10 *
                                       cell.pos[0])

        # Store the plot_pos_dict on the graph-object
        G.graph['plot_pos_dict'] = plot_pos_dict

        return G

    def __init_diamond_board(self, board_size):
        G = nx.empty_graph(len(self.current_state))
        G = self.graph_add_connections(G)

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

    def __init_neighbor_cells(self, board_type, board_size, open_cell_positions):
        if (board_type == BoardType.Triangle):
            return self.__init_neighbor_cells_triangle(
                board_size, open_cell_positions)
        elif (board_type == BoardType.Diamond):
            return self.__init_cells_diamond(
                board_size, open_cell_positions)
        else:
            raise NotImplementedError()

    def __init_neighbor_cells_triangle(self, board_size, open_cell_positions):
        cells = []

        # https://en.wikipedia.org/wiki/Triangular_number
        cell_count = int((board_size * (board_size + 1)) / 2)

        current_row_width = 1
        y = 0
        x = 0
        for i in range(cell_count):
            if(x >= current_row_width):
                y += 1
                x = 0
                current_row_width += 1

            status = 0 if i in open_cell_positions else 1
            cell = Cell(status=status, pos=(y, x))

            # Top-left
            if(x > 0 and y > 0):
                cell.neighbor_indices.append(i - current_row_width)
            else:
                cell.neighbor_indices.append(None)

            # Top
            if(x < current_row_width - 1 and y > 0):
                cell.neighbor_indices.append(i - (current_row_width - 1))
            else:
                cell.neighbor_indices.append(None)

            # Left
            if(x > 0):
                cell.neighbor_indices.append(i - 1)
            else:
                cell.neighbor_indices.append(None)

            # Right
            if(x < current_row_width - 1):
                cell.neighbor_indices.append(i + 1)
            else:
                cell.neighbor_indices.append(None)

            # Bottom and bottom-right
            if(y < board_size - 1):
                cell.neighbor_indices.append(i + current_row_width)
                cell.neighbor_indices.append(i + current_row_width + 1)
            else:
                cell.neighbor_indices.append(None)
                cell.neighbor_indices.append(None)

            x += 1
            cells.append(cell)

        return cells

    def __init_cells_diamond(self, board_size, open_cell_positions):
        cells = []
        cell_count = board_size**2

        for i in range(cell_count):
            status = 0 if i in open_cell_positions else 1
            cell = Cell(status=status,
                        pos=self.index_to_coordinate_diamond(i, board_size))
            current_row = cell.pos[0]
            neighbor_indices = []

            # Top
            k = i - board_size
            if(k >= 0):
                neighbor_indices.append(k)
            else:
                neighbor_indices.append(None)

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
                neighbor_indices.append(None)

            # Right
            k = i + 1
            if(k < cell_count and (k // board_size) == current_row):
                neighbor_indices.append(k)
            else:
                neighbor_indices.append(None)

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
                neighbor_indices.append(None)

            cell.neighbor_indices = neighbor_indices
            cells.append(cell)
        return cells

    def index_to_coordinate_diamond(self, index, board_size):
        count = 0
        for y in range(board_size):
            for x in range(board_size):
                if index == count:
                    return (y, x)
                count += 1
        raise IndexError(
            f"index '{index}' is not within the bounds given by board_size")

    # TODO must be changed if state repr. changes to [status]
    # Add all edges and connect the Cell to the graph_node
    def graph_add_connections(self, G):
        for i, cell in enumerate(self.current_state):
            G.nodes[i]['data'] = cell
            for neighbor_index in cell.neighbor_indices:
                if neighbor_index is not None:
                    G.add_edge(i, neighbor_index)
        return G

    # TODO: remove?
    # def get_node_key_list(self):
    #     return list(self.current_state.nodes.keys())

    # TODO: remove?
    # def get_cell_from_index(self, index):
    #     node_keys = self.get_node_key_list()
    #     return self.current_state.nodes[node_keys[index]]['data']

    def reset_board(self):
        self.current_state = self.__init_neighbor_cells(
            self.board_type, self.board_size, self.open_cell_positions)
        self.__init_board(
            self.board_type, self.board_size)

    def pick_new_state(self, new_state):
        # TODO Check if state is in child_states maybe for security
        for i, cell in enumerate(self.current_state):
            cell.status = new_state[i]

    # this is directly called from rl_agent.py
    def get_current_state_statuses(self):
        return list(map(lambda x: x.status, self.current_state))

    def get_reward_and_state_status(self, child_states_length):
        pegs = self.get_remaining_pegs()
        if(pegs == 1):
            # TODO experiment with different rewards
            return 100, StateStatus.SUCCESS_FINISH

        # If no child_states, then this is an end state
        if(child_states_length == 0):
            return -1, StateStatus.INCOMPLETE_FINISH

        return 0, StateStatus.IN_PROGRESS

    def get_remaining_pegs(self):
        # Used if state is statuses:
        # return self.current_state.count(1)
        statuses = self.get_current_state_statuses()
        return statuses.count(1)

    def find_child_states(self):
        child_states = []
        child_states_with_visualization = []
        for i, cell in enumerate(self.current_state):
            if (cell.status == BoardCell.FULL_CELL.value):
                for j, neighbor_index in enumerate(cell.neighbor_indices):

                    if(neighbor_index is None):
                        # outside of the board
                        continue

                    neighbor_cell = self.current_state[neighbor_index]
                    if neighbor_cell.status != BoardCell.FULL_CELL.value:
                        # if this neighbor_cell does not contain Peg, then no child state
                        continue

                    # index of next cell in the same direction as neighbor_index
                    next_neighbor_index = self.current_state[neighbor_index].neighbor_indices[j]
                    if (next_neighbor_index is None):
                        # outside of the board
                        continue

                    if (self.current_state[next_neighbor_index].status == BoardCell.EMPTY_CELL.value):
                        # the state that gets delivered to the RL-agent is represented as [cell.status]

                        current_state = list(map(
                            lambda x: x.status, self.current_state))

                        new_state = current_state.copy()
                        new_state[i] = BoardCell.EMPTY_CELL.value
                        new_state[neighbor_index] = BoardCell.EMPTY_CELL.value
                        new_state[next_neighbor_index] = BoardCell.FULL_CELL.value
                        child_states.append(new_state)

                        new_state_with_visualization = current_state.copy()
                        new_state_with_visualization[i] = BoardCell.JUMPED_FROM_CELL.value
                        new_state_with_visualization[neighbor_index] = BoardCell.PRUNED_CELL.value
                        new_state_with_visualization[next_neighbor_index] = BoardCell.JUMPED_TO_CELL.value
                        child_states_with_visualization.append(
                            new_state_with_visualization)

        return child_states, child_states_with_visualization
