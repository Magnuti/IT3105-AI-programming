from constants import *
from argument_parser import Arguments
from visualization import visualize_board
import networkx as nx


class Cell:
    def __str__(self):
        return 'Cell_' + str(self.index)

    def __repr__(self):
        return 'Cell_' + str(self.index)

    def __init__(self, status, pos):
        self.status = status
        self.neighbor_indexes = []
        self.pos = pos


class SimWorld:
    def __init__(self, board_type, open_cell_positions, board_size):
        self.board_type = board_type
        self.board_size = board_size
        self.open_cell_positions = open_cell_positions

        self.__init_neighbor_cells(board_type, board_size, open_cell_positions)
        self.__init_board(board_type, board_size)

        # TODO check if init board has valid IN_PROGRESS status
        self.current_state_status = StateStatus.IN_PROGRESS

    def __init_board(self, board_type, board_size):
        if(board_type == BoardType.Triangle):
            self.graph = self.__init_triangle_board(board_size)
        elif(board_type == BoardType.Diamond):
            self.graph = self.__init_diamond_board(board_size)
        else:
            raise NotImplementedError()

    def __init_triangle_board(self, board_size):

        raise NotImplementedError()

    def __init_diamond_board(self, board_size):
        G = nx.empty_graph(len(self.current_state))

        # # Add all edges
        for i, cell in enumerate(self.current_state):
            for j, neighbor_cell in enumerate(cell.neighbors):
                G[i].add_edge(G[i], G[neighbor_cell.index])

        # # # Set node-positions for plotting and add diagonal edges
        # pos_dict = {}
        # for node_key in G.nodes():
        #     # This line is inspired by Mathias.TA
        #     pos_dict[node_key] = (-node_key[0] + node_key[1], -
        #                           node_key[0] - node_key[1])
        #     # if node_key[0] != 0 and node_key[1] < (board_size - 1):
        #     #     G.add_edge(node_key, (node_key[0] - 1, node_key[1] + 1))

        # # # Storing the visualization positions on the graph-object
        # G.graph['pos_dict'] = pos_dict

        # # # Add Cells to the graph node's ['data']
        # for i, node_key in enumerate(G.nodes()):
        #     # status 0 for the cells that are open initially
        #     if i in open_cell_positions:
        #         status = 0
        #     else:
        #         status = 1
        #     _cell = Cell(index=i, status=status)
        #     G.nodes[node_key]['data'] = _cell

        return G

    def __init_neighbor_cells(self, board_type, board_size, open_cell_positions):
        # for node_key in self.current_state.nodes():
        #     this_cell = self.current_state.nodes[node_key]['data']
        #     neighbor_cells = []
        #     for node_key in self.current_state.adj[node_key]:
        #         neighbor_cells.append(
        #             self.current_state.nodes[node_key]['data'])
        #     this_cell.set_neighbors(neighbor_cells)
        if (board_type == BoardType.Triangle):
            self.neighbors_indices = self.__init_neighbor_cells_triangle(
                board_size, open_cell_positions)
        elif (board_type == BoardType.Diamond):
            self.current_state = self.__init_cells_diamond(
                board_size, open_cell_positions)
        else:
            raise NotImplementedError()

    def __init_neighbor_cells_triangle(self, board_size, open_cell_positions):
        raise NotImplementedError()

    def __init_cells_diamond(self, board_size, open_cell_positions):
        cells = []
        cell_count = board_size**2

        for i in range(cell_count):
            status = 0 if i in open_cell_positions else 1
            cell = Cell(status=status,
                        pos=self.index_to_coordinate_diamond(i, board_size))
            current_row = cell.pos[0]
            neighbor_indexes = []

            # Top
            k = i - board_size
            if(k >= 0):
                neighbor_indexes.append(k)
            else:
                neighbor_indexes.append(None)

            # Top-right
            k = i - board_size + 1
            if(k >= 0 and (k // board_size) == current_row - 1):
                neighbor_indexes.append(k)
            else:
                neighbor_indexes.append(None)

            # Left
            k = i - 1
            if(k >= 0 and (k // board_size) == current_row):
                neighbor_indexes.append(k)
            else:
                neighbor_indexes.append(None)

            # Right
            k = i + 1
            if(k < cell_count and (k // board_size) == current_row):
                neighbor_indexes.append(k)
            else:
                neighbor_indexes.append(None)

            # Bottom-left
            k = i + board_size - 1
            if(k < cell_count and (k // board_size) == current_row + 1):
                neighbor_indexes.append(k)
            else:
                neighbor_indexes.append(None)

            # Bottom
            k = i + board_size
            if(k < cell_count):
                neighbor_indexes.append(k)
            else:
                neighbor_indexes.append(None)

            cell.neighbor_indexes = neighbor_indexes
            cells.append(cell)
        return cells

    def index_to_coordinate_diamond(self, index, board_size):
        if index == 0:
            return (0, 0)
        count = 0
        for i in range(board_size):
            for j in range(board_size):
                count += 1
                if index == count:
                    return (i, j)
        raise IndexError(
            f"index '{index}' is not within the bounds given by board_size")

    def get_node_key_list(self):
        return list(self.current_state.nodes.keys())

    def get_cell_from_index(self, index):
        node_keys = self.get_node_key_list()
        return self.current_state.nodes[node_keys[index]]['data']

    def get_cells(self):
        # TODO doing this method for each move may be unnecessary, maybe just store the cells in the sim_world class?
        return list(nx.get_node_attributes(self.current_state, 'data').values())

    def reset_board(self):
        self.__init_board(
            self.board_type, self.open_cell_positions, self.board_size)

    # TODO
    def pick_new_state(self, state):
        # ? Check if state is in child_states maybe for security
        self.current_state = state

    # TODO
    def get_reward_and_state_status(self):
        # TODO: new get_pegs_func
        pegs = self.current_state.count(1)
        if(pegs == 1):
            # TODO experiment with different rewards
            return 100, StateStatus.SUCCESS_FINISH

        child_states, _ = self.find_child_states()  # TODO duplicate call somewhere
        if(len(child_states) == 0):
            return -1, StateStatus.INCOMPLETE_FINISH

        return 0, StateStatus.IN_PROGRESS

    # TODO:
    def get_remaining_pegs(self):
        return self.current_state.count(1)

    # TODO
    def find_child_states(self):
        if(self.board_type == BoardType.Triangle):
            return self.__find_child_states_triangle()
        elif(self.board_type == BoardType.Diamond):
            return self.__find_child_states_diamond()
        else:
            raise NotImplementedError()

    def __find_child_states_triangle(self):
        raise NotImplementedError()

    # TODO in general
    def __find_child_states_diamond(self):
        child_states = []
        child_states_with_visualization = []
        # TODO: node_key unused?
        for i, cell in enumerate(self.get_cells()):
            if (cell.status == BoardCell.FULL_CELL.value):
                # TODO: all "i" may need to be rewritten
                # TODO: Jonas continue from here
                for j, neighbor_cell in enumerate(cell.neighbors):

                    # if(neighbor_cell_index is None):
                    #     continue

                    # TODO: is this still needed?
                    next_neighbor_cell_index = neighbor_cell.neighbors[j]
                    # if(next_neighbor_cell_index is None):
                    # continue

                    if(neighbor_cell.status == BoardCell.FULL_CELL.value and self.current_state[next_neighbor_cell_index] == BoardCell.EMPTY_CELL.value):
                        new_board = self.current_state.copy()
                        new_board[i] = BoardCell.EMPTY_CELL.value
                        new_board[neighbor_cell_index] = BoardCell.EMPTY_CELL.value
                        new_board[next_neighbor_cell_index] = BoardCell.FULL_CELL.value
                        child_states.append(new_board)

                        new_board_with_visualization = self.current_state.copy()
                        new_board_with_visualization[i] = BoardCell.JUMPED_FROM_CELL.value
                        new_board_with_visualization[neighbor_cell_index] = BoardCell.PRUNED_CELL.value
                        new_board_with_visualization[next_neighbor_cell_index] = BoardCell.JUMPED_TO_CELL.value
                        child_states_with_visualization.append(
                            new_board_with_visualization)

        return child_states, child_states_with_visualization


if __name__ == "__main__":
    sim_world = SimWorld(
        BoardType.Diamond, [2], 3)
    # print(sim_world.get_cells()[4].neighbors)
    # visualize_board(sim_world.board_type, sim_world.current_state)
