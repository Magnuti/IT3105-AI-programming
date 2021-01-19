from constants import *
from argument_parser import Arguments
from visualization import visualize_board


class SimWorld:
    def __init__(self, board_type, cell_positions, board_size):
        self.board_type = board_type
        self.board_size = board_size
        self.cell_positions = cell_positions

        self.__init_board(board_type, cell_positions, board_size)
        self.__init_neighbor_cells(board_type, board_size)

        # TODO check if init board has valid IN_PROGRESS status
        self.current_state_status = StateStatus.IN_PROGRESS

    def __init_board(self, board_type, cell_positions, board_size):
        if(board_type == BoardType.Triangle):
            self.current_state = self.__init_triangle_board(
                cell_positions, board_size)
        elif(board_type == BoardType.Diamond):
            self.current_state = self.__init_diamond_board(
                cell_positions, board_size)
        else:
            raise NotImplementedError()

    def __init_triangle_board(self, cell_positions, board_size):
        raise NotImplementedError()

    def __init_diamond_board(self, cell_positions, board_size):
        board = [1 for i in range(board_size**2)]
        for i in cell_positions:
            board[i] = 0
        return board

    def __init_neighbor_cells(self, board_type, board_size):
        if(board_type == BoardType.Triangle):
            self.neighbors_indices = self.__init_neighbor_cells_triangle(
                board_size)
        elif(board_type == BoardType.Diamond):
            self.neighbors_indices = self.__init_neighbor_cells_diamond(
                board_size)
        else:
            raise NotImplementedError()

    def __init_neighbor_cells_triangle(self, board_size):
        raise NotImplementedError()

    def __init_neighbor_cells_diamond(self, board_size):
        neighbors = {}
        cell_count = board_size**2

        for i in range(cell_count):
            neighbors[i] = []
            current_row = i // board_size

            # Top
            k = i - board_size
            if(k >= 0):
                neighbors[i].append(k)
            else:
                neighbors[i].append(None)

            # Top-right
            k = i - board_size + 1
            if(k >= 0 and (k // board_size) == current_row - 1):
                neighbors[i].append(k)
            else:
                neighbors[i].append(None)

            # Left
            k = i - 1
            if(k >= 0 and (k // board_size) == current_row):
                neighbors[i].append(k)
            else:
                neighbors[i].append(None)

            # Right
            k = i + 1
            if(k < cell_count and (k // board_size) == current_row):
                neighbors[i].append(k)
            else:
                neighbors[i].append(None)

            # Bottom-left
            k = i + board_size - 1
            if(k < cell_count and (k // board_size) == current_row + 1):
                neighbors[i].append(k)
            else:
                neighbors[i].append(None)

            # Bottom
            k = i + board_size
            if(k < cell_count):
                neighbors[i].append(k)
            else:
                neighbors[i].append(None)

        return neighbors

    def reset_board(self):
        self.__init_board(
            self.board_type, self.cell_positions, self.board_size)

    def pick_new_state(self, state):
        # ? Check if state is in child_states maybe for security
        self.current_state = state

    def get_reward_and_state_status(self):
        pegs = self.current_state.count(1)
        if(pegs == 1):
            return 1, StateStatus.SUCCESS_FINISH

        child_states, _ = self.find_child_states()  # TODO duplicate call somewhere
        if(len(child_states) == 0):
            return -1, StateStatus.INCOMPLETE_FINISH

        return 0, StateStatus.IN_PROGRESS

    def find_child_states(self):
        if(self.board_type == BoardType.Triangle):
            return self.__find_child_states_triangle()
        elif(self.board_type == BoardType.Diamond):
            return self.__find_child_states_diamond()
        else:
            raise NotImplementedError()

    def __find_child_states_triangle(self):
        raise NotImplementedError()

    def __find_child_states_diamond(self):
        child_states = []
        child_states_with_visualization = []
        for i, cell_index in enumerate(self.current_state):
            if(cell_index == BoardCell.FULL_CELL.value):
                for j, neighbor_cell_index in enumerate(self.neighbors_indices[i]):
                    if(neighbor_cell_index is None):
                        continue

                    next_nextbour_cell_index = self.neighbors_indices[neighbor_cell_index][j]
                    if(next_nextbour_cell_index is None):
                        continue

                    if(self.current_state[neighbor_cell_index] == BoardCell.FULL_CELL.value and self.current_state[next_nextbour_cell_index] == BoardCell.EMPTY_CELL.value):
                        new_board = self.current_state.copy()
                        new_board[i] = BoardCell.EMPTY_CELL.value
                        new_board[neighbor_cell_index] = BoardCell.EMPTY_CELL.value
                        new_board[next_nextbour_cell_index] = BoardCell.FULL_CELL.value
                        child_states.append(new_board)

                        new_board_with_visualization = self.current_state.copy()
                        new_board_with_visualization[i] = BoardCell.JUMPED_FROM_CELL.value
                        new_board_with_visualization[neighbor_cell_index] = BoardCell.PRUNED_CELL.value
                        new_board_with_visualization[next_nextbour_cell_index] = BoardCell.JUMPED_TO_CELL.value
                        child_states_with_visualization.append(
                            new_board_with_visualization)

        return child_states, child_states_with_visualization


if __name__ == "__main__":
    args = Arguments()
    args.parse_arguments()

    sim_world = SimWorld(args.board, args.cell_positions, args.board_size)
    visualize_board(sim_world.board_type, sim_world.current_state)
