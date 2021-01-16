import matplotlib.pyplot as plt
from math import sqrt

from argument_parser import Arguments

EMPTY_CELL = 0
FULL_CELL = 1
PRUNED_CELL = 2
JUMPED_FROM_CELL = 3
JUMPED_TO_CELL = 4


def dict_print(d):
    for key, value in d.items():
        print("{}: {}".format(key, value))


def init_board(cell_positions, board_size):
    board = [1 for i in range(board_size**2)]
    for i in cell_positions:
        board[i] = 0
    return board


def visualize_hex_triangle(board):
    '''
    Visualize a left-shift triangle.
    First element is top row, 2nd and 3rd element is second row etc.

    ---
    Input
    ---
    1D array where the triangles height is
    '''
    pass


def visualize_hex_diamond(board):
    board_size = sqrt(len(board))
    if not board_size.is_integer():
        print("Invalid board size")

    board_size = int(board_size)

    x_full_cells = []
    y_full_cells = []
    x_open_cells = []
    y_open_cells = []
    x_pruned_cells = []
    y_pruned_cells = []
    x_jumped_from_cells = []
    y_jumped_from_cells = []
    x_jumped_to_cells = []
    y_jumped_to_cells = []

    x = 0
    y = 0
    for i in range(board_size):
        x = i * -0.5
        y = -i
        for j in range(board_size):
            x += 0.5
            y -= 1
            cell_value = board[i * board_size + j]
            if(cell_value == EMPTY_CELL):
                x_open_cells.append(x)
                y_open_cells.append(y)
            elif(cell_value == FULL_CELL):
                x_full_cells.append(x)
                y_full_cells.append(y)
            elif(cell_value == PRUNED_CELL):
                x_pruned_cells.append(x)
                y_pruned_cells.append(y)
            elif(cell_value == JUMPED_FROM_CELL):
                x_jumped_from_cells.append(x)
                y_jumped_from_cells.append(y)
            elif(cell_value == JUMPED_TO_CELL):
                x_jumped_to_cells.append(x)
                y_jumped_to_cells.append(y)
            else:
                print("Unknown board cell")

    plt.plot(x_full_cells, y_full_cells, 'o', color="black")
    plt.plot(x_open_cells, y_open_cells, 'o',
             markeredgecolor="black", markeredgewidth=2, color="white")
    plt.plot(x_pruned_cells, y_pruned_cells, 'o',
             markeredgecolor="red", markeredgewidth=2,  color="white")
    plt.plot(x_jumped_from_cells, y_jumped_from_cells, 'o',
             markeredgecolor="green", markeredgewidth=2, color="white")
    plt.plot(x_jumped_to_cells, y_jumped_to_cells, 'o',
             markeredgecolor="green", markeredgewidth=2, color="black")

    plt.show()


def find_neighbor_cells_triangle(board_size):
    pass


def find_neighbor_cells_diamond(board_size):
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


def find_child_states_diamond(board, neighbors_indecies):
    child_boards = []
    for i, cell_index in enumerate(board):
        if(cell_index == FULL_CELL):
            for j, neighbor_cell_index in enumerate(neighbors_indecies[i]):
                if(neighbor_cell_index is None):
                    continue

                next_nextbour_cell_index = neighbors_indecies[neighbor_cell_index][j]
                if(next_nextbour_cell_index is None):
                    continue

                if(board[neighbor_cell_index] == FULL_CELL and board[next_nextbour_cell_index] == EMPTY_CELL):
                    new_board = board.copy()
                    new_board[i] = JUMPED_FROM_CELL
                    new_board[neighbor_cell_index] = PRUNED_CELL
                    new_board[next_nextbour_cell_index] = JUMPED_TO_CELL
                    child_boards.append(new_board)

    return child_boards


if __name__ == "__main__":
    arguments = Arguments()
    arguments.parse_arguments()

    board = init_board(arguments.cell_positions, arguments.board_size)

    if(arguments.board == "triangle"):
        # visualize_hex_triangle(board)
        pass
    else:
        visualize_hex_diamond(board)
        neighbors_indecies = find_neighbor_cells_diamond(arguments.board_size)
        child_boards = find_child_states_diamond(board, neighbors_indecies)
