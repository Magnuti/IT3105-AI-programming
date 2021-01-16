import matplotlib.pyplot as plt

from argument_parser import Arguments


def dict_print(d):
    for key, value in d.items():
        print("{}: {}".format(key, value))


def visualize_hex_triangle(cell_positions, board_size):
    '''
    Visualize a left-shift triangle.
    First element is top row, 2nd and 3rd element is second row etc.

    ---
    Input
    ---
    1D array where the triangles height is
    '''
    pass


def visualize_hex_diamond(cell_positions, board_size):
    x = 0
    y = 0
    x_coordinates = []
    y_coordinates = []
    x_coordinates_open_cells = []
    y_coordinates_open_cells = []

    cell_counter = 0
    for i in range(board_size):
        x = i * -0.5
        y = -i
        for j in range(board_size):
            x += 0.5
            y -= 1
            if(cell_counter in cell_positions):
                x_coordinates_open_cells.append(x)
                y_coordinates_open_cells.append(y)
            else:
                x_coordinates.append(x)
                y_coordinates.append(y)

            cell_counter += 1

    plt.plot(x_coordinates, y_coordinates, 'o', color="black")
    plt.plot(x_coordinates_open_cells,
             y_coordinates_open_cells, 'o', markeredgecolor="black", color="white")
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


if __name__ == "__main__":
    arguments = Arguments()
    arguments.parse_arguments()
    if(arguments.board == "triangle"):
        visualize_hex_triangle(arguments.cell_positions, arguments.board_size)
    else:
        visualize_hex_diamond(arguments.cell_positions, arguments.board_size)
        neighbors = find_neighbor_cells_diamond(arguments.board_size)
        dict_print(neighbors)
