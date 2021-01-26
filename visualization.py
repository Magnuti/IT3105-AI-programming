import matplotlib.pyplot as plt
from math import sqrt
import networkx as nx

from constants import *
import pprint


def plot_performance(remaining_pegs_list):
    plt.plot(remaining_pegs_list)
    plt.show()


def visualize_board(baord_type, board):
    if(baord_type == BoardType.Triangle):
        visualize_hex_triangle(board)
    elif(baord_type == BoardType.Diamond):
        visualize_hex_diamond(board)
    else:
        raise NotImplementedError()


def visualize_hex_triangle(board):
    '''
    Visualize a left-shift triangle.
    First element is top row, 2nd and 3rd element is second row etc.

    ---
    Input
    ---
    1D array where the triangles height is
    '''
    raise NotImplementedError()


def visualize_hex_diamond(board):
    board_size = sqrt(len(board))
    if not board_size.is_integer():
        raise ValueError("Invalid board size")

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
            if(cell_value == BoardCell.EMPTY_CELL.value):
                x_open_cells.append(x)
                y_open_cells.append(y)
            elif(cell_value == BoardCell.FULL_CELL.value):
                x_full_cells.append(x)
                y_full_cells.append(y)
            elif(cell_value == BoardCell.PRUNED_CELL.value):
                x_pruned_cells.append(x)
                y_pruned_cells.append(y)
            elif(cell_value == BoardCell.JUMPED_FROM_CELL.value):
                x_jumped_from_cells.append(x)
                y_jumped_from_cells.append(y)
            elif(cell_value == BoardCell.JUMPED_TO_CELL.value):
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


if __name__ == '__main__':
    from visualization import Cell

    board_size = 4
    G = nx.grid_2d_graph(board_size, board_size)
    pos = {}
    # Set node-position on plot and add diagonal edges
    for node_key in G.nodes():
        # This line is inspired by Mathias.TA
        pos[node_key] = (-node_key[0] + node_key[1], -
                         node_key[0] - node_key[1])
        if node_key[0] != 0 and node_key[1] < (board_size - 1):
            # print('Adding edge', node_key, (node_key[0]+1, node_key[1]+1))
            G.add_edge(node_key, (node_key[0] - 1, node_key[1] + 1))

    # Storing the visualization positions on the graph-object
    G.graph['pos_dict'] = pos

    board = []
    for i, node_key in enumerate(G.nodes()):
        # status 0 for the cells that are open initially
        # if i in [10]:
        if i in open_cell_positions:
            status = 0
        else:
            status = 1
        # print(pos[node_key])
        _cell = Cell(index=i, status=status)
        board.append(_cell)
        G.nodes[node_key]['data'] = _cell

    nx.draw(G, pos=G.graph['pos_dict'], with_labels=True, font_weight='bold')
    plt.show()

    for _cell in G.nodes():
        print(_cell, ': ', G.nodes[_cell]['data'].status)
