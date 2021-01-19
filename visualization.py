import matplotlib.pyplot as plt
from math import sqrt
import networkx as nx

from constants import *


class Cell:
    def __init__(self, status, position, index):
        self.status = status
        # self.position = position
        self.index = index
        self.neighbors = []

    def setNeighbors(self, CellList):
        self.neighbors = CellList


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

    # G = nx.Graph()
    # G.add_nodes_from([1, 2, 3, 4])
    # G.add_edge(1, 3)
    # G.add_edge(2, 4)
    # G.add_edge(3, 4)
    # nx.draw(G, with_labels=True, font_weight='bold')
    # plt.show()
