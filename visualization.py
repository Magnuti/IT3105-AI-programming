import matplotlib.pyplot as plt
from math import sqrt
import networkx as nx

from constants import *
# import pprint


def plot_performance(remaining_pegs_list):
    plt.plot(remaining_pegs_list)
    plt.show()


def visualize_board(graph, state):
    reassign_cells_to_graph(graph, state)

    # plt.plot(x_full_cells, y_full_cells, 'o', color="black")
    # plt.plot(x_open_cells, y_open_cells, 'o',
    #          markeredgecolor="black", markeredgewidth=2, color="white")
    # plt.plot(x_pruned_cells, y_pruned_cells, 'o',
    #          markeredgecolor="red", markeredgewidth=2,  color="white")
    # plt.plot(x_jumped_from_cells, y_jumped_from_cells, 'o',
    #          markeredgecolor="green", markeredgewidth=2, color="white")
    # plt.plot(x_jumped_to_cells, y_jumped_to_cells, 'o',
    #          markeredgecolor="green", markeredgewidth=2, color="black")

    nx.draw(graph, pos=graph.graph['plot_pos_dict'],
            with_labels=True, font_weight='bold')
    plt.show()


def reassign_cells_to_graph(graph, state):
    for i in range(len(state)):
        graph.nodes[i]['data'] = state[i]
