import matplotlib.pyplot as plt
from math import sqrt
import networkx as nx
# import time

# from constants import *
# import pprint

node_fillcolors = ['white', 'black', 'white', 'white', '#79eb44']
node_edgecolors = ['black', 'black', 'red', '#79eb44', 'black']


def plot_performance(remaining_pegs_list):
    plt.clf()
    plt.plot(remaining_pegs_list)
    plt.show()


def visualize_board(graph, state, episode):
    fillcolor_map = list(map(lambda x: node_fillcolors[x], state))
    edgecolor_map = list(map(lambda x: node_edgecolors[x], state))

    plt.clf()
    plt.title(f'Episode {episode}')
    nx.draw(graph, pos=graph.graph['plot_pos_dict'],
            with_labels=False, node_color=fillcolor_map, edgecolors=edgecolor_map, linewidths=3.0)
    plt.show(block=False)
    plt.pause(0.001)
