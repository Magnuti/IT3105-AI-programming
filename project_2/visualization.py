import matplotlib.pyplot as plt
import networkx as nx

from constants import BoardCell

node_fillcolors = {
    BoardCell.EMPTY_CELL: "white",
    BoardCell.PLAYER_0_CELL_PART_OF_WINNING_PATH: "red",
    BoardCell.PLAYER_0_CELL: "red",
    BoardCell.PLAYER_1_CELL_PART_OF_WINNING_PATH: "black",
    BoardCell.PLAYER_1_CELL: "black"
}
node_edgecolors = {
    BoardCell.EMPTY_CELL: "black",
    BoardCell.PLAYER_0_CELL_PART_OF_WINNING_PATH: "green",
    BoardCell.PLAYER_0_CELL: "red",
    BoardCell.PLAYER_1_CELL_PART_OF_WINNING_PATH: "green",
    BoardCell.PLAYER_1_CELL: "black"
}


def visualize_board(graph, state_status_list, episode):
    """
    Args:
        graph: networkx graph
        state_status_list: list of BoardCell enums
    """

    fillcolor_map = list(map(lambda x: node_fillcolors[x], state_status_list))
    edgecolor_map = list(map(lambda x: node_edgecolors[x], state_status_list))

    plt.clf()
    # plt.title(f'Episode {episode}')
    nx.draw(graph, pos=graph.graph['plot_pos_dict'],
            with_labels=False, node_color=fillcolor_map, edgecolors=edgecolor_map, linewidths=3.0)
    plt.show(block=False)
    plt.pause(0.001)
