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


def visualize_board_manually(graph_list, state_status_list_list):
    """
    This function makes it possible to navigate through the moves in the game
    with the use of the arrow keys on the keyboard.
    """

    assert len(graph_list) == len(state_status_list_list)

    currrent_position = 0

    # Inspired from https://stackoverflow.com/questions/18390461/scroll-backwards-and-forwards-through-matplotlib-plots
    def key_event(e):
        nonlocal currrent_position

        if e.key == "right":
            currrent_position += 1
        elif e.key == "left":
            currrent_position -= 1
        else:
            return

        currrent_position = currrent_position % len(graph_list)

        state_status_list = state_status_list_list[currrent_position]
        graph = graph_list[currrent_position]

        fillcolor_map = list(
            map(lambda x: node_fillcolors[x], state_status_list))
        edgecolor_map = list(
            map(lambda x: node_edgecolors[x], state_status_list))

        ax.cla()
        ax.set_title("Move number {}".format(currrent_position + 1))
        nx.draw(graph, pos=graph.graph['plot_pos_dict'],
                with_labels=False, node_color=fillcolor_map, edgecolors=edgecolor_map, linewidths=3.0)
        fig.canvas.draw()

    fig = plt.figure()
    fig.canvas.mpl_connect('key_press_event', key_event)
    ax = fig.add_subplot(111)

    # The first time we open it we need to manually set the plot
    state_status_list = state_status_list_list[0]
    graph = graph_list[0]
    fillcolor_map = list(map(lambda x: node_fillcolors[x], state_status_list))
    edgecolor_map = list(map(lambda x: node_edgecolors[x], state_status_list))
    ax.set_title("Move number 1")
    nx.draw(graph, pos=graph.graph['plot_pos_dict'], with_labels=False,
            node_color=fillcolor_map, edgecolors=edgecolor_map, linewidths=3.0)

    plt.show()
