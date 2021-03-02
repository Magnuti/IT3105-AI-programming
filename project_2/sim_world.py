import numpy as np


class SimWorldNim:
    """
    This class houses all the game logic of the game Nim.
    E.g., generating initial board states, successor board states, legal moves;
    and recognizing final states and the winning player
    """

    def __init__(self, N, K):
        """
        Args
            N: int
                The number of pieces on the board
            K: int
                The maximum number that a player can take off the board on their turn.
                The minimum pieces to remove is always 1.
        """
        self.N = N
        self.K = K
        self.reset_game()

    def reset_game(self):
        self.current_player = 0
        self.remaining_pieces = self.N

    def get_game_state(self):
        return [self.current_player, self.remaining_pieces]

    def get_child_states(self):
        child_states = []
        if self.remaining_pieces < self.K:
            for i in range(1, self.remaining_pieces + 1):
                child_states.append(self.remaining_pieces - i)
        else:
            for i in range(1, self.K + 1):
                child_states.append(self.remaining_pieces - i)

        return child_states

    def pick_move(self, next_state):
        self.remaining_pieces = next_state
        self.current_player = 1 - self.current_player  # Flip between 0 and 1

    def get_gameover_and_reward(self):
        if self.remaining_pieces > 0:
            return False, 0

        # Player 0 wants to
        reward = 1 if self.current_player else -1
        return True, reward

    def print_current_game_state(self):
        game_state = self.get_game_state()
        print("Current player:", game_state[0])
        print("Game state:", game_state[1:])
        print("Child states:", self.get_child_states())
        gameover, reward = self.get_gameover_and_reward()
        print("Gameover:", gameover)
        print("Reward:", reward)
        print("")


if __name__ == "__main__":
    nim = SimWorldNim(10, 5)
    gameover, reward = nim.get_gameover_and_reward()
    while not gameover:
        nim.print_current_game_state()
        child_states = nim.get_child_states()
        move = np.random.choice(child_states)
        nim.pick_move(move)
        gameover, reward = nim.get_gameover_and_reward()

    nim.print_current_game_state()
