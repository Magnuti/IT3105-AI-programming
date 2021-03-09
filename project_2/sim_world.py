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
        self.state = np.zeros(self.N, dtype=int)
        self.state[-1] = 1

    def __get_remaining_pieces(self):
        # print("Get rp", type(self.state))
        return np.where(self.state == 1)[0][0]

    def get_game_state(self):
        state = [0, 1] if self.current_player else [1, 0]
        state.extend(self.state)
        return np.array(state)

    def get_child_states(self):
        """
        Returns a list of size K of all possible states where the child states are not None
            For example, [[some state], [some state], None, None] where None means that the action
            is not legal. We need to pass down None values because we need to scale these illegal
            action's probabilities down to 0.
        """
        child_states = []
        for i in range(1, self.K + 1):
            if i > self.__get_remaining_pieces():
                child_states.append(None)
            else:
                child_state = [0, 1] if not self.current_player else [1, 0]
                child_state.extend(np.roll(self.state, -i))
                child_states.append(np.array(child_state))

        return child_states

    def pick_move(self, next_state):
        self.state = next_state[2:]
        self.current_player = 1 - self.current_player  # Flip between 0 and 1

    def set_state_and_player(self, state):
        """
        Used to reset the board to the state it was before the rollout simulation started.
        """
        self.current_player = state[:2]
        self.state = state[2:]

    def get_gameover_and_reward(self):
        if self.__get_remaining_pieces() > 0:
            return False, 0

        # Player 1 wants to maximize, while player 0 wants to minimize
        reward = 1 if self.current_player else -1
        return True, reward

    def print_current_game_state(self):
        game_state = self.get_game_state()
        # print("Current player:", game_state[0:2])
        print("Game state:\n\t", game_state)
        print("Child states:")
        for child_state in self.get_child_states():
            print("\t", child_state)
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
        legal_child_states = []
        for i in range(len(child_states)):
            if child_states[i] is not None:
                legal_child_states.append(child_states[i])

        move_index = np.random.randint(0, len(legal_child_states))
        nim.pick_move(legal_child_states[move_index])
        gameover, reward = nim.get_gameover_and_reward()

    nim.print_current_game_state()
