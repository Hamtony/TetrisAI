import numpy as np
import random
from collections import deque
from queue import Queue
def add_attack_rows(board, attack_rows):
    # Convert the board to a deque of rows
    board_deque = deque(board.tolist(),maxlen=20)
    print(board_deque)
    # Generate and add attack rows to the top
    for _ in range(attack_rows):
        row = [1] * 10
        empty_index = random.randint(0, 9)
        row[empty_index] = 0
        board_deque.append(row)
    
    # Ensure the board size is 20 rows by popping from the bottom if necessary
    while len(board_deque) > 20:
        board_deque.pop()
    b = []
    for row in board_deque:
        b.append(row)
    
    return b

# Example usage:
if __name__ == "__main__":
    # Create an example 20x10 board
    board = np.zeros((20, 10), dtype=int)
    
    # Fill some rows for testing
    board[18] = [1, 1, 1, 1, 0, 1, 1, 1, 1, 1]
    board[19] = [1, 1, 1, 1, 1, 1, 1, 1, 1, 0]
    print(board)
    
    # Number of attack rows to add
    attack_rows = 3
    
    # Add attack rows to the board
    updated_board = add_attack_rows(board, attack_rows)
    
    # Print the updated board
    print("Updated Tetris Board:")
    print(updated_board)
