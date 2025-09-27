import math

"""
Here’s a simple Python implementation of the Minimax Algorithm (used in decision-making and game theory, 
especially in two-player games like Tic-Tac-Toe or Chess).
We’ll implement a general version for a game tree:
"""
# Minimax function
def minimax(depth, node_index, is_maximizing_player, scores, height):
    # Base case: leaf node reached
    if depth == height:
        return scores[node_index]

    if is_maximizing_player:
        return max(
            minimax(depth + 1, node_index * 2, False, scores, height),
            minimax(depth + 1, node_index * 2 + 1, False, scores, height)
        )
    else:
        return min(
            minimax(depth + 1, node_index * 2, True, scores, height),
            minimax(depth + 1, node_index * 2 + 1, True, scores, height)
        )

# Example usage
if __name__ == "__main__":
    # Game tree (scores at the leaf nodes)
    scores = [3, 5, 2, 9, 12, 5, 23, 23]

    # Height of the tree = log2(len(scores))
    height = math.log2(len(scores))

    optimal_value = minimax(0, 0, True, scores, int(height))
    print("The optimal value is:", optimal_value)
