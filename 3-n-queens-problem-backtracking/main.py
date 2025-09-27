def solve_n_queens(n):
    """
    Solve the N-Queens problem and return all solutions.
    Each solution is returned as a list of column indices:
      solution[row] = column index (0-based) where the queen is placed in that row.
    """

    solutions = []
    cols = set()          # occupied columns
    diag1 = set()         # occupied "r - c" diagonals
    diag2 = set()         # occupied "r + c" diagonals
    board = [-1] * n      # board[r] = column of queen in row r

    def backtrack(row):
        if row == n:
            # Found a valid placement for all rows; save a copy
            solutions.append(board.copy())
            return

        for col in range(n):
            if col in cols or (row - col) in diag1 or (row + col) in diag2:
                continue

            # Place queen
            cols.add(col)
            diag1.add(row - col)
            diag2.add(row + col)
            board[row] = col

            backtrack(row + 1)

            # Remove queen (backtrack)
            cols.remove(col)
            diag1.remove(row - col)
            diag2.remove(row + col)
            board[row] = -1

    backtrack(0)
    return solutions


def pretty_print_solutions(solutions):
    """
    Convert solutions in index form to a readable board representation
    and print them.
    """
    if not solutions:
        print("No solutions.")
        return

    n = len(solutions[0])
    for idx, sol in enumerate(solutions, start=1):
        print(f"Solution #{idx}:")
        for r in range(n):
            row = ['.'] * n
            row[sol[r]] = 'Q'
            print(' '.join(row))
        print()


if __name__ == "__main__":
    N = 8  # change N to any positive integer
    sols = solve_n_queens(N)
    print(f"Total solutions for N={N}: {len(sols)}\n")
    # Print first 3 solutions (or all, if you prefer)
    pretty_print_solutions(sols[:3])
