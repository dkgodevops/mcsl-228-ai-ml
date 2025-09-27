import heapq
from typing import List, Tuple, Dict, Optional

Grid = List[List[int]]  # 0 = free cell, 1 = obstacle
Point = Tuple[int, int]

def heuristic_manhattan(a: Point, b: Point) -> int:
    """Manhattan distance heuristic for grid (admissible for 4-direction moves)."""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def neighbors_4(grid: Grid, node: Point) -> List[Point]:
    """Return valid 4-directional neighbors (up, down, left, right)."""
    rows, cols = len(grid), len(grid[0])
    r, c = node
    candidates = [(r-1,c), (r+1,c), (r,c-1), (r,c+1)]
    return [(nr, nc) for nr, nc in candidates
            if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] == 0]

def reconstruct_path(came_from: Dict[Point, Point], current: Point) -> List[Point]:
    """Reconstruct path from start to goal using came_from map."""
    path = [current]
    while current in came_from:
        current = came_from[current]
        path.append(current)
    return path[::-1]

def a_star(grid: Grid, start: Point, goal: Point,
           heuristic=heuristic_manhattan,
           neighbor_fn=neighbors_4) -> Optional[List[Point]]:
    """
    A* search on a 2D grid.
    Returns a list of points from start to goal (inclusive) if path found, otherwise None.
    """

    if grid[start[0]][start[1]] != 0:
        raise ValueError("Start is blocked")
    if grid[goal[0]][goal[1]] != 0:
        raise ValueError("Goal is blocked")

    open_set = []  # heap of (f_score, g_score, node)
    heapq.heappush(open_set, (heuristic(start, goal), 0, start))

    came_from: Dict[Point, Point] = {}
    g_score: Dict[Point, int] = {start: 0}
    f_score: Dict[Point, int] = {start: heuristic(start, goal)}

    closed_set = set()

    while open_set:
        _, current_g, current = heapq.heappop(open_set)

        if current in closed_set:
            continue

        if current == goal:
            return reconstruct_path(came_from, current)

        closed_set.add(current)

        for neighbor in neighbor_fn(grid, current):
            if neighbor in closed_set:
                continue

            tentative_g = g_score[current] + 1  # cost between neighbors = 1

            if tentative_g < g_score.get(neighbor, float('inf')):
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f = tentative_g + heuristic(neighbor, goal)
                f_score[neighbor] = f
                heapq.heappush(open_set, (f, tentative_g, neighbor))

    return None  # no path found

# ------------ Example usage ------------
if __name__ == "__main__":
    # 0 = free, 1 = obstacle
    grid = [
        [0,0,0,0,0,0],
        [0,1,1,1,0,0],
        [0,0,0,1,0,0],
        [0,1,0,0,0,0],
        [0,0,0,1,1,0],
        [0,0,0,0,0,0]
    ]

    start = (0, 0)
    goal  = (5, 5)

    path = a_star(grid, start, goal)

    if path:
        print("Path found (row, col):", path)
        # visualize path on grid
        visual = [['#' if grid[r][c] == 1 else '.' for c in range(len(grid[0]))] for r in range(len(grid))]
        for r, c in path:
            visual[r][c] = 'P'  # mark path
        visual[start[0]][start[1]] = 'S'
        visual[goal[0]][goal[1]] = 'G'
        print("\nGrid with path (S=start, G=goal, P=path, #=obstacle):\n")
        for row in visual:
            print(' '.join(row))
    else:
        print("No path found.")
