from collections import deque

def bfs(graph, start):
    visited = set()           # To keep track of visited nodes
    queue = deque([start])    # Queue for BFS
    bfs_order = []            # To store the BFS traversal order

    while queue:
        vertex = queue.popleft()
        if vertex not in visited:
            visited.add(vertex)
            bfs_order.append(vertex)

            # Add all unvisited neighbors to the queue
            for neighbor in graph[vertex]:
                if neighbor not in visited:
                    queue.append(neighbor)

    return bfs_order

# Example usage
if __name__ == "__main__":
    # Graph represented as an adjacency list (dictionary)
    graph = {
        'A': ['B', 'C'],
        'B': ['D', 'E'],
        'C': ['F'],
        'D': [],
        'E': ['F'],
        'F': []
    }

    start_node = 'A'
    # Perform BFS and print the traversal order
    print("BFS Traversal starting from node", start_node, ":", bfs(graph, start_node))
