import numpy as np
import pickle
import heapq
from collections import deque
import time
import tracemalloc

# General Notes:
# - Update the provided file name (code_<RollNumber>.py) as per the instructions.
# - Do not change the function name, number of parameters or the sequence of parameters.
# - The expected output for each function is a path (list of node names)
# - Ensure that the returned path includes both the start node and the goal node, in the correct order.
# - If no valid path exists between the start and goal nodes, the function should return None.


# Algorithm: Iterative Deepening Search (IDS)

# Input:
#   - adj_matrix: Adjacency matrix representing the graph.
#   - start_node: The starting node in the graph.
#   - goal_node: The target node in the graph.

# Return:
#   - A list of node names representing the path from the start_node to the goal_node.
#   - If no path exists, the function should return None.

# Sample Test Cases:

#   Test Case 1:
#     - Start node: 1, Goal node: 2
#     - Return: [1, 7, 6, 2]

#   Test Case 2:
#     - Start node: 5, Goal node: 12
#     - Return: [5, 97, 98, 12]

#   Test Case 3:
#     - Start node: 12, Goal node: 49
#     - Return: None

#   Test Case 4:
#     - Start node: 4, Goal node: 12
#     - Return: [4, 6, 2, 9, 8, 5, 97, 98, 12]

def get_ids_path(adj_matrix, start_node, end_node):
    def dls(node, depth, visited):
        if depth == 0 and node == end_node:
            return [node]
        if depth > 0:
            visited[node] = True
            for neighbor, connected in enumerate(adj_matrix[node]):
                if connected and not visited[neighbor]:
                    path = dls(neighbor, depth - 1, visited)
                    if path:
                        return [node] + path
            visited[node] = False
        return None

    depth = 0
    while depth < len(adj_matrix):
        visited = [False] * len(adj_matrix)
        result = dls(start_node, depth, visited)
        if result:
            return result
        depth += 1
    return None  # If no path found


def track_performance(func, *args):
    """Wrapper function to track time and memory usage."""
    # Start tracking memory usage
    tracemalloc.start()
    
    # Start tracking time
    start_time = time.time()

    # Call the function
    result = func(*args)

    # Stop tracking time
    end_time = time.time()
    
    # Stop tracking memory usage
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    # Compute time and memory stats
    execution_time = end_time - start_time
    memory_usage = peak / 1024  # Convert to KB

    return result, execution_time, memory_usage

# Algorithm: Bi-Directional Search

# Input:
#   - adj_matrix: Adjacency matrix representing the graph.
#   - start_node: The starting node in the graph.
#   - goal_node: The target node in the graph.

# Return:
#   - A list of node names representing the path from the start_node to the goal_node.
#   - If no path exists, the function should return None.

# Sample Test Cases:

#   Test Case 1:
#     - Start node: 1, Goal node: 2
#     - Return: [1, 7, 6, 2]

#   Test Case 2:
#     - Start node: 5, Goal node: 12
#     - Return: [5, 97, 98, 12]

#   Test Case 3:
#     - Start node: 12, Goal node: 49
#     - Return: None

#   Test Case 4:
#     - Start node: 4, Goal node: 12
#     - Return: [4, 6, 2, 9, 8, 5, 97, 98, 12]

def get_bidirectional_search_path(adj_matrix, start_node, goal_node):
    forward_queue = deque([(start_node, [start_node])])
    backward_queue = deque([(goal_node, [goal_node])])
    forward_visited = set([start_node])
    backward_visited = set([goal_node])
    max_iterations = len(adj_matrix) * 2  # Set a maximum number of iterations

    for _ in range(max_iterations):
        if not forward_queue or not backward_queue:
            return None  # No path exists

        # Forward search
        current, path = forward_queue.popleft()
        if current in backward_visited:
            for bw_node, bw_path in backward_queue:
                if bw_node == current:
                    return path + bw_path[::-1][1:]
        for neighbor in range(len(adj_matrix)):
            if adj_matrix[current][neighbor] > 0 and neighbor not in forward_visited:
                forward_queue.append((neighbor, path + [neighbor]))
                forward_visited.add(neighbor)

        # Backward search
        current, path = backward_queue.popleft()
        if current in forward_visited:
            for fw_node, fw_path in forward_queue:
                if fw_node == current:
                    return fw_path + path[::-1][1:]
        for neighbor in range(len(adj_matrix)):
            if adj_matrix[neighbor][current] > 0 and neighbor not in backward_visited:
                backward_queue.append((neighbor, path + [neighbor]))
                backward_visited.add(neighbor)

    return None  # No path found within the maximum number of iterations


# Algorithm: A* Search Algorithm

# Input:
#   - adj_matrix: Adjacency matrix representing the graph.
#   - node_attributes: Dictionary of node attributes containing x, y coordinates for heuristic calculations.
#   - start_node: The starting node in the graph.
#   - goal_node: The target node in the graph.

# Return:
#   - A list of node names representing the path from the start_node to the goal_node.
#   - If no path exists, the function should return None.

# Sample Test Cases:

#   Test Case 1:
#     - Start node: 1, Goal node: 2
#     - Return: [1, 7, 6, 2]

#   Test Case 2:
#     - Start node: 5, Goal node: 12
#     - Return: [5, 97, 28, 10, 12]

#   Test Case 3:
#     - Start node: 12, Goal node: 49
#     - Return: None

#   Test Case 4:
#     - Start node: 4, Goal node: 12
#     - Return: [4, 6, 27, 9, 8, 5, 97, 28, 10, 12]

def get_astar_search_path(adj_matrix, node_attributes, start_node, goal_node):
    def heuristic(node):
        return ((node_attributes[node]['x'] - node_attributes[goal_node]['x'])**2 +
                (node_attributes[node]['y'] - node_attributes[goal_node]['y'])**2)**0.5

    open_set = [(0, start_node)]
    came_from = {}
    g_score = {start_node: 0}
    f_score = {start_node: heuristic(start_node)}
    closed_set = set()
    max_iterations = len(adj_matrix) * 2  # Set a maximum number of iterations

    for _ in range(max_iterations):
        if not open_set:
            return None  # No path exists

        current = heapq.heappop(open_set)[1]

        if current == goal_node:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start_node)
            return path[::-1]

        closed_set.add(current)

        for neighbor in range(len(adj_matrix)):
            if adj_matrix[current][neighbor] > 0 and neighbor not in closed_set:
                tentative_g_score = g_score[current] + adj_matrix[current][neighbor]
                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = g_score[neighbor] + heuristic(neighbor)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))

    return None  # No path found within the maximum number of iterations


# Algorithm: Bi-Directional Heuristic Search

# Input:
#   - adj_matrix: Adjacency matrix representing the graph.
#   - node_attributes: Dictionary of node attributes containing x, y coordinates for heuristic calculations.
#   - start_node: The starting node in the graph.
#   - goal_node: The target node in the graph.

# Return:
#   - A list of node names representing the path from the start_node to the goal_node.
#   - If no path exists, the function should return None.

# Sample Test Cases:

#   Test Case 1:
#     - Start node: 1, Goal node: 2
#     - Return: [1, 7, 6, 2]

#   Test Case 2:
#     - Start node: 5, Goal node: 12
#     - Return: [5, 97, 98, 12]

#   Test Case 3:
#     - Start node: 12, Goal node: 49
#     - Return: None

#   Test Case 4:
#     - Start node: 4, Goal node: 12
#     - Return: [4, 34, 33, 11, 32, 31, 3, 5, 97, 28, 10, 12]

def get_bidirectional_heuristic_search_path(adj_matrix, node_attributes, start_node, goal_node):
    def heuristic(node, target):
        return ((node_attributes[node]['x'] - node_attributes[target]['x'])**2 +
                (node_attributes[node]['y'] - node_attributes[target]['y'])**2)**0.5

    forward_open_set = [(0, start_node)]
    backward_open_set = [(0, goal_node)]
    forward_came_from = {}
    backward_came_from = {}
    forward_g_score = {start_node: 0}
    backward_g_score = {goal_node: 0}
    forward_closed_set = set()
    backward_closed_set = set()
    max_iterations = len(adj_matrix) * 2  # Set a maximum number of iterations

    for _ in range(max_iterations):
        if not forward_open_set or not backward_open_set:
            return None  # No path exists

        # Forward search
        _, current_forward = heapq.heappop(forward_open_set)
        if current_forward in backward_came_from:
            return reconstruct_bidirectional_path(forward_came_from, backward_came_from, current_forward)

        forward_closed_set.add(current_forward)

        for neighbor in range(len(adj_matrix)):
            if adj_matrix[current_forward][neighbor] > 0 and neighbor not in forward_closed_set:
                tentative_g_score = forward_g_score[current_forward] + adj_matrix[current_forward][neighbor]
                if neighbor not in forward_g_score or tentative_g_score < forward_g_score[neighbor]:
                    forward_came_from[neighbor] = current_forward
                    forward_g_score[neighbor] = tentative_g_score
                    f_score = forward_g_score[neighbor] + heuristic(neighbor, goal_node)
                    heapq.heappush(forward_open_set, (f_score, neighbor))

        # Backward search
        _, current_backward = heapq.heappop(backward_open_set)
        if current_backward in forward_came_from:
            return reconstruct_bidirectional_path(forward_came_from, backward_came_from, current_backward)

        backward_closed_set.add(current_backward)

        for neighbor in range(len(adj_matrix)):
            if adj_matrix[neighbor][current_backward] > 0 and neighbor not in backward_closed_set:
                tentative_g_score = backward_g_score[current_backward] + adj_matrix[neighbor][current_backward]
                if neighbor not in backward_g_score or tentative_g_score < backward_g_score[neighbor]:
                    backward_came_from[neighbor] = current_backward
                    backward_g_score[neighbor] = tentative_g_score
                    f_score = backward_g_score[neighbor] + heuristic(neighbor, start_node)
                    heapq.heappush(backward_open_set, (f_score, neighbor))

    return None  # No path found within the maximum number of iterations

  
def reconstruct_bidirectional_path(forward_came_from, backward_came_from, meeting_point):
        # Reconstruct the path from the start node to the meeting point
    forward_path = []
    current = meeting_point
    while current in forward_came_from:
        forward_path.append(current)
        current = forward_came_from[current]
    forward_path.append(current)
    forward_path.reverse()  # Reverse the forward path to get the correct order

    # Reconstruct the path from the meeting point to the goal node
    backward_path = []
    current = meeting_point
    while current in backward_came_from:
        current = backward_came_from[current]
        backward_path.append(current)

    # Combine both paths, excluding the meeting point from the backward path
    return forward_path + backward_path




# Bonus Problem
 
# Input:
# - adj_matrix: A 2D list or numpy array representing the adjacency matrix of the graph.

# Return:
# - A list of tuples where each tuple (u, v) represents an edge between nodes u and v.
#   These are the vulnerable roads whose removal would disconnect parts of the graph.

# Note:
# - The graph is undirected, so if an edge (u, v) is vulnerable, then (v, u) should not be repeated in the output list.
# - If the input graph has no vulnerable roads, return an empty list [].

def bonus_problem(adj_matrix):
    def dfs(node, visited):
        visited[node] = True
        for neighbor in range(len(adj_matrix)):
            if adj_matrix[node][neighbor] > 0 and not visited[neighbor]:
                dfs(neighbor, visited)

    vulnerable_roads = []
    n = len(adj_matrix)

    for u in range(n):
        for v in range(u + 1, n):
            if adj_matrix[u][v] > 0:
                # Remove the edge
                adj_matrix[u][v] = adj_matrix[v][u] = 0

                # Check if the graph is still connected
                visited = [False] * n
                dfs(0, visited)

                if not all(visited):
                    vulnerable_roads.append((u, v))

                # Restore the edge
                adj_matrix[u][v] = adj_matrix[v][u] = 1

    return vulnerable_roads


# if __name__ == "__main__":
#   adj_matrix = np.load('IIIT_Delhi.npy')
#   with open('IIIT_Delhi.pkl', 'rb') as f:
#     node_attributes = pickle.load(f)

#   start_node = int(input("Enter the start node: "))
#   end_node = int(input("Enter the end node: "))

#   print(f'Iterative Deepening Search Path: {get_ids_path(adj_matrix,start_node,end_node)}')
#   print(f'Bidirectional Search Path: {get_bidirectional_search_path(adj_matrix,start_node,end_node)}')
#   print(f'A* Path: {get_astar_search_path(adj_matrix,node_attributes,start_node,end_node)}')
#   print(f'Bidirectional Heuristic Search Path: {get_bidirectional_heuristic_search_path(adj_matrix,node_attributes,start_node,end_node)}')
#   print(f'Bonus Problem: {bonus_problem(adj_matrix)}')

if __name__ == "__main__":
    adj_matrix = np.load('IIIT_Delhi.npy')
    with open('IIIT_Delhi.pkl', 'rb') as f:
        node_attributes = pickle.load(f)

    start_node = int(input("Enter the start node: "))
    end_node = int(input("Enter the end node: "))

    # Iterative Deepening Search
    ids_path, ids_time, ids_memory = track_performance(get_ids_path, adj_matrix, start_node, end_node)
    print(f'Iterative Deepening Search Path: {ids_path if ids_path else "None"}')
    print(f'Execution Time: {ids_time:.6f} seconds, Memory Usage: {ids_memory:.2f} KB')

    # Bidirectional Search
    bidirectional_path, bidirectional_time, bidirectional_memory = track_performance(get_bidirectional_search_path, adj_matrix, start_node, end_node)
    print(f'Bidirectional Search Path: {bidirectional_path if bidirectional_path else "None"}')
    print(f'Execution Time: {bidirectional_time:.6f} seconds, Memory Usage: {bidirectional_memory:.2f} KB')

    # A* Search
    astar_path, astar_time, astar_memory = track_performance(get_astar_search_path, adj_matrix, node_attributes, start_node, end_node)
    print(f'A* Path: {astar_path if astar_path else "None"}')
    print(f'Execution Time: {astar_time:.6f} seconds, Memory Usage: {astar_memory:.2f} KB')

    # Bidirectional Heuristic Search
    bidirectional_heuristic_path, bidirectional_heuristic_time, bidirectional_heuristic_memory = track_performance(get_bidirectional_heuristic_search_path, adj_matrix, node_attributes, start_node, end_node)
    print(f'Bidirectional Heuristic Search Path: {bidirectional_heuristic_path if bidirectional_heuristic_path else "None"}')
    print(f'Execution Time: {bidirectional_heuristic_time:.6f} seconds, Memory Usage: {bidirectional_heuristic_memory:.2f} KB')

    # Bonus Problem
    bonus_solution, bonus_time, bonus_memory = track_performance(bonus_problem, adj_matrix)
    print(f'Bonus Problem: {bonus_solution if bonus_solution else "None"}')
    print(f'Execution Time: {bonus_time:.6f} seconds, Memory Usage: {bonus_memory:.2f} KB')
