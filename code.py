
import numpy as np
import pickle
import heapq
from collections import deque
import time
import tracemalloc

# Algorithm: Iterative Deepening Search (IDS)
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
    visited = [False] * len(adj_matrix)
    while depth < len(adj_matrix):
        if all(visited):  # Terminate if all nodes are visited
            return None
        visited = [False] * len(adj_matrix)
        result = dls(start_node, depth, visited)
        if result:
            return result
        depth += 1
    return None  # If no path found

# Algorithm: Bi-Directional Search
def get_bidirectional_search_path(adj_matrix, start_node, goal_node):
    forward_queue = deque([(start_node, [start_node])])
    backward_queue = deque([(goal_node, [goal_node])])
    forward_visited = set([start_node])
    backward_visited = set([goal_node])
    total_visited = set([start_node, goal_node])
    
    while forward_queue and backward_queue:
        if len(total_visited) == len(adj_matrix):  # Terminate if all nodes are visited
            return None

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
                total_visited.add(neighbor)

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
                total_visited.add(neighbor)

    return None  # If no path found

# Algorithm: A* Search Algorithm
def get_astar_search_path(adj_matrix, node_attributes, start_node, goal_node):
    def heuristic(node):
        return ((node_attributes[node]['x'] - node_attributes[goal_node]['x']) ** 2 +
                (node_attributes[node]['y'] - node_attributes[goal_node]['y']) ** 2) ** 0.5

    open_set = [(0, start_node)]
    came_from = {}
    g_score = {start_node: 0}
    f_score = {start_node: heuristic(start_node)}
    closed_set = set()
    total_visited = set([start_node])

    while open_set:
        if len(total_visited) == len(adj_matrix):  # Terminate if all nodes are visited
            return None

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
                    total_visited.add(neighbor)

    return None  # If no path found

# Algorithm: Bi-Directional Heuristic Search
def get_bidirectional_heuristic_search_path(adj_matrix, node_attributes, start_node, goal_node):
    def heuristic(node, target):
        return ((node_attributes[node]['x'] - node_attributes[target]['x']) ** 2 +
                (node_attributes[node]['y'] - node_attributes[target]['y']) ** 2) ** 0.5

    forward_open_set = [(0, start_node)]
    backward_open_set = [(0, goal_node)]
    forward_came_from = {}
    backward_came_from = {}
    forward_g_score = {start_node: 0}
    backward_g_score = {goal_node: 0}
    forward_closed_set = set()
    backward_closed_set = set()
    total_visited = set([start_node, goal_node])

    while forward_open_set and backward_open_set:
        if len(total_visited) == len(adj_matrix):  # Terminate if all nodes are visited
            return None

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
                    total_visited.add(neighbor)

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
                    total_visited.add(neighbor)

    return None  # If no path found

# Helper function to reconstruct the path
def reconstruct_bidirectional_path(forward_came_from, backward_came_from, meeting_point):
    forward_path = []
    current = meeting_point
    while current in forward_came_from:
        forward_path.append(current)
        current = forward_came_from[current]
    forward_path.append(current)
    forward_path.reverse()

    backward_path = []
    current = meeting_point
    while current in backward_came_from:
        current = backward_came_from[current]
        backward_path.append(current)

    return forward_path + backward_path

# Bonus Problem
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

# Track performance function
def track_performance(func, *args):
    """Wrapper function to track time and memory usage."""
    tracemalloc.start()
    start_time = time.time()

    result = func(*args)

    end_time = time.time()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    execution_time = end_time - start_time
    memory_usage = peak / 1024  # Convert to KB

    return result, execution_time, memory_usage


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






import numpy as np
import heapq
import pickle
from collections import deque
import time
import tracemalloc

def get_ids_path(adj_matrix, start_node, end_node):
    def dls(node, depth, visited):
        if depth == 0 and node == end_node:
            return [node]
        if depth > 0:
            visited.add(node)
            for neighbor, connected in enumerate(adj_matrix[node]):
                if connected and neighbor not in visited:
                    path = dls(neighbor, depth - 1, visited)
                    if path:
                        return [node] + path
        return None

    depth = 0
    visited = set()
    while depth < len(adj_matrix):
        if len(visited) == len(adj_matrix):  # All nodes visited
            return None
        visited = set()  # Reset visited for each depth
        result = dls(start_node, depth, visited)
        if result:
            return result
        depth += 1
    return None  # No path found

# Algorithm: Bi-Directional Search
def get_bidirectional_search_path(adj_matrix, start_node, goal_node):
    forward_queue = deque([(start_node, [start_node])])
    backward_queue = deque([(goal_node, [goal_node])])
    forward_visited = set([start_node])
    backward_visited = set([goal_node])
    total_visited = set([start_node, goal_node])
    
    while forward_queue and backward_queue:
        if len(total_visited) == len(adj_matrix):  # Terminate if all nodes are visited
            return None

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
                total_visited.add(neighbor)

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
                total_visited.add(neighbor)

    return None  # No path found

# Algorithm: A* Search Algorithm
def get_astar_search_path(adj_matrix, node_attributes, start_node, goal_node):
    def heuristic(node, goal):
        return ((node_attributes[node]['x'] - node_attributes[goal]['x']) ** 2 + 
                (node_attributes[node]['y'] - node_attributes[goal]['y']) ** 2) ** 0.5

    open_set = [(0, start_node)]
    came_from = {}
    g_score = {start_node: 0}
    f_score = {start_node: heuristic(start_node, goal_node)}
    closed_set = set()
    total_visited = set([start_node])

    while open_set:
        if len(total_visited) == len(adj_matrix):  # Terminate if all nodes are visited
            return None

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
                    f_score[neighbor] = g_score[neighbor] + heuristic(neighbor, goal_node)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))
                    total_visited.add(neighbor)

    return None  # No path found

# Algorithm: Bi-Directional Heuristic Search
def get_bidirectional_heuristic_search_path(adj_matrix, node_attributes, start_node, goal_node):
    def heuristic(node, target):
        return ((node_attributes[node]['x'] - node_attributes[target]['x']) ** 2 + 
                (node_attributes[node]['y'] - node_attributes[target]['y']) ** 2) ** 0.5

    forward_open_set = [(0, start_node)]
    backward_open_set = [(0, goal_node)]
    forward_came_from = {}
    backward_came_from = {}
    forward_g_score = {start_node: 0}
    backward_g_score = {goal_node: 0}
    forward_closed_set = set()
    backward_closed_set = set()
    total_visited = set([start_node, goal_node])

    while forward_open_set and backward_open_set:
        if len(total_visited) == len(adj_matrix):  # Terminate if all nodes are visited
            return None

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
                    total_visited.add(neighbor)

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
                    total_visited.add(neighbor)

    return None  # No path found

# Helper function to reconstruct the path for bidirectional searches
def reconstruct_bidirectional_path(forward_came_from, backward_came_from, meeting_point):
    forward_path = []
    current = meeting_point
    while current in forward_came_from:
        forward_path.append(current)
        current = forward_came_from[current]
    forward_path.append(current)
    forward_path.reverse()

    backward_path = []
    current = meeting_point
    while current in backward_came_from:
        current = backward_came_from[current]
        backward_path.append(current)

    return forward_path + backward_path

# Bonus Problem
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

# Track performance function
def track_performance(func, *args):
    """Wrapper function to track time and memory usage."""
    tracemalloc.start()
    start_time = time.time()

    result = func(*args)

    end_time = time.time()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    execution_time = end_time - start_time
    memory_usage = peak / 1024  # Convert to KB

    return result, execution_time, memory_usage

# Main script
if __name__ == "__main__":
    adj_matrix = np.load('IIIT_Delhi.npy')
    with open('IIIT_Delhi.pkl', 'rb') as f:
        node_attributes = pickle.load(f)

    start_node = int(input("Enter the Start Node: "))
    goal_node = int(input("Enter the Goal Node: " ))
    print("\n")
    # Run Iterative Deepening Search (IDS)
    ids_result, ids_time, ids_memory = track_performance(get_ids_path, adj_matrix, start_node, goal_node)
    print(f"Iterative Deepening Seach Path {ids_result}\nIDS Execution Time: {ids_time:.4f}s\nIDS Memory Usage: {ids_memory:.2f}KB\n")

    # Run Bi-Directional Search
    bidir_result, bidir_time, bidir_memory = track_performance(get_bidirectional_search_path, adj_matrix, start_node, goal_node)
    print(f"Bi-Directional Search Path: {bidir_result}\nBDS Execution Time: {bidir_time:.4f}s\nBDS Memory Usage: {bidir_memory:.2f}KB\n")

    # Run A* Search
    astar_result, astar_time, astar_memory = track_performance(get_astar_search_path, adj_matrix, node_attributes, start_node, goal_node)
    print(f"A* Search Path: {astar_result}\nA* Execution Time: {astar_time:.4f}s\nA* Memory Usage: {astar_memory:.2f}KB\n")

    # Run Bi-Directional Heuristic Search
    bidir_heur_result, bidir_heur_time, bidir_heur_memory = track_performance(get_bidirectional_heuristic_search_path, adj_matrix, node_attributes, start_node, goal_node)
    print(f"Bi-Directional Heuristic Search Path: {bidir_heur_result}\nExecution Time: {bidir_heur_time:.4f}s\nMemory Usage: {bidir_heur_memory:.2f}KB\n")

    # Run Bonus Problem
    bonus_result, bonus_time, bonus_memory = track_performance(bonus_problem, adj_matrix)
    print(f"Bonus Problem: {bonus_result}\nBonus Problem Execution Time: {bonus_time:.4f}s\nBonus Problem Memory Usage: {bonus_memory:.2f}KB\n")
