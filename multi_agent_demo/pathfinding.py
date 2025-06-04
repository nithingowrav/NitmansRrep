import heapq
from typing import List, Tuple, Dict, Optional
# Attempt to import from parent directory for warehouse_environment
import sys
import os
# Correctly go up two levels from multi_agent_demo/pathfinding.py to the root /app
# to allow importing multi_agent_demo.warehouse_environment
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from multi_agent_demo.warehouse_environment import WarehouseGrid


# Helper for A*
def heuristic(a: Tuple[int, int], b: Tuple[int, int]) -> int:
    """Manhattan distance heuristic."""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def get_neighbors(grid: WarehouseGrid, node: Tuple[int, int]) -> List[Tuple[int, int]]:
    """Get valid, non-obstacle neighbors of a node."""
    neighbors = []
    for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]: # Four cardinal directions
        x, y = node[0] + dx, node[1] + dy
        if grid.is_valid_location(x, y) and not grid.is_obstacle(x, y):
            neighbors.append((x, y))
    return neighbors

def a_star_search(grid: WarehouseGrid, start: Tuple[int, int], goal: Tuple[int, int]) -> List[Tuple[int, int]]:
    """
    A* pathfinding algorithm.
    Returns a list of (x,y) tuples representing the path from start to goal,
    or an empty list if no path is found.
    The path includes the start and goal nodes.
    """
    if not grid.is_valid_location(start[0], start[1]) or not grid.is_valid_location(goal[0], goal[1]):
        # print(f"A* Error: Start {start} or Goal {goal} is outside grid boundaries.")
        return []
    if grid.is_obstacle(start[0], start[1]) or grid.is_obstacle(goal[0], goal[1]):
        # print(f"A* Error: Start {start} or Goal {goal} is an obstacle.")
        return []
    if start == goal:
        return [start]

    open_set = []  # Priority queue: (f_score, g_score, node) - g_score is tie-breaker
    heapq.heappush(open_set, (heuristic(start, goal), 0, start)) # Add h_score as part of tuple for consistent sorting

    came_from: Dict[Tuple[int, int], Tuple[int, int]] = {} # Stores the previous node in the optimal path

    g_score: Dict[Tuple[int, int], float] = { (x,y): float('inf') for x in range(grid.width) for y in range(grid.height) }
    g_score[start] = 0

    f_score: Dict[Tuple[int, int], float] = { (x,y): float('inf') for x in range(grid.width) for y in range(grid.height) }
    f_score[start] = heuristic(start, goal)

    open_set_nodes = {start} # Keep track of nodes in open_set for faster lookups

    while open_set:
        _, current_g, current_node = heapq.heappop(open_set)

        # This check is important: if we've already processed this node via a shorter path discovered later
        # or if it was re-added with a higher g_score (less optimal path), we might ignore it.
        # However, standard A* with consistent heuristic should not re-add with higher g_score to open_set if already processed.
        # If node was popped, it means it was the best path found so far.
        # The check `if current_node not in open_set_nodes:` is usually for when removing from open_set_nodes
        # but here we remove it right after popping.
        # A check like `if current_g > g_score[current_node]: continue` can be useful if nodes can be re-added.
        # For now, this simple removal should be okay.
        if current_node not in open_set_nodes: # Node already processed or removed
             # This condition should ideally not be met if items are only removed from open_set_nodes after popping.
             # However, if it's possible for a node to be in open_set multiple times (e.g. if not checking before adding)
             # this could be a safeguard. But heapq doesn't allow easy update of priorities, so nodes might be added multiple times.
             # A more robust open_set_nodes check is `if node in closed_set` or by checking if g_score is already lower.
             pass # Node already processed with a better or equal path.

        open_set_nodes.remove(current_node)


        if current_node == goal:
            # Reconstruct path
            path = []
            temp = current_node
            while temp in came_from: # Check if temp has a predecessor
                path.append(temp)
                temp = came_from[temp]
            path.append(start) # Add the start node
            return path[::-1]  # Return reversed path

        for neighbor in get_neighbors(grid, current_node):
            tentative_g_score = g_score[current_node] + 1 # Cost to move to neighbor is 1 (g_score[current_node] is used here, not current_g)

            if tentative_g_score < g_score[neighbor]:
                came_from[neighbor] = current_node
                g_score[neighbor] = tentative_g_score
                new_f_score = tentative_g_score + heuristic(neighbor, goal)
                # f_score[neighbor] = new_f_score # f_score dict is mostly for reference/debugging here
                if neighbor not in open_set_nodes: # Check if neighbor is already scheduled for processing
                    heapq.heappush(open_set, (new_f_score, tentative_g_score, neighbor))
                    open_set_nodes.add(neighbor)
                # If using a closed set, add current_node to closed_set here.

    # print(f"A* Info: No path found from {start} to {goal}.")
    return [] # No path found


if __name__ == '__main__':
    # Example Usage
    # This example usage needs to be adjusted because warehouse_environment is now a module
    # For direct execution, you'd run this from the multi_agent_demo directory or adjust PYTHONPATH
    if os.path.basename(os.getcwd()) != 'multi_agent_demo' and os.path.basename(os.getcwd()) != 'app':
        sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))) # Go up one level to app

    # Now try importing for the __main__ block
    from multi_agent_demo.warehouse_environment import WarehouseGrid
    # pathfinding functions are already defined in this file

    # Create a simple warehouse grid
    warehouse = WarehouseGrid(width=10, height=10)
    warehouse.obstacles = [(1,0), (1,1), (1,2), (1,3), (1,4)] # A wall

    start_node = (0,2)
    goal_node = (3,2)

    print(f"Finding path from {start_node} to {goal_node} in a {warehouse.width}x{warehouse.height} grid.")
    print(f"Obstacles at: {warehouse.obstacles}")

    path = a_star_search(warehouse, start_node, goal_node)

    if path:
        print(f"Path found: {path}")
        # Visualize path (simple text print)
        for y in range(warehouse.height):
            row_str = "" # Renamed from 'row' to avoid conflict with outer scope 'row' if any
            for x in range(warehouse.width):
                if (x,y) == start_node:
                    row_str += "S "
                elif (x,y) == goal_node:
                    row_str += "G "
                elif (x,y) in path:
                    row_str += "* "
                elif (x,y) in warehouse.obstacles:
                    row_str += "# "
                else:
                    row_str += ". "
            print(row_str)
    else:
        print("No path found.")

    print("\nTesting path to self:")
    path_to_self = a_star_search(warehouse, start_node, start_node)
    print(f"Path to self: {path_to_self}")

    print("\nTesting path to obstacle:")
    path_to_obstacle = a_star_search(warehouse, start_node, (1,0))
    print(f"Path to obstacle: {path_to_obstacle}")

    print("\nTesting path from obstacle:")
    path_from_obstacle = a_star_search(warehouse, (1,0), goal_node)
    print(f"Path from obstacle: {path_from_obstacle}")

    print("\nTesting path to out of bounds:")
    path_to_oob = a_star_search(warehouse, start_node, (100,100))
    print(f"Path to out of bounds: {path_to_oob}")

    print("\nTesting path where no path exists (blocked):")
    warehouse.obstacles.append((2,2)) # Block the path
    path_blocked = a_star_search(warehouse, start_node, goal_node)
    print(f"Path (blocked): {path_blocked}")
