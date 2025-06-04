import time
# Ensure other modules can be imported
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__))) # Add multi_agent_demo to path

# Need to import List for type hinting in print_warehouse_state if not already globally available
# For Python 3.9+ List can be imported from typing or used directly if it's a type hint for list.
# Explicitly importing for clarity, though often not strictly needed for List in type hints in newer Python.
from typing import List

from warehouse_environment import WarehouseGrid, Package
from agents.robot_agent import RobotAgent
from task_manager import TaskManager
# Pathfinding might be needed here if we want to draw paths, but robots handle their own.

def print_warehouse_state(warehouse: WarehouseGrid, robots: List[RobotAgent], tick: int, tasks_info: str):
    """Prints a simple text-based representation of the warehouse state."""
    grid_repr = [['.' for _ in range(warehouse.width)] for _ in range(warehouse.height)]

    # Mark obstacles
    for ox, oy in warehouse.obstacles:
        if 0 <= oy < warehouse.height and 0 <= ox < warehouse.width:
            grid_repr[oy][ox] = '#'

    # Mark shelves (S) and packages on shelves (P)
    for (sx, sy), pkgs_on_shelf in warehouse.shelves.items():
        if 0 <= sy < warehouse.height and 0 <= sx < warehouse.width:
            if pkgs_on_shelf: # If packages are there
                grid_repr[sy][sx] = 'P' # Corrected: Use sy, sx for grid_repr index
            else:
                grid_repr[sy][sx] = 'S' # Corrected: Use sy, sx

    # Mark drop-off zones
    for dx, dy in warehouse.drop_off_zones:
        if 0 <= dy < warehouse.height and 0 <= dx < warehouse.width:
            grid_repr[dy][dx] = 'D'

    # Mark charging_stations
    for cx, cy in warehouse.charging_stations:
        if 0 <= cy < warehouse.height and 0 <= cx < warehouse.width:
            grid_repr[cy][cx] = 'C'

    # Mark robots (R or r if carrying package)
    for robot in robots:
        rx, ry = robot.state.current_location
        if 0 <= ry < warehouse.height and 0 <= rx < warehouse.width:
            robot_char = robot.state.id[0].upper() # Default to first char of ID (e.g. R for R1)
            if robot.package_in_possession:
                robot_char = robot.state.id[0].lower() # Lowercase if carrying
            elif robot.state.status == "needs_charging" or robot.state.battery_level < 30 :
                # Use a different char or keep it as is, but maybe indicate low battery
                # For now, let's keep it simple, or use a fixed 'L' if preferred over ID char.
                # For example, robot_char = 'L'
                pass # Robot character already set to its ID's first letter.

            # Append robot number to character e.g. R1, r2
            display_char = f"{robot_char}{robot.state.id[-1]}"

            # If cell has multiple features (e.g. robot on shelf/dropoff), prioritize robot
            grid_repr[ry][rx] = display_char

    print(f"\n--- Tick: {tick} ---")
    for r_idx, row_list in enumerate(grid_repr): # Renamed 'row' to 'row_list' to avoid conflict
        print(f"{str(r_idx).zfill(2)}| {' '.join(row_list)}")
    print("   " + "-" * (warehouse.width * 2 -1))
    print("   " + " ".join([str(i).zfill(1) for i in range(warehouse.width)])) # X-axis numbers

    print("\nRobots Status:")
    for robot in robots:
        print(f"  {robot}")
    print(f"\nTasks Overview: {tasks_info}")
    print("-" * (warehouse.width * 2 + 5 if warehouse.width > 0 else 20))


def main():
    print("Starting Smart Warehouse Simulation...")

    # 1. Initialize Warehouse Environment
    sim_warehouse = WarehouseGrid(width=15, height=10)
    # Clear any defaults from __post_init__ if we are setting everything manually
    sim_warehouse.obstacles.clear()
    sim_warehouse.drop_off_zones.clear()
    sim_warehouse.charging_stations.clear()
    sim_warehouse.shelves.clear()

    sim_warehouse.obstacles = [(2,2), (2,3), (2,4), (7,5), (7,6), (7,7), (7,8), (12,4), (12,5)]
    sim_warehouse.drop_off_zones = [(14, 8), (14,1)]
    sim_warehouse.charging_stations = [(0,0), (0,9), (14,0), (14,9)]

    packages_to_add = [
        Package(id="PkgA", location=(1,1), destination_zone=sim_warehouse.drop_off_zones[0]),
        Package(id="PkgB", location=(5,3), destination_zone=sim_warehouse.drop_off_zones[1]),
        Package(id="PkgC", location=(8,2), destination_zone=sim_warehouse.drop_off_zones[0]),
        Package(id="PkgD", location=(10,6), destination_zone=sim_warehouse.drop_off_zones[1]),
        Package(id="PkgE", location=(3,7), destination_zone=sim_warehouse.drop_off_zones[0]),
    ]
    for pkg in packages_to_add:
        sim_warehouse.add_package_to_shelf(pkg, pkg.location)

    # 2. Instantiate Robot Agents
    robot_agents = [
        RobotAgent(robot_id="R1", initial_location=(0,1), warehouse=sim_warehouse),
        RobotAgent(robot_id="R2", initial_location=(0,8), warehouse=sim_warehouse),
        RobotAgent(robot_id="R3", initial_location=(7,0), warehouse=sim_warehouse),
    ]

    # 3. Instantiate Task Manager
    task_manager = TaskManager(sim_warehouse, robot_agents)

    # 4. Simulation Loop
    max_ticks = 200
    simulation_delay = 0.3 # Adjusted for potentially faster simulation if printing is quick

    all_tasks_completed = False

    for tick in range(1, max_ticks + 1):
        # a. Assign tasks (TaskManager checks for idle robots and pending tasks)
        task_manager.assign_task_to_robots()

        # b. Robots run their step
        for robot in robot_agents:
            robot.run_step(task_manager)

        # c. Print state
        active_tasks_count = 0
        pending_tasks_count = 0
        completed_tasks_count = 0
        failed_tasks_count = 0
        for task_id, task_obj in task_manager.tasks.items():
            if task_obj.status == "pending":
                pending_tasks_count +=1
            elif task_obj.status == "completed":
                completed_tasks_count +=1
            elif "failed" in task_obj.status: # Count all types of failed tasks
                failed_tasks_count +=1
            else: # Any other non-pending, non-completed, non-failed status is active
                active_tasks_count +=1

        tasks_summary = (f"Pending: {pending_tasks_count}, Active: {active_tasks_count}, "
                         f"Completed: {completed_tasks_count}/{len(task_manager.tasks)}, Failed: {failed_tasks_count}")

        print_warehouse_state(sim_warehouse, robot_agents, tick, tasks_summary)

        # d. Check for completion condition
        if completed_tasks_count == len(task_manager.tasks):
            all_tasks_completed = True # Not used, but good for clarity
            print(f"\nAll {len(task_manager.tasks)} tasks completed at tick {tick}!")
            break

        if (pending_tasks_count == 0 and active_tasks_count == 0 and
            completed_tasks_count + failed_tasks_count == len(task_manager.tasks)):
            print(f"\nAll tasks concluded (completed or failed) at tick {tick}.")
            print(f"Summary: Completed: {completed_tasks_count}, Failed: {failed_tasks_count}")
            break

        if tick == max_ticks:
            print(f"\nSimulation ended after {max_ticks} ticks. Tasks completed: {completed_tasks_count}/{len(task_manager.tasks)}")

        time.sleep(simulation_delay)

    print("Simulation Finished.")

if __name__ == "__main__":
    main()
