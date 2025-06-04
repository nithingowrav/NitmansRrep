from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple

# Ensure imports from project root work
import sys
import os
# Corrected sys.path: Go up two levels if task_manager.py is in multi_agent_demo
# to access multi_agent_demo as a package (e.g., multi_agent_demo.warehouse_environment)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from multi_agent_demo.warehouse_environment import WarehouseGrid, Package, RobotState # type: ignore
from multi_agent_demo.agents.robot_agent import RobotAgent # type: ignore
from multi_agent_demo.pathfinding import a_star_search # type: ignore

@dataclass
class Task:
    package_id: str
    pickup_location: Tuple[int, int]
    dropoff_location: Tuple[int, int]
    status: str = "pending"  # pending, assigned_pickup, picked_up, assigned_dropoff, completed, failed_pickup, failed_no_path_to_pickup, failed_no_path_to_dropoff, failed_pkg_missing, failed_dropoff
    assigned_robot_id: Optional[str] = None

class TaskManager:
    def __init__(self, warehouse: WarehouseGrid, robots: List[RobotAgent]):
        self.warehouse = warehouse
        self.robots = {robot.state.id: robot for robot in robots} # Store robots in a dict by ID
        self.tasks: Dict[str, Task] = {}  # package_id -> Task
        self.discover_initial_packages()

    def discover_initial_packages(self):
        """Scans the warehouse shelves for packages and creates initial tasks."""
        print("TaskManager: Discovering initial packages...")
        for location, packages_on_shelf in self.warehouse.shelves.items():
            for package in packages_on_shelf:
                if package.id not in self.tasks: # Avoid re-adding if called multiple times
                    self.tasks[package.id] = Task(
                        package_id=package.id,
                        pickup_location=package.location,
                        dropoff_location=package.destination_zone
                    )
                    # print(f"TaskManager: Discovered package {package.id} at {package.location} for {package.destination_zone}. Created task.")
        print(f"TaskManager: Total tasks created: {len(self.tasks)}")


    def get_task_for_package(self, package_id: str) -> Optional[Task]:
        return self.tasks.get(package_id)

    def update_task_status(self, package_id: str, new_status: str, robot_id: Optional[str] = None):
        task = self.get_task_for_package(package_id)
        if task:
            task.status = new_status
            if robot_id: # Assign robot if ID is provided
                task.assigned_robot_id = robot_id
            if new_status == "completed" or "failed" in new_status: # Check if task is terminal
                task.assigned_robot_id = None # Robot is free from this task

            # print(f"TaskManager: Updated task for package {package_id} to {new_status} (Robot: {robot_id or 'N/A'})")
        # else:
            # print(f"TaskManager Error: Could not find task for package {package_id} to update status.")

    def assign_task_to_robots(self):
        """Assigns pending tasks to idle robots."""
        # print(f"TaskManager: Attempting to assign tasks. Current tasks: {[(t.package_id, t.status) for t in self.tasks.values()]}")
        for robot_id, robot in self.robots.items():
            # print(f"TaskManager: Checking robot {robot.state.id}, status: {robot.state.status}, task: {robot.state.current_task_package_id}")
            if robot.state.status == "idle" and not robot.state.current_task_package_id and not robot.package_in_possession:
                # Find a pending task for this idle robot
                for package_id, task in self.tasks.items():
                    if task.status == "pending":
                        # print(f"TaskManager: Considering task for package {package_id} for robot {robot.state.id}.")

                        path_to_pickup = a_star_search(self.warehouse, robot.state.current_location, task.pickup_location)

                        if not path_to_pickup:
                            print(f"TaskManager Info: Robot {robot.state.id} could not find path to pickup {task.pickup_location} for package {package_id}. Task remains pending.")
                            # self.update_task_status(package_id, "failed_no_path_to_pickup") # Mark task as failed for this attempt
                            continue

                        print(f"TaskManager: Assigning task for package {package_id} to robot {robot.state.id}. Path: {path_to_pickup}")
                        robot.state.current_task_package_id = package_id
                        robot.set_path(path_to_pickup)
                        robot.state.status = "moving_to_pickup"

                        self.update_task_status(package_id, "assigned_pickup", robot.state.id)
                        break # Robot assigned, move to next robot
                # else:
                    # print(f"TaskManager: No pending tasks available for robot {robot.state.id}")
            # Additional conditions for logging why a robot isn't assigned a task
            # elif robot.state.status != "idle":
                # print(f"TaskManager: Robot {robot.state.id} is not idle (status: {robot.state.status}).")
            # elif robot.state.current_task_package_id:
                # print(f"TaskManager: Robot {robot.state.id} already has task {robot.state.current_task_package_id}.")
            # elif robot.package_in_possession:
                # print(f"TaskManager: Robot {robot.state.id} is already carrying package {robot.package_in_possession.id}.")


if __name__ == '__main__':
    # Correct sys.path for __main__ execution
    # If task_manager.py is in multi_agent_demo, and multi_agent_demo is the package root for imports
    current_dir_name = os.path.basename(os.getcwd())
    if current_dir_name == 'multi_agent_demo': # If run from parent dir (e.g. /app/multi_agent_demo)
         # No change needed if multi_agent_demo is already in path due to how pytest runs or if /app is in path
         pass
    elif os.path.basename(os.path.dirname(os.path.abspath(__file__))) == 'multi_agent_demo': # If in /app
        # This means multi_agent_demo is a subdir of current_dir, which is fine.
        pass
    else: # Fallback: if run from somewhere unexpected, try to ensure /app is in path
        # This assumes the script is /app/multi_agent_demo/task_manager.py
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if project_root not in sys.path:
            sys.path.insert(0, project_root)

    # Re-import with corrected path for main, if necessary (though top-level imports should be fine now)
    from multi_agent_demo.warehouse_environment import WarehouseGrid, Package
    from multi_agent_demo.agents.robot_agent import RobotAgent
    from multi_agent_demo.pathfinding import a_star_search


    # Example Usage
    warehouse = WarehouseGrid(width=10, height=10)
    warehouse.shelves.clear() # Clear default shelves from WarehouseGrid's __post_init__
    warehouse.obstacles = [(5,5)] # Set specific obstacles

    pkg1 = Package(id="P1", location=(2,2), destination_zone=(8,8))
    pkg2 = Package(id="P2", location=(3,3), destination_zone=(8,8)) # Different destination to test path logic
    warehouse.add_package_to_shelf(pkg1, (2,2))
    warehouse.add_package_to_shelf(pkg2, (3,3))

    robot_agent1 = RobotAgent(robot_id="R1", initial_location=(0,0), warehouse=warehouse)
    robot_agent2 = RobotAgent(robot_id="R2", initial_location=(0,1), warehouse=warehouse)

    robots_list = [robot_agent1, robot_agent2]

    task_manager = TaskManager(warehouse, robots_list) # This will call discover_initial_packages

    print(f"\nInitial Robot1: {robot_agent1}")
    print(f"Initial Robot2: {robot_agent2}")
    for task_id, task_obj in task_manager.tasks.items():
        print(f"Initial Task {task_id}: {task_obj}")

    print("\n--- Assigning tasks (Round 1) ---")
    task_manager.assign_task_to_robots()

    print(f"\nRobot1 after assignment: {robot_agent1}")
    print(f"Robot2 after assignment: {robot_agent2}")
    for task_id, task_obj in task_manager.tasks.items():
        print(f"Task {task_id} after assignment: {task_obj}")

    # Simulate R1 run_step until it reaches pickup or path ends
    print("\n--- Simulating R1 moving to P1 ---")
    if robot_agent1.state.current_task_package_id == "P1":
        loop_count = 0
        while robot_agent1.state.status == "moving_to_pickup" and loop_count < 20: # Safety break
            print(f"R1 run_step: {robot_agent1}")
            robot_agent1.run_step(task_manager) # Pass task_manager reference
            loop_count += 1
        print(f"R1 after moving towards P1: {robot_agent1}")

        # Simulate R1 picking up P1
        if robot_agent1.state.status == "at_pickup_location":
            print(f"\nR1 at pickup, attempting pickup of P1...")
            robot_agent1.run_step(task_manager) # This should trigger _attempt_pickup
            print(f"R1 after pickup attempt: {robot_agent1}")
            print(f"Task P1 status: {task_manager.tasks.get('P1')}")

            # Simulate R1 moving to dropoff
            if robot_agent1.state.status == "moving_to_dropoff":
                print("\n--- Simulating R1 moving to dropoff P1 ---")
                loop_count = 0
                while robot_agent1.state.status == "moving_to_dropoff" and loop_count < 20:
                    print(f"R1 run_step: {robot_agent1}")
                    robot_agent1.run_step(task_manager)
                    loop_count += 1
                print(f"R1 after moving towards dropoff: {robot_agent1}")

                if robot_agent1.state.status == "at_dropoff_location":
                    print(f"\nR1 at dropoff, attempting dropoff of P1...")
                    robot_agent1.run_step(task_manager) # This should trigger _attempt_dropoff
                    print(f"R1 after dropoff attempt: {robot_agent1}")
                    print(f"Task P1 status: {task_manager.tasks.get('P1')}")
    else:
        print("R1 did not get assigned P1 as expected.")


    # Try assigning tasks again (R2 should get P2 if available)
    print("\n--- Assigning tasks (Round 2, for R2) ---")
    task_manager.assign_task_to_robots() # R1 might be busy or done, R2 should be idle
    print(f"\nRobot1 final: {robot_agent1}")
    print(f"Robot2 final: {robot_agent2}")
    for task_id, task_obj in task_manager.tasks.items():
        print(f"Task {task_id} final: {task_obj}")
