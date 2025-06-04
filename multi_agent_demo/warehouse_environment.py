from dataclasses import dataclass, field
from typing import Tuple, List, Dict, Optional

@dataclass
class Package:
    id: str
    location: Tuple[int, int]  # Current (x, y) or shelf location
    destination_zone: Tuple[int, int] # Target drop-off zone (e.g., a specific cell)
    picked_by_robot: Optional[str] = None # ID of robot that picked it

@dataclass
class RobotState:
    id: str
    current_location: Tuple[int, int]
    status: str = "idle"  # e.g., "idle", "moving_to_pickup", "picking", "moving_to_dropoff", "dropping_off", "charging"
    battery_level: int = 100
    current_task_package_id: Optional[str] = None
    path: List[Tuple[int, int]] = field(default_factory=list) # Current path robot is following

@dataclass
class WarehouseGrid:
    width: int
    height: int
    shelves: Dict[Tuple[int, int], List[Package]] = field(default_factory=dict) # (x,y) -> list of packages
    drop_off_zones: List[Tuple[int, int]] = field(default_factory=list)
    charging_stations: List[Tuple[int, int]] = field(default_factory=list)
    obstacles: List[Tuple[int, int]] = field(default_factory=list)
    robot_positions: Dict[str, Tuple[int, int]] = field(default_factory=dict) # robot_id -> (x,y)

    def __post_init__(self):
        # Example: Add a few shelves, a drop-off zone, and an obstacle
        if not self.shelves:
            self.shelves[(2, 2)] = [Package(id="pkg1", location=(2,2), destination_zone=(8,8))]
            self.shelves[(2, 3)] = [Package(id="pkg2", location=(2,3), destination_zone=(8,8))]
        if not self.drop_off_zones:
            self.drop_off_zones.append((8, 8))
        if not self.obstacles:
            self.obstacles.append((5, 5)) # An obstacle at (5,5)

    def is_valid_location(self, x: int, y: int) -> bool:
        return 0 <= x < self.width and 0 <= y < self.height

    def is_obstacle(self, x: int, y: int) -> bool:
        return (x, y) in self.obstacles

    def get_package_at_shelf(self, location: Tuple[int, int]) -> Optional[Package]:
        # Return first package on shelf, or None if no packages or shelf doesn't exist
        packages_on_shelf = self.shelves.get(location, [])
        return packages_on_shelf[0] if packages_on_shelf else None

    def add_package_to_shelf(self, package: Package, location: Tuple[int, int]):
        if location not in self.shelves:
            self.shelves[location] = []
        self.shelves[location].append(package)
        package.location = location

    def remove_package_from_shelf(self, package: Package, location: Tuple[int, int]):
        if self.shelves.get(location) and package in self.shelves[location]:
            self.shelves[location].remove(package)
            if not self.shelves[location]: # Remove shelf if empty
                del self.shelves[location]

    # --- New methods for robot position tracking ---
    def update_robot_position(self, robot_id: str, new_location: Tuple[int, int]):
        self.robot_positions[robot_id] = new_location

    def remove_robot_position(self, robot_id: str):
        if robot_id in self.robot_positions:
            del self.robot_positions[robot_id]

    def get_robot_position(self, robot_id: str) -> Optional[Tuple[int, int]]:
        return self.robot_positions.get(robot_id)

    def is_cell_occupied_by_another_robot(self, location: Tuple[int, int], asking_robot_id: str) -> bool:
        for robot_id, pos in self.robot_positions.items():
            if robot_id != asking_robot_id and pos == location:
                return True
        return False

# Example usage (will not run during subtask, just for illustration)
if __name__ == '__main__':
    warehouse = WarehouseGrid(width=10, height=10)
    robot1_state = RobotState(id="R1", current_location=(0,0))

    print(f"Warehouse: {warehouse.width}x{warehouse.height}")
    print(f"Shelves: {warehouse.shelves}") # Will show default shelves
    print(f"Drop-off zones: {warehouse.drop_off_zones}") # Default drop-off
    print(f"Obstacles: {warehouse.obstacles}") # Default obstacle
    print(f"Robot1: {robot1_state}")

    pkg_on_shelf = warehouse.get_package_at_shelf((2,2))
    if pkg_on_shelf:
        print(f"Package at (2,2): {pkg_on_shelf.id}")

    # Test new robot position methods
    warehouse.update_robot_position("R1", (0,0))
    print(f"Robot R1 position: {warehouse.get_robot_position('R1')}")
    print(f"Is (0,0) occupied by another robot (asking R2)? {warehouse.is_cell_occupied_by_another_robot((0,0), 'R2')}") # True
    warehouse.remove_robot_position("R1")
    print(f"Robot R1 position after removal: {warehouse.get_robot_position('R1')}") # None
