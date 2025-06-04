from typing import Tuple, Optional, List, TYPE_CHECKING
import sys
import os
# Correctly go up two levels to /app for imports like multi_agent_demo.warehouse_environment
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from multi_agent_demo.warehouse_environment import RobotState, Package, WarehouseGrid # type: ignore
from multi_agent_demo.pathfinding import a_star_search # type: ignore

if TYPE_CHECKING:
    from multi_agent_demo.task_manager import TaskManager # For type hinting

class RobotAgent:
    def __init__(self, robot_id: str, initial_location: Tuple[int, int], warehouse: WarehouseGrid):
        self.state = RobotState(id=robot_id, current_location=initial_location)
        self.warehouse = warehouse
        self.package_in_possession: Optional[Package] = None
        self.warehouse.update_robot_position(self.state.id, self.state.current_location)
        self._previous_semantic_status: Optional[str] = None

    def __str__(self):
        pkg_id = self.package_in_possession.id if self.package_in_possession else "None"
        return (f"Robot {self.state.id} at {self.state.current_location}, status: {self.state.status}, "
                f"battery: {self.state.battery_level}, task_pkg_id: {self.state.current_task_package_id}, "
                f"carrying: {pkg_id}, path_len: {len(self.state.path)})")

    def set_path(self, path: List[Tuple[int, int]]):
        self.state.path = list(path)

    def _can_move_to(self, target_location: Tuple[int, int]) -> bool:
        if not self.warehouse.is_valid_location(target_location[0], target_location[1]):
            return False
        if self.warehouse.is_obstacle(target_location[0], target_location[1]):
            return False
        if self.warehouse.is_cell_occupied_by_another_robot(target_location, self.state.id):
            return False
        return True

    def _move_one_step_on_path(self) -> bool:
        if not self.state.path:
            return False
        next_step = self.state.path[0]
        if not self._can_move_to(next_step):
            if self.state.status != "waiting_collision":
                 self._previous_semantic_status = self.state.status
            self.state.status = "waiting_collision"
            return False
        self.state.current_location = next_step
        self.warehouse.update_robot_position(self.state.id, self.state.current_location)
        self.state.battery_level = max(0, self.state.battery_level - 1)
        self.state.path.pop(0)
        return True

    def _direct_move_to(self, target_location: Tuple[int, int]):
        if not self._can_move_to(target_location):
            return
        self.state.current_location = target_location
        self.warehouse.update_robot_position(self.state.id, self.state.current_location)
        self.state.battery_level = max(0, self.state.battery_level - 1)

    def _attempt_pickup(self) -> bool:
        if not self.state.current_task_package_id:
            return False
        pkg_to_pickup: Optional[Package] = None
        shelf_packages = self.warehouse.shelves.get(self.state.current_location, [])
        for p in shelf_packages:
            if p.id == self.state.current_task_package_id:
                pkg_to_pickup = p
                break
        if pkg_to_pickup and self.state.current_location == pkg_to_pickup.location and not self.package_in_possession:
            self.package_in_possession = pkg_to_pickup
            pkg_to_pickup.picked_by_robot = self.state.id
            self.warehouse.remove_package_from_shelf(pkg_to_pickup, pkg_to_pickup.location)
            self.state.battery_level = max(0, self.state.battery_level - 2)
            print(f"Robot {self.state.id}: Picked up package {pkg_to_pickup.id} at {pkg_to_pickup.location}")
            return True
        return False

    def _attempt_dropoff(self) -> Optional[Package]: # Does NOT change status or task ID
        if self.package_in_possession and self.state.current_location == self.package_in_possession.destination_zone:
            dropped_package = self.package_in_possession
            # print(f"Robot {self.state.id}: Physically dropping package {dropped_package.id} at {self.state.current_location}")
            dropped_package.location = self.state.current_location
            dropped_package.picked_by_robot = None
            self.package_in_possession = None # Robot is no longer carrying it
            self.state.battery_level = max(0, self.state.battery_level - 1)
            return dropped_package
        return None

    def _charge_fully(self):
        if self.state.current_location in self.warehouse.charging_stations:
            self.state.battery_level = 100
            self.state.status = "idle" # Charging complete, robot is now idle
            print(f"Robot {self.state.id}: Charged to full at {self.state.current_location}.")
        else:
            print(f"Robot {self.state.id}: ERROR - _charge_fully called but not at a charging station.")
            self.state.status = "task_error_charging_failed"

    def run_step(self, task_manager: Optional['TaskManager'] = None):
        if self.state.battery_level <= 0:
            if self.state.status != "dead_battery":
                print(f"Robot {self.state.id}: Battery DEAD (level: {self.state.battery_level}). Status set to dead_battery.")
                self.state.status = "dead_battery"
            return

        if self.state.status == "waiting_collision":
            if self._previous_semantic_status:
                self.state.status = self._previous_semantic_status
                self._previous_semantic_status = None
            else:
                self.state.status = "idle"

        if self.state.battery_level < 30 and self.state.status not in ["moving_to_charge", "at_charging_station", "dead_battery"]:
            if self.state.status == "idle" or not self.state.current_task_package_id : # If idle or no task, prioritize charging
                 self.state.status = "needs_charging"


        if self.state.status == "needs_charging":
            if not self.warehouse.charging_stations:
                self.state.status = "task_error_no_charging_stations"
                return
            if self.state.current_location in self.warehouse.charging_stations:
                self.state.status = "at_charging_station"
            else:
                chosen_station = self.warehouse.charging_stations[0]
                path_to_charge = a_star_search(self.warehouse, self.state.current_location, chosen_station)
                if path_to_charge:
                    self.set_path(path_to_charge)
                    self.state.status = "moving_to_charge"
                else:
                    self.state.status = "task_error_no_path_to_charge"

        if self.state.status == "moving_to_charge":
            moved = self._move_one_step_on_path()
            if moved and not self.state.path:
                if self.state.current_location in self.warehouse.charging_stations:
                    self.state.status = "at_charging_station"
                else:
                    self.state.status = "task_error_charging_failed"
            return

        if self.state.status == "at_charging_station":
            self._charge_fully()
            return

        if self.state.status == "moving_to_pickup":
            moved = self._move_one_step_on_path()
            if moved and not self.state.path:
                self.state.status = "at_pickup_location"

        elif self.state.status == "at_pickup_location":
            if self._attempt_pickup():
                self.state.status = "carrying_package" # Intermediate status
                if task_manager and self.state.current_task_package_id:
                    task = task_manager.get_task_for_package(self.state.current_task_package_id)
                    if task:
                        path_to_dropoff = a_star_search(self.warehouse, self.state.current_location, task.dropoff_location)
                        if path_to_dropoff:
                            self.set_path(path_to_dropoff)
                            self.state.status = "moving_to_dropoff"
                            task_manager.update_task_status(task.package_id, "assigned_dropoff", self.state.id)
                        else:
                            self.state.status = "task_error_no_path_to_dropoff"
                            task_manager.update_task_status(task.package_id, "failed_no_path_to_dropoff", self.state.id)
                    else:
                        self.state.status = "task_error_internal"
            else:
                if task_manager and self.state.current_task_package_id:
                    task_manager.update_task_status(self.state.current_task_package_id, "failed_pickup", self.state.id)
                self.state.status = "idle"
                self.state.current_task_package_id = None

        elif self.state.status == "moving_to_dropoff":
            moved = self._move_one_step_on_path()
            if moved and not self.state.path:
                self.state.status = "at_dropoff_location"

        elif self.state.status == "at_dropoff_location":
            dropped_pkg = self._attempt_dropoff()
            if dropped_pkg:
                print(f"Robot {self.state.id}: Successfully dropped {dropped_pkg.id}. Task complete from robot's side.")
                if task_manager:
                    task_manager.update_task_status(dropped_pkg.id, "completed", self.state.id)
                self.state.current_task_package_id = None
                if self.state.battery_level < 30 and self.warehouse.charging_stations:
                    self.state.status = "needs_charging"
                else:
                    self.state.status = "idle"
            else:
                current_task_id_for_error = self.state.current_task_package_id
                self.state.status = "task_error_dropoff_failed"
                if task_manager and current_task_id_for_error:
                     task_manager.update_task_status(current_task_id_for_error, "failed_dropoff", self.state.id)
                self.state.current_task_package_id = None # Give up on this task after logging failure

        if self.state.status == "idle" and self.state.battery_level < 30 and self.warehouse.charging_stations:
             if not self.state.current_task_package_id: # Only go charge if truly idle (no task)
                self.state.status = "needs_charging"
