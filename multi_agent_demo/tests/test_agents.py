import pytest
import sys
import os

# Ensure project root (/app) is in path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from multi_agent_demo.agents.robot_agent import RobotAgent
from multi_agent_demo.warehouse_environment import WarehouseGrid, Package
from multi_agent_demo.task_manager import TaskManager # Added for the new test

@pytest.fixture
def warehouse():
    wh = WarehouseGrid(width=10, height=10)
    wh.robot_positions.clear()
    wh.shelves.clear()
    wh.obstacles.clear()
    wh.drop_off_zones.clear()
    wh.charging_stations.clear()
    # Re-add default items if necessary for specific tests, or use a separate fixture
    # For a clean slate, we ensure __post_init__ doesn't add its own defaults by checking if lists are empty.
    # However, the current WarehouseGrid.__post_init__ adds items if the lists are empty.
    # So, to truly control, either WarehouseGrid changes or tests accept/clear defaults.
    # Let's stick to a clean warehouse for most agent tests.
    return wh

@pytest.fixture
def robot1(warehouse) -> RobotAgent:
    r1 = RobotAgent(robot_id="R1", initial_location=(0,0), warehouse=warehouse)
    return r1

@pytest.fixture
def robot2(warehouse) -> RobotAgent:
    r2 = RobotAgent(robot_id="R2", initial_location=(1,0), warehouse=warehouse)
    return r2

@pytest.fixture # Define package1 fixture for use in the new test
def package1(warehouse): # warehouse fixture can be used if needed for setup
    pkg = Package(id="P1", location=(0,0), destination_zone=(8,8))
    # warehouse.add_package_to_shelf(pkg, pkg.location) # Add to warehouse if test needs it pre-added
    return pkg

# --- Corrected Tests ---
def test_robot_agent_instantiation(robot1: RobotAgent, warehouse: WarehouseGrid):
    assert robot1.state.id == "R1"
    assert robot1.state.current_location == (0,0)
    assert robot1.state.status == "idle"
    assert warehouse.get_robot_position("R1") == (0,0)

def test_robot_direct_move_to_valid(robot1: RobotAgent, warehouse: WarehouseGrid):
    robot1._direct_move_to((1,1))
    assert robot1.state.current_location == (1,1)
    assert warehouse.get_robot_position(robot1.state.id) == (1,1)
    assert robot1.state.battery_level < 100

def test_robot_direct_move_to_obstacle(robot1: RobotAgent, warehouse: WarehouseGrid):
    warehouse.obstacles.append((2,2))
    initial_location = robot1.state.current_location
    robot1._direct_move_to((2,2))
    assert robot1.state.current_location == initial_location

def test_robot_attempt_pickup(robot1: RobotAgent, warehouse: WarehouseGrid, package1: Package):
    # Robot1 at (0,0), package1 also at (0,0) for this test
    package1.location = (0,0)
    robot1.state.current_location = (0,0) # Ensure robot is at package location
    warehouse.update_robot_position(robot1.state.id, (0,0))
    warehouse.add_package_to_shelf(package1, package1.location)
    robot1.state.current_task_package_id = package1.id # Agent needs to know which package to pick

    assert robot1._attempt_pickup() is True
    assert robot1.package_in_possession == package1
    assert package1.picked_by_robot == robot1.state.id
    assert warehouse.get_package_at_shelf(package1.location) is None

def test_robot_attempt_dropoff(robot1: RobotAgent, warehouse: WarehouseGrid, package1: Package):
    pkg_dest = (5,5)
    package1.destination_zone = pkg_dest # Ensure package1 has this destination
    robot1.package_in_possession = package1
    package1.picked_by_robot = robot1.state.id
    robot1.state.current_location = pkg_dest
    warehouse.update_robot_position(robot1.state.id, pkg_dest)
    # current_task_package_id is not directly used by _attempt_dropoff but good for consistency
    robot1.state.current_task_package_id = package1.id

    dropped = robot1._attempt_dropoff()
    assert dropped == package1
    assert robot1.package_in_possession is None
    # _attempt_dropoff itself does not set status to idle anymore, run_step does.
    # So we don't check robot1.state.status here.

def test_robot_charge_fully(robot1: RobotAgent, warehouse: WarehouseGrid):
    charge_loc = (3,3)
    warehouse.charging_stations.append(charge_loc)
    robot1.state.battery_level = 20
    robot1.state.current_location = charge_loc # Simulate robot is at charging station
    warehouse.update_robot_position(robot1.state.id, charge_loc)

    robot1._charge_fully() # Call the correct method
    assert robot1.state.battery_level == 100
    assert robot1.state.status == "idle" # _charge_fully now sets to idle

def test_robot_set_path(robot1: RobotAgent):
    path = [(1,0), (2,0), (2,1)]
    robot1.set_path(path)
    assert robot1.state.path == path
    # robot1.state.status is NOT changed by set_path alone anymore
    assert robot1.state.status == "idle" # Should remain in its prior status

def test_robot_move_one_step_on_path_successfully(robot1: RobotAgent, warehouse: WarehouseGrid):
    robot1.set_path([(1,0), (2,0)])
    robot1.state.status = "moving_to_pickup" # Set a semantic status for context

    assert robot1._move_one_step_on_path() is True
    assert robot1.state.current_location == (1,0)
    assert warehouse.get_robot_position(robot1.state.id) == (1,0)
    assert robot1.state.path == [(2,0)]
    assert robot1.state.status == "moving_to_pickup" # Should retain semantic status if move is part of it

    assert robot1._move_one_step_on_path() is True
    assert robot1.state.current_location == (2,0)
    assert warehouse.get_robot_position(robot1.state.id) == (2,0)
    assert not robot1.state.path
    assert robot1.state.status == "moving_to_pickup" # Status isn't changed by _move_one_step alone

def test_robot_move_one_step_on_path_hits_obstacle(robot1: RobotAgent, warehouse: WarehouseGrid):
    obstacle_loc = (1,0)
    warehouse.obstacles.append(obstacle_loc)
    robot1.set_path([obstacle_loc, (2,0)])
    robot1.state.status = "moving_to_pickup"

    assert robot1._move_one_step_on_path() is False
    assert robot1.state.current_location == (0,0)
    assert robot1.state.path == [obstacle_loc, (2,0)]
    assert robot1.state.status == "waiting_collision"

def test_collision_avoidance_waits(robot1: RobotAgent, robot2: RobotAgent, warehouse: WarehouseGrid):
    # R1 at (0,0), R2 at (1,0)
    robot1.set_path([(1,0)])
    robot1.state.status = "moving_to_pickup" # Set semantic status

    moved = robot1._move_one_step_on_path()
    assert moved is False
    assert robot1.state.current_location == (0,0)
    assert robot1.state.status == "waiting_collision"
    assert warehouse.get_robot_position(robot1.state.id) == (0,0)

def test_collision_avoidance_proceeds_after_clear(robot1: RobotAgent, robot2: RobotAgent, warehouse: WarehouseGrid):
    # R1 at (0,0), R2 at (1,0)
    robot1.set_path([(1,0)])
    robot1.state.status = "moving_to_pickup" # R1's intent
    assert not robot1._move_one_step_on_path() # R1 tries to move to (1,0), fails, status becomes waiting_collision
    assert robot1.state.status == "waiting_collision"

    # R2 moves away from (1,0) to (2,0)
    robot2.set_path([(2,0)])
    robot2.state.status = "moving_to_pickup" # R2's intent
    assert robot2._move_one_step_on_path()
    assert robot2.state.current_location == (2,0)
    assert warehouse.get_robot_position(robot2.state.id) == (2,0)

    # R1 tries to move to (1,0) again
    # Restore R1's semantic status before calling _move_one_step_on_path
    # In full simulation, run_step would handle this restoration.
    robot1.state.status = "moving_to_pickup"
    assert robot1._move_one_step_on_path() # Should succeed now
    assert robot1.state.current_location == (1,0)
    assert warehouse.get_robot_position(robot1.state.id) == (1,0)

# --- run_step Tests ---
def test_run_step_low_battery_idle_needs_charging(robot1: RobotAgent, warehouse: WarehouseGrid):
    robot1.state.battery_level = 25
    robot1.state.status = "idle"
    warehouse.charging_stations.append((5,5)) # Ensure a station exists

    robot1.run_step()
    # After one run_step, if idle and low battery, it should try to find a path and set status to moving_to_charge or error.
    # If pathfinding is successful (default grid is open), it should be moving_to_charge.
    assert robot1.state.status == "moving_to_charge" # Or task_error_no_path_to_charge if pathing fails

def test_run_step_needs_charging_at_station_charges(robot1: RobotAgent, warehouse: WarehouseGrid):
    charge_loc = (5,5)
    robot1.state.battery_level = 25
    robot1.state.current_location = charge_loc
    warehouse.update_robot_position(robot1.state.id, charge_loc)
    warehouse.charging_stations.append(charge_loc)
    robot1.state.status = "needs_charging" # Start in this state

    robot1.run_step() # Should transition to at_charging_station, then _charge_fully, then idle, all in one step.
    assert robot1.state.battery_level == 100
    assert robot1.state.status == "idle"

def test_run_step_needs_charging_has_path_moves(robot1: RobotAgent, warehouse: WarehouseGrid):
    charge_loc = (5,5)
    warehouse.charging_stations.append(charge_loc) # Ensure station exists
    robot1.state.battery_level = 25
    robot1.state.status = "needs_charging"
    # Path is set by run_step itself if not already charging.
    # To test this part, we assume path is already set to simulate "moving_to_charge"
    # The previous test, test_run_step_low_battery_idle_needs_charging, covers path generation part.
    # Here, let's test if it follows a path once status is moving_to_charge

    # Path set by internal logic of needs_charging if no path
    robot1.run_step() # First run_step: sets status to moving_to_charge and sets path
    assert robot1.state.status == "moving_to_charge"
    assert len(robot1.state.path) > 0

    # Second run_step: should move along the path
    if robot1.state.path: # If path was actually found
        expected_next_loc = robot1.state.path[0]
        robot1.run_step()
        assert robot1.state.current_location == expected_next_loc
        assert robot1.state.status == "moving_to_charge" # Continues moving
    else:
        # This case can happen if path to charger was not found in the first place
        # The status would be task_error_no_path_to_charge from the first run_step call
        assert robot1.state.status == "task_error_no_path_to_charge"


def test_run_step_moving_on_path_continues_move(robot1: RobotAgent, warehouse: WarehouseGrid):
    path = [(1,0), (2,0)]
    robot1.set_path(path)
    robot1.state.status = "moving_to_pickup" # Set initial semantic status

    robot1.run_step()
    assert robot1.state.current_location == (1,0)
    assert robot1.state.path == [(2,0)]
    assert robot1.state.status == "moving_to_pickup"

    robot1.run_step()
    assert robot1.state.current_location == (2,0)
    assert not robot1.state.path
    assert robot1.state.status == "at_pickup_location" # Transitions to this when path ends

def test_run_step_needs_charging_no_stations(robot1: RobotAgent, warehouse: WarehouseGrid):
    warehouse.charging_stations = [] # NO stations
    robot1.state.battery_level = 25
    robot1.state.status = "idle" # Start from idle to trigger needs_charging transition

    robot1.run_step() # First step, status becomes "needs_charging" internally, then immediately "task_error_no_charging_stations"
    assert robot1.state.status == "task_error_no_charging_stations"


def test_path_is_copied_in_set_path(robot1: RobotAgent):
    original_path = [(1,0), (2,0)]
    robot1.set_path(original_path)
    assert robot1.state.path is not original_path
    assert robot1.state.path == original_path

# --- New test for run_step interaction with TaskManager for dropoff path ---
def test_robot_run_step_pickup_and_sets_path_to_dropoff(robot1: RobotAgent, warehouse: WarehouseGrid, package1: Package):
    package1.location = (0,0)
    warehouse.add_package_to_shelf(package1, package1.location)
    tm = TaskManager(warehouse, [robot1])
    robot1.state.current_task_package_id = package1.id
    robot1.state.status = "at_pickup_location"

    robot1.run_step(task_manager=tm)

    assert robot1.package_in_possession == package1
    assert robot1.state.status == "moving_to_dropoff"
    assert tm.tasks[package1.id].status == "assigned_dropoff"
    assert len(robot1.state.path) > 0
    assert robot1.state.path[-1] == package1.destination_zone
