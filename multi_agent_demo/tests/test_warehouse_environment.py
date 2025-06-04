import pytest
import sys
import os

# Ensure project root (/app) is in path for imports like 'multi_agent_demo.warehouse_environment'
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from multi_agent_demo.warehouse_environment import WarehouseGrid, Package, RobotState

@pytest.fixture
def grid():
    # Create a fresh grid for each test, clearing defaults for more control
    g = WarehouseGrid(width=10, height=10)
    g.obstacles.clear()
    g.shelves.clear()
    g.drop_off_zones.clear()
    g.charging_stations.clear()
    g.robot_positions.clear()
    return g

@pytest.fixture
def grid_with_defaults(): # Fixture that keeps the __post_init__ defaults
    return WarehouseGrid(width=10, height=10)

@pytest.fixture
def sample_package():
    return Package(id="P1", location=(1,1), destination_zone=(8,8))

def test_warehouse_initialization_with_defaults(grid_with_defaults):
    assert grid_with_defaults.width == 10
    assert grid_with_defaults.height == 10
    # Default items from __post_init__
    assert (5,5) in grid_with_defaults.obstacles
    assert (8,8) in grid_with_defaults.drop_off_zones
    assert len(grid_with_defaults.shelves) > 0 # Default shelves with packages

def test_is_valid_location(grid):
    assert grid.is_valid_location(0,0)
    assert grid.is_valid_location(9,9)
    assert not grid.is_valid_location(-1,0)
    assert not grid.is_valid_location(10,5)

def test_is_obstacle(grid):
    grid.obstacles.append((3,3))
    assert grid.is_obstacle(3,3)
    assert not grid.is_obstacle(0,0)

def test_add_remove_package_from_shelf(grid, sample_package):
    shelf_loc = (2,2)
    assert grid.get_package_at_shelf(shelf_loc) is None # Should be empty from fixture 'grid'

    grid.add_package_to_shelf(sample_package, shelf_loc)
    assert grid.get_package_at_shelf(shelf_loc) == sample_package
    assert sample_package.location == shelf_loc
    assert shelf_loc in grid.shelves
    assert sample_package in grid.shelves[shelf_loc]

    grid.remove_package_from_shelf(sample_package, shelf_loc)
    assert grid.get_package_at_shelf(shelf_loc) is None
    assert shelf_loc not in grid.shelves # If last package removed, shelf key is deleted

def test_robot_position_management(grid):
    robot_id = "R_test"
    location1 = (1,1)
    location2 = (2,2)

    assert grid.get_robot_position(robot_id) is None
    grid.update_robot_position(robot_id, location1)
    assert grid.get_robot_position(robot_id) == location1
    assert grid.robot_positions[robot_id] == location1

    grid.update_robot_position(robot_id, location2)
    assert grid.get_robot_position(robot_id) == location2 # Now R_test is at location2

    # At this point, only R_test is at location2.
    # So, if R_other asks if location2 is occupied by someone other than R_other, it should be True (occupied by R_test).
    assert grid.is_cell_occupied_by_another_robot(location2, "R_other_that_is_not_R_test") is True
    # If R_test asks if location2 is occupied by someone other than R_test, it should be False.
    assert grid.is_cell_occupied_by_another_robot(location2, robot_id) is False

    # Now add another robot R_another to location2
    grid.update_robot_position("R_another", location2)
    # If R_test asks if location2 is occupied by someone other than R_test, it should be True (occupied by R_another).
    assert grid.is_cell_occupied_by_another_robot(location2, robot_id) is True
    # If R_another asks if location2 is occupied by someone other than R_another, it should be True (occupied by R_test).
    assert grid.is_cell_occupied_by_another_robot(location2, "R_another") is True
    # If a third robot R_yet_another asks, it should be true (occupied by R_test and R_another - though current logic returns true on first find)
    assert grid.is_cell_occupied_by_another_robot(location2, "R_yet_another") is True


    assert grid.is_cell_occupied_by_another_robot((0,0), robot_id) is False # (0,0) is empty

    grid.remove_robot_position(robot_id)
    assert grid.get_robot_position(robot_id) is None
    assert robot_id not in grid.robot_positions
    # R_other_at_loc2 is still at location2
    assert grid.is_cell_occupied_by_another_robot(location2, "R_some_other_robot") is True
