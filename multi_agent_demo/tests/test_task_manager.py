import pytest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from multi_agent_demo.warehouse_environment import WarehouseGrid, Package
from multi_agent_demo.agents.robot_agent import RobotAgent
from multi_agent_demo.task_manager import TaskManager, Task

@pytest.fixture
def warehouse_for_tm():
    wh = WarehouseGrid(width=10, height=10)
    wh.shelves.clear()
    wh.obstacles.clear()
    wh.robot_positions.clear()
    return wh

@pytest.fixture
def robot1_tm(warehouse_for_tm):
    return RobotAgent(robot_id="R1", initial_location=(0,0), warehouse=warehouse_for_tm)

@pytest.fixture
def robot2_tm(warehouse_for_tm):
    return RobotAgent(robot_id="R2", initial_location=(0,1), warehouse=warehouse_for_tm)

@pytest.fixture
def package1():
    return Package(id="P1", location=(2,2), destination_zone=(8,8))

@pytest.fixture
def package2():
    return Package(id="P2", location=(3,3), destination_zone=(7,7))

def test_task_manager_initialization_no_packages(warehouse_for_tm, robot1_tm):
    tm = TaskManager(warehouse_for_tm, [robot1_tm])
    assert len(tm.tasks) == 0
    assert tm.robots["R1"] == robot1_tm

def test_task_manager_discover_initial_packages(warehouse_for_tm, robot1_tm, package1, package2):
    warehouse_for_tm.add_package_to_shelf(package1, package1.location)
    warehouse_for_tm.add_package_to_shelf(package2, package2.location)

    tm = TaskManager(warehouse_for_tm, [robot1_tm])
    assert len(tm.tasks) == 2
    assert "P1" in tm.tasks
    assert "P2" in tm.tasks
    assert tm.tasks["P1"].status == "pending"
    assert tm.tasks["P1"].pickup_location == (2,2)

def test_assign_task_to_idle_robot_success(warehouse_for_tm, robot1_tm, package1):
    warehouse_for_tm.add_package_to_shelf(package1, package1.location)
    tm = TaskManager(warehouse_for_tm, [robot1_tm])
    robot1_tm.state.status = "idle"
    tm.assign_task_to_robots()
    assert tm.tasks["P1"].status == "assigned_pickup"
    assert tm.tasks["P1"].assigned_robot_id == "R1"
    assert robot1_tm.state.current_task_package_id == "P1"
    assert robot1_tm.state.status == "moving_to_pickup"
    assert len(robot1_tm.state.path) > 0

def test_assign_task_no_idle_robots(warehouse_for_tm, robot1_tm, package1):
    warehouse_for_tm.add_package_to_shelf(package1, package1.location)
    tm = TaskManager(warehouse_for_tm, [robot1_tm])
    robot1_tm.state.status = "moving_to_charge"
    tm.assign_task_to_robots()
    assert tm.tasks["P1"].status == "pending"
    assert robot1_tm.state.current_task_package_id is None

def test_assign_task_no_pending_tasks(warehouse_for_tm, robot1_tm):
    tm = TaskManager(warehouse_for_tm, [robot1_tm])
    robot1_tm.state.status = "idle"
    tm.assign_task_to_robots()
    assert robot1_tm.state.current_task_package_id is None

def test_assign_task_robot_already_has_task(warehouse_for_tm, robot1_tm, package1):
    warehouse_for_tm.add_package_to_shelf(package1, package1.location)
    tm = TaskManager(warehouse_for_tm, [robot1_tm])
    robot1_tm.state.status = "idle"
    robot1_tm.state.current_task_package_id = "SomeOtherP"
    tm.assign_task_to_robots()
    assert tm.tasks["P1"].status == "pending"

def test_assign_task_pathfinding_fails(warehouse_for_tm, robot1_tm, package1):
    package1.location = (5,5) # Place package somewhere
    warehouse_for_tm.add_package_to_shelf(package1, package1.location)
    # Wall off the package completely
    for i in range(4, 7):
        warehouse_for_tm.obstacles.append((i, 4))
        warehouse_for_tm.obstacles.append((i, 6))
    warehouse_for_tm.obstacles.append((4, 5))
    warehouse_for_tm.obstacles.append((6, 5))

    tm = TaskManager(warehouse_for_tm, [robot1_tm]) # robot1_tm is at (0,0)
    robot1_tm.state.status = "idle"
    tm.assign_task_to_robots()
    assert tm.tasks["P1"].status == "pending"
    assert robot1_tm.state.current_task_package_id is None
    assert robot1_tm.state.status == "idle"

def test_update_task_status(warehouse_for_tm, robot1_tm, package1):
    warehouse_for_tm.add_package_to_shelf(package1, package1.location)
    tm = TaskManager(warehouse_for_tm, [robot1_tm])
    tm.update_task_status("P1", "assigned_pickup", "R1")
    assert tm.tasks["P1"].status == "assigned_pickup"
    assert tm.tasks["P1"].assigned_robot_id == "R1"
    tm.update_task_status("P1", "completed")
    assert tm.tasks["P1"].status == "completed"
    assert tm.tasks["P1"].assigned_robot_id is None
    tm.update_task_status("P_NON_EXIST", "completed")
