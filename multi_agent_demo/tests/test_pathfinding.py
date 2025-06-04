import pytest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from multi_agent_demo.warehouse_environment import WarehouseGrid
from multi_agent_demo.pathfinding import a_star_search

@pytest.fixture
def grid_10x10():
    g = WarehouseGrid(width=10, height=10)
    g.obstacles.clear()
    g.shelves.clear()
    g.drop_off_zones.clear()
    return g

def test_path_simple_direct(grid_10x10):
    path = a_star_search(grid_10x10, (0,0), (3,0))
    assert path == [(0,0), (1,0), (2,0), (3,0)]

def test_path_diagonal_movement_not_allowed_implicitly(grid_10x10):
    path = a_star_search(grid_10x10, (0,0), (1,1))
    assert len(path) == 3
    assert path[0] == (0,0)
    assert path[-1] == (1,1)
    assert (1,0) in path or (0,1) in path

def test_path_with_obstacle(grid_10x10):
    grid_10x10.obstacles.append((1,0))
    path = a_star_search(grid_10x10, (0,0), (2,0))
    # Example path could be [(0,0), (0,1), (1,1), (2,1), (2,0)] or other valid alternatives
    assert path is not None and len(path) > 0
    assert path[0] == (0,0)
    assert path[-1] == (2,0)
    assert (1,0) not in path # Must not go through obstacle

def test_no_path_available(grid_10x10):
    # Box in start completely
    grid_10x10.obstacles = [(1,0), (0,1)]
    if grid_10x10.width > 1 and grid_10x10.height > 1: # Also (1,1) if grid is larger
        grid_10x10.obstacles.append((1,1))
    path = a_star_search(grid_10x10, (0,0), (3,3))
    assert path == []

    # Wall across grid
    grid_10x10.obstacles = [(x,1) for x in range(grid_10x10.width)]
    path = a_star_search(grid_10x10, (0,0), (0,2))
    assert path == []

def test_start_or_goal_is_obstacle(grid_10x10):
    grid_10x10.obstacles.append((0,0))
    path = a_star_search(grid_10x10, (0,0), (2,2))
    assert path == []

    grid_10x10.obstacles = [(2,2)]
    path = a_star_search(grid_10x10, (0,0), (2,2))
    assert path == []

def test_start_or_goal_out_of_bounds(grid_10x10):
    path = a_star_search(grid_10x10, (-1,0), (2,2))
    assert path == []
    path = a_star_search(grid_10x10, (0,0), (10,2))
    assert path == []

def test_path_to_self(grid_10x10):
    path = a_star_search(grid_10x10, (0,0), (0,0))
    assert path == [(0,0)]

def test_path_larger_grid_scenario(grid_10x10):
    grid_10x10.obstacles = [(i, 5) for i in range(8)]
    path = a_star_search(grid_10x10, (0,0), (9,9))
    assert len(path) > 0
    assert path[0] == (0,0)
    assert path[-1] == (9,9)
    for i in range(8):
        assert (i,5) not in path
