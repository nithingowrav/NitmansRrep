# Smart Warehouse Robots Simulation 🤖📦

This project simulates a warehouse environment where multiple AI-powered robots are tasked with picking up packages from shelves and delivering them to designated drop-off zones.

## Features & AI Concepts Demonstrated

*   **Multi-Agent System:** Multiple robot agents operate concurrently in the shared warehouse environment.
*   **Pathfinding:** Robots use the A* (A-star) algorithm to find optimal paths to package pickup locations, drop-off zones, and charging stations, while navigating around obstacles. Implemented in `pathfinding.py`.
*   **Task Assignment:** A central `TaskManager` assigns pickup and delivery tasks to available robots. The current strategy is to assign tasks to idle robots that are not currently engaged in charging. Implemented in `task_manager.py`.
*   **Basic Collision Avoidance:** Robots can detect if their next intended cell is occupied by another robot and will wait if it is. This is a simple reactive mechanism.
*   **Resource Management (Battery & Charging):**
    *   Robots consume battery power for actions like moving.
    *   When battery levels get low, robots will autonomously seek out the nearest available charging station, navigate to it, and recharge.
*   **Simulation Engine:** A central simulation loop in `main.py` orchestrates the environment, agent actions, and task assignments over discrete time ticks.
*   **Text-Based Visualization:** The simulation provides a simple, real-time text-based representation of the warehouse grid, showing robot positions, packages, obstacles, and other key elements.

## Core Components

*   `main.py`: The main entry point to run the simulation. Contains the simulation loop and environment setup.
*   `warehouse_environment.py`: Defines the core data structures for the simulation, including `WarehouseGrid`, `Package`, and `RobotState`.
*   `agents/robot_agent.py`: Contains the `RobotAgent` class, defining the behavior, decision-making logic (including task execution and charging), and actions of individual robots.
*   `task_manager.py`: Implements the `TaskManager` responsible for discovering packages and assigning tasks to robots.
*   `pathfinding.py`: Provides the A* pathfinding algorithm used by robots.
*   `tests/`: Contains unit tests for various components (though currently focused on `RobotAgent` and basic structures).

## How to Run the Simulation

1.  **Prerequisites:**
    *   Python 3.x (developed with Python 3.7+)
    *   No external libraries are strictly required for the core simulation beyond the Python standard library. `pytest` is used for running tests if you choose to do so.

2.  **Navigate to the project directory:**
    Make sure your terminal is in the root of this repository (where this README is located). The `multi_agent_demo` directory should be directly inside this root.

3.  **Run the main simulation script from the repository root:**
    ```bash
    python multi_agent_demo/main.py
    ```
    Alternatively, navigate into `multi_agent_demo` and run `python main.py`.

4.  **Observe the Simulation:**
    The simulation will start, and you'll see a text-based representation of the warehouse updating in your terminal. It shows robot movements, status changes, and task progression. The simulation runs for a predefined number of ticks or until all tasks are completed.

## Future Enhancements (Ideas)

*   More sophisticated collision avoidance and deadlock resolution.
*   Dynamic task prioritization or re-assignment.
*   More complex robot behaviors (e.g., different robot types).
*   GUI-based visualization.
*   Persistent storage of simulation results or configurations.
