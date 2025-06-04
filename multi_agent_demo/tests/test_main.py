import pytest
from multi_agent_demo import main as app_main # Assuming main.py is in multi_agent_demo

def test_main_function_callable():
    # Simple test to ensure the main function can be called
    try:
        app_main.main()
    except Exception as e:
        pytest.fail(f"main.main() raised an exception: {e}")
