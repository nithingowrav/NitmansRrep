import pytest
from multi_agent_demo.agents.agent_one import AgentOne
from multi_agent_demo.agents.agent_two import AgentTwo

def test_agent_one_instantiation():
    try:
        agent = AgentOne()
        assert agent.name == "AgentOne"
    except Exception as e:
        pytest.fail(f"AgentOne instantiation or basic check failed: {e}")

def test_agent_two_instantiation():
    try:
        agent = AgentTwo()
        assert agent.name == "AgentTwo"
    except Exception as e:
        pytest.fail(f"AgentTwo instantiation or basic check failed: {e}")
