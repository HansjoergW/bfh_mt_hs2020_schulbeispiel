from _01_environment.carenv_v10 import CarEnvV10
from _02_agent.simple_agent_v10 import SimpleAgentV10
from _03_bridge.simple_bridge_v10 import SimpleBridgeV10
from _04_loopcontrol.loop_control_v10 import LoopControlV10

def basic_init_bridge() -> SimpleBridgeV10:
    env = CarEnvV10()
    agent = SimpleAgentV10(env, "cpu", gamma=0.9, buffer_size=1000)
    bridge = SimpleBridgeV10(agent, gamma=0.9)

    return bridge

def test_basic_init():
    bridge = basic_init_bridge()
    LoopControlV10(bridge, "dummy")
