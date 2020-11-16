from _02_agent.simple_agent_v10 import SimpleAgentV10
from _01_environment.carenv_v10 import CarEnvV10

GAMMA = 0.9
REPLAY_SIZE = 1000

def test_simpleagent_cpu():
    print("test cpu")
    env = CarEnvV10()
    agent = SimpleAgentV10(env, "cpu", gamma=GAMMA, buffer_size=REPLAY_SIZE)

def test_simpleagent_cuda():
    print("test cuda")
    env = CarEnvV10()
    agent = SimpleAgentV10(env, "cuda", gamma=GAMMA, buffer_size=REPLAY_SIZE)