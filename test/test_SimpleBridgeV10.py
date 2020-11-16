from _01_environment.carenv_v10 import CarEnvV10
from _02_agent.simple_agent_v10 import SimpleAgentV10
from _03_bridge.simple_bridge_v10 import SimpleBridgeV10


from typing import Iterable, Tuple, List
import numpy as np

from ignite.engine import Engine

from ptan.experience import ExperienceFirstLast

def basic_simple_init(devicestr="cpu") -> SimpleBridgeV10:
    env = CarEnvV10()
    agent = SimpleAgentV10(env, devicestr, gamma=0.9, buffer_size=1000)
    bridge = SimpleBridgeV10(agent, gamma=0.9)

    return bridge

def simple_experiences() -> List[ExperienceFirstLast]:
    return [
        ExperienceFirstLast( np.array([0.0, 0.0, 0.0], dtype=np.float32), np.int64(0), 1.0,  np.array([0.5, 0.5, 0.5], dtype=np.float32)),
        ExperienceFirstLast( np.array([1.0, 1.0, 1.0], dtype=np.float32), np.int64(1), 2.0,  None)
    ]

def test_init_cuda():
    assert basic_simple_init("cuda") != None

def test_init_cpu():
    assert basic_simple_init("cpu") != None

def test_unpack():
    bridge = basic_simple_init()
    batch = simple_experiences()
    unpacked = bridge._unpack_batch(batch)
    # todo -Checks

def test_calc_loss():
    bridge = basic_simple_init()
    batch = simple_experiences()
    loss = bridge._calc_loss(batch)
    # todo -Checks

from ignite.engine import Engine

def test_process_batch(devicestr="cpu"):
    bridge = basic_simple_init(devicestr)
    batch = simple_experiences()
    bridge.process_batch(Engine(bridge.process_batch), batch)
    # todo -Checks

def test_batch_generator(devicestr="cpu"):
    # Test Iterator
    bridge = basic_simple_init(devicestr)
    a = bridge.batch_generator()
    nextbatch = next(a)
    assert len(nextbatch) == 32
