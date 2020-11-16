from _01_environment.carenv_v10 import CarEnvV10

def test_init():
    env = CarEnvV10()
    result = env.reset()

    assert result[0] == 0.0
    assert result[1] == env.distance
    assert result[2] == 0.0

def test_accelerate():
    env = CarEnvV10()
    env.reset()

    state, reward, done, _ = env.step(0)

    assert reward == 0
    assert done == False
    assert state[0] == 1.0
    assert state[1] == 999.0
    assert state[2] == 1.0

def test_break():
    env = CarEnvV10()
    env.reset()

    env.step(0)
    state, reward, done, _ = env.step(1)

    assert reward == -1
    assert done == False
    assert state[0] == 1.0
    assert state[1] == 999.0
    assert state[2] == 0.0

def test_keep_velocity():
    env = CarEnvV10()
    env.reset()
    env.step(0)

    state, reward, done, _ = env.step(2)

    assert reward == 0
    assert done == False
    assert state[0] == 2.0
    assert state[1] == 998.0
    assert state[2] == 1.0

def test_declutch():
    env = CarEnvV10()
    env.reset()
    env.step(0)

    state, reward, done, _ = env.step(3)

    assert reward == 0
    assert done == False
    assert abs(state[0] - 1.9)<0.001
    assert abs(state[1] - 998.1)<0.001
    assert abs(state[2] - 0.9)<0.001