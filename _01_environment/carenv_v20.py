import gym
from gym.spaces import Tuple, Discrete, Box
import numpy as np
import random

class CarEnvV20(gym.Env):
    """
    Actions:
    0: accelerate
    1: break
    2: keep velocity
    3: declutch

    Observationspace
    1: distance travelled in meters
    2: distance to go in meters
    3: current velocity

    Rewards:
    - standing still     -1
    - goal reached      100
    - overshoot:       current velocity^2
    - closer to target    0
    - Option: not reaching goal within time: reward = - int(distance_left)
    - Option: per used 1 Unit of Energy -1

    Options:
    - Penalty for using energy
    - Stochastic environment / random environment -> accelaration and deceleration have a random part
    - limit then number of possible steps. if target is not reached after a certain amount of steps
    - penalty of -1 point for every time step that passes
    """
    def __init__(self, mode_energy_penalty:bool = False, mode_random:bool = False, mode_limit_steps:bool = False,
                 mode_time_penalty:bool = False, mode_reward:str = "lin"):
        super(CarEnvV20, self).__init__()

        self.mode_energy_penalty = mode_energy_penalty
        self.mode_random = mode_random
        self.mode_limit_steps = mode_limit_steps
        self.mode_time_penalty = mode_time_penalty
        self.mode_reward = mode_reward

        # define distance
        self.distance: float = 1000.0
        self.maxspeed: float = 35.0 # maxspeed 35/ms
        self.velocityenergy_unit: float = 10.0 # 1 second at the speed of velocityenergy_unit uses 1 energy unit
        self.accelerationenergy_factor: float = 0.05 # acceleration uses an extra amount of engergy
        self.max_timesteps = int(self.distance)

        self.currentposition: float = 0.0
        self.currentvelocity: float = 0.0
        self.usedenergy: float = 0.0
        self.is_done: bool = False


        # definition of observation value array
        low = np.array([0.0,
                        0.0,
                        0.0],
                       dtype=np.float32
                       )

        high = np.array([self.distance,
                         self.distance,
                         self.maxspeed],
                        dtype=np.float32
                        )

        self.observation_space = Box(low, high, dtype=np.float32)

        self.action_space = Discrete(n=4)

        self.step_index = 0


    def reset(self):
        self.currentposition = 0.0
        self.currentvelocity = 0.0
        self.usedenergy = 0.0
        self.is_done = False
        self.step_index = 0

        return self._calculate_state()

    def _calculate_state(self):
        return np.array([self.currentposition,
                         self.distance - self.currentposition,
                         self.currentvelocity
                         ], dtype=np.float32)


    def _set_new_velocity(self, acceleration:float):

        # in case of randommode, accelleration and decelaration have a uniform random part
        if self.mode_random:
            acceleration += acceleration * random.uniform(-0.2, +0.2)

        self.currentvelocity = max(0, min(self.maxspeed, self.currentvelocity + acceleration))


    def step(self, action):
        zero_state = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        if self.is_done:
            return zero_state, 0, self.is_done

        err_msg = "%r (%s) invalid" % (action, type(action))
        assert self.action_space.contains(action), err_msg

        energy_for_step:float = 0.0
        reward = 0

        if action==0:
            # accelerate
            self._set_new_velocity(1.0)
            energy_for_step = (self.currentvelocity / self.velocityenergy_unit) * (1 + self.accelerationenergy_factor)

        if action==1:
            # break
            self._set_new_velocity(-1.0)
            energy_for_step = 0.0 # using the breaks doesn't need energy

        if action==2:
            # keep velocity
            energy_for_step = (self.currentvelocity / self.velocityenergy_unit)

        if action==3:
            # declutch
            self._set_new_velocity(-0.1)
            energy_for_step = 0.0 # declutch doesn't use energy

        if self.mode_random:
            self.currentvelocity += self.currentvelocity * random.uniform(-0.2, +0.2)

        self.currentposition += self.currentvelocity
        self.step_index      += 1
        self.usedenergy      += energy_for_step

        goal_reached = bool(self.currentposition > self.distance -1
                            and self.currentposition <= self.distance
                            and abs(self.currentvelocity - 0.0)<0.00001)

        overshoot = self.currentposition > self.distance

        timeup = self.mode_limit_steps and (self.step_index >= self.max_timesteps)

        if overshoot:
            if self.mode_reward == "lin":
                reward = -int(self.currentvelocity * 30)
            if self.mode_reward == "exp":
                reward = -int(self.currentvelocity ** 2)
            if self.mode_reward == "log":
                reward = 1.2**(self.maxspeed - self.currentvelocity) - 1.2**self.maxspeed
            if self.mode_reward == "log2":
                reward = 1.22**(self.maxspeed - self.currentvelocity*.5) - 1.22**self.maxspeed

            self.is_done = True
            return zero_state, reward, self.is_done, {}

        if timeup:
            if self.mode_reward == "lin":
                reward = -int(self.distance - self.currentposition)
            if self.mode_reward == "exp":
                reward = -int(((self.distance - self.currentposition) / 30) ** 2)
            if self.mode_reward == "log":
                reward = 1.2**(self.currentposition/30) - 1.2**(self.distance/30)
            if self.mode_reward == "log2":
                reward = 1.22**(self.distance/30 - ((self.distance - self.currentposition)/30)*0.5) - 1.22**(self.distance/30)

            self.is_done = True
            return zero_state, reward, self.is_done, {}

        if goal_reached:
            self.is_done = True
            reward = 100

            if self.mode_energy_penalty:
                reward -= self.usedenergy

            return zero_state, reward, self.is_done, {}

        if self.mode_time_penalty:
            reward += -1

        if abs(self.currentvelocity - 0.0)<0.00001:
            reward += -1

        return self._calculate_state(), reward, self.is_done, {}