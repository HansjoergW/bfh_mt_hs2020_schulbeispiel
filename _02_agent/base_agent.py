from gym import Env

from abc import ABC, abstractmethod

import torch

class AgentBase(ABC):

    def __init__(self, env: Env, devicestr:str):
        self.env = env
        self.device = torch.device(devicestr)

    @abstractmethod
    def get_net(self):
        pass

    @abstractmethod
    def get_tgtnet(self):
        pass

    @abstractmethod
    def get_buffer(self):
        pass

    @abstractmethod
    def iteration_completed(self, iteration: int):
        pass
