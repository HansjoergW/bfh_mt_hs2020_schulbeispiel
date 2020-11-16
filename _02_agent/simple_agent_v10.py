import torch.nn as nn
from gym import Env
from _02_agent.base_agent import AgentBase

class SimpleNet(nn.Module):
    def __init__(self, obs_size, hidden_size, n_actions):
        super(SimpleNet, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_actions)
        )


    def forward(self, x):
        return self.net(x.float())

import gym
import ptan
import torch
from torch import device

class SimpleAgentV10(AgentBase):

    def __init__(self, env: Env,
                 devicestr:str,
                 gamma:float,
                 buffer_size:int,
                 target_net_sync:int = 1000,
                 eps_start:float = 1.0,
                 eps_final:float = 0.02,
                 eps_frames:int = 10**5):

        super(SimpleAgentV10, self).__init__(env, devicestr)

        self.target_net_sync = target_net_sync

        self.hiddensize = 128

        self.net = self._config_net()

        self.tgt_net = ptan.agent.TargetNet(self.net)

        self.selector = ptan.actions.EpsilonGreedyActionSelector(
            epsilon=1,
            selector=ptan.actions.ArgmaxActionSelector())

        self.epsilon_tracker = ptan.actions.EpsilonTracker(selector=self.selector, eps_start=eps_start, eps_final=eps_final, eps_frames=eps_frames)

        self.agent = agent = ptan.agent.DQNAgent(self.net, self.selector, device = self.device)

        self.exp_source = ptan.experience.ExperienceSourceFirstLast(self.env, self.agent, gamma=gamma)
        self.buffer = ptan.experience.ExperienceReplayBuffer(self.exp_source, buffer_size=buffer_size)


    def _config_net(self)-> nn.Module:
        return SimpleNet(self.env.observation_space.shape[0], self.hiddensize, self.env.action_space.n).to(self.device)


    def iteration_completed(self, iteration: int):

        self.epsilon_tracker.frame(iteration)

        if iteration % self.target_net_sync == 0:
            self.tgt_net.sync()

    def get_net(self):
        return self.net

    def get_tgtnet(self):
        return self.tgt_net

    def get_buffer(self):
        return self.buffer
