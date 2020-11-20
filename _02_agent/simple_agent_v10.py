from _02_agent.base_agent import AgentBase

import torch.nn as nn
from gym import Env

import ptan


def create_net(input_size:int, output_size:int, hidden_layers:int, hidden_size: int) -> nn.Sequential:
    modules = []
    modules.append(nn.Linear(input_size, hidden_size))
    modules.append(nn.ReLU())

    for i in range(0, hidden_layers-1):
        modules.append(nn.Linear(hidden_size, hidden_size))
        modules.append(nn.ReLU())

    modules.append(nn.Linear(hidden_size, output_size))

    return nn.Sequential(*modules)


class SimpleNet(nn.Module):
    def __init__(self, obs_size, n_actions, hidden_layers, hidden_size):
        super(SimpleNet, self).__init__()
        self.net = create_net(obs_size, n_actions, hidden_layers, hidden_size)

    def forward(self, x):
        return self.net(x.float())


class DuelingNet(nn.Module):
    # gemäss https://towardsdatascience.com/dueling-deep-q-networks-81ffab672751

    def __init__(self, obs_size, n_actions, hidden_layers, hidden_size):
        super(DuelingNet, self).__init__()

        self.feauture_layer = create_net(obs_size, hidden_size, hidden_layers, hidden_size)

        self.value_stream = create_net(hidden_size, 1, hidden_layers, hidden_size)

        self.advantage_stream = create_net(hidden_size, n_actions, hidden_layers, hidden_size)


    def forward(self, x):
        features = self.feauture_layer(x)
        values = self.value_stream(features)
        advantages = self.advantage_stream(features)
        qvals = values + (advantages - advantages.mean())
        return qvals


class CombinedReplayBuffer(ptan.experience.ExperienceReplayBuffer):
    # https://arxiv.org/pdf/1712.01275.pdf

    def __init__(self, experience_source, buffer_size):
        super(CombinedReplayBuffer, self).__init__(experience_source, buffer_size)
        self.last_added = None

    def _add(self, sample):
        self.last_added = sample
        super()._add(sample)

    def sample(self, batch_size):
        batch = super().sample(batch_size)
        if self.last_added is not None:
            batch[0] = self.last_added

        return batch


# class ArgmaxActionSelector(ptan.actions.ActionSelector):
#     """
#     Selects actions using argmax
#     """
#     def __init__(self, env: Env, parent_selector: ptan.actions.ActionSelector):
#         self.env = env
#         self.parent_selector = parent_selector
#
#
#     def __call__(self, scores):
#
#         return self.parent_selector(scores)

class SimpleAgentV10(AgentBase):

    def __init__(self, env: Env,
                 devicestr:str,
                 gamma:float,
                 buffer_size:int,
                 target_net_sync:int = 1000,
                 eps_start:float = 1.0,
                 eps_final:float = 0.02,
                 eps_frames:int = 10**5,
                 hidden_size:int = 128,
                 hidden_layers:int = 1,
                 dueling_network:bool = False,
                 steps_count:int = 1,
                 use_combined_replay_buffer:bool = False):

        super(SimpleAgentV10, self).__init__(env, devicestr)

        self.target_net_sync = target_net_sync

        self.hidden_size = hidden_size
        self.hidden_layers = hidden_layers

        self.net = self._config_dueling_net() if dueling_network else self._config_simple_net()

        print(self.net)

        self.tgt_net = ptan.agent.TargetNet(self.net)

        self.selector = ptan.actions.EpsilonGreedyActionSelector(
            epsilon=1,
            selector=ptan.actions.ArgmaxActionSelector())

        self.epsilon_tracker = ptan.actions.EpsilonTracker(selector=self.selector, eps_start=eps_start, eps_final=eps_final, eps_frames=eps_frames)

        self.agent = ptan.agent.DQNAgent(self.net, self.selector, device = self.device)

        self.exp_source = ptan.experience.ExperienceSourceFirstLast(self.env, self.agent, gamma=gamma, steps_count = steps_count)

        if use_combined_replay_buffer:
            self.buffer = CombinedReplayBuffer(self.exp_source, buffer_size=buffer_size)
        else:
            self.buffer = ptan.experience.ExperienceReplayBuffer(self.exp_source, buffer_size=buffer_size)

    def _config_simple_net(self) -> nn.Module:
        return SimpleNet(self.env.observation_space.shape[0], self.env.action_space.n,
                         self.hidden_layers,  self.hidden_size).to(self.device)

    def _config_dueling_net(self) -> nn.Module:
        return DuelingNet(self.env.observation_space.shape[0], self.env.action_space.n,
                          self.hidden_layers,  self.hidden_size).to(self.device)

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
