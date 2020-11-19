from types import SimpleNamespace
import warnings

from _01_environment.carenv_v20 import CarEnvV20
from _02_agent.simple_agent_v10 import SimpleAgentV10
from _03_bridge.simple_bridge_v10 import SimpleBridgeV10
from _04_loopcontrol.loop_control_v10 import LoopControlV10

HYPERPARAMS = {
    'dueling_net_log_hl_2_hs_64': SimpleNamespace(**{
        # env
        'env_mode_energy_penalty'     : False,    # should there be a -1 point penalty for a used energy unit
        'env_mode_random'             : False,    # does acceleration and decelartion have a random part
        'env_mode_limit_steps'        : True,     # * are the maximum possible steps limited
        'env_mode_time_penalty'       : False,    # Penalty for every timestep
        'env_mode_reward'             : "log",    # * Reward mode for overshoot or not reaching goal "lin", "exp" or "log"

        # agent
        'agent_type'                  : "s",      # agent type: s=simple, r=rainbow
        'agent_device'                : "cpu",    # * cpu or cuda
        'agent_gamma_exp'             : 0.9,      # discount_factor for experience_first_last.. shouldn't matter since step_size is only 1
        'agent_buffer_size'           : 50000,    # size of replay buffer
        'agent_target_net_sync'       : 1000,     # sync TargetNet with weights of DNN every .. iterations
        'agent_simple_eps_start'      : 1.0,      # simpleagent: epsilon start
        'agent_simple_eps_final'      : 0.02,     # simpleagent: epsilon end
        'agent_simple_eps_frames'     : 5*10**5,    # * simpleagent: epsilon frames -> how many frames until 0.02 should be reached .. decay is linear
        'agent_hidden_size'           : 64,        # * Neurons in the hidden layer
        'agent_hidden_layers'         : 2,        # * number of hidden layers
        'agent_dueling_network'       : True,     # enable dueling agent
        'agent_steps_count'           : 1,        # Steps Count

        # bridge
        'bridge_optimizer'            : None,     # Optimizer -> default ist Adam
        'bridge_learning_rate'        : 0.0001,   # learningrate
        'bridge_gamma'                : 0.9,      # discount_factor for reward
        'bridge_initial_population'   : 5000,     # initial number of experiences in buffer
        'bridge_batch_size'           : 32,       # batch_size for training

        # loop control
        'loop_bound_avg_reward'       : 95.0,   # target avg reward
        'loop_logtb'                  : True,     # Log to Tensorboard Logfile
    }),
    'dueling_net_lin_hl_2_hs_64': SimpleNamespace(**{
        # env
        'env_mode_energy_penalty'     : False,    # should there be a -1 point penalty for a used energy unit
        'env_mode_random'             : False,    # does acceleration and decelartion have a random part
        'env_mode_limit_steps'        : True,     # * are the maximum possible steps limited
        'env_mode_time_penalty'       : False,    # Penalty for every timestep
        'env_mode_reward'             : "lin",    # * Reward mode for overshoot or not reaching goal "lin", "exp" or "log"

        # agent
        'agent_type'                  : "s",      # agent type: s=simple, r=rainbow
        'agent_device'                : "cpu",    # * cpu or cuda
        'agent_gamma_exp'             : 0.9,      # discount_factor for experience_first_last.. shouldn't matter since step_size is only 1
        'agent_buffer_size'           : 50000,    # size of replay buffer
        'agent_target_net_sync'       : 1000,     # sync TargetNet with weights of DNN every .. iterations
        'agent_simple_eps_start'      : 1.0,      # simpleagent: epsilon start
        'agent_simple_eps_final'      : 0.02,     # simpleagent: epsilon end
        'agent_simple_eps_frames'     : 5*10**5,    # * simpleagent: epsilon frames -> how many frames until 0.02 should be reached .. decay is linear
        'agent_hidden_size'           : 64,        # * Neurons in the hidden layer
        'agent_hidden_layers'         : 2,        # * number of hidden layers
        'agent_dueling_network'       : True,     # enable dueling agent
        'agent_steps_count'           : 1,        # Steps Count

        # bridge
        'bridge_optimizer'            : None,     # Optimizer -> default ist Adam
        'bridge_learning_rate'        : 0.0001,   # learningrate
        'bridge_gamma'                : 0.9,      # discount_factor for reward
        'bridge_initial_population'   : 5000,     # initial number of experiences in buffer
        'bridge_batch_size'           : 32,       # batch_size for training

        # loop control
        'loop_bound_avg_reward'       : 95.0,   # target avg reward
        'loop_logtb'                  : True,     # Log to Tensorboard Logfile
    }),
    'dueling_net_lin_hl_1_hs_64_cuda': SimpleNamespace(**{
        # env
        'env_mode_energy_penalty'     : False,    # should there be a -1 point penalty for a used energy unit
        'env_mode_random'             : False,    # does acceleration and decelartion have a random part
        'env_mode_limit_steps'        : True,     # * are the maximum possible steps limited
        'env_mode_time_penalty'       : False,    # Penalty for every timestep
        'env_mode_reward'             : "lin",    # * Reward mode for overshoot or not reaching goal "lin", "exp" or "log"

        # agent
        'agent_type'                  : "s",      # agent type: s=simple, r=rainbow
        'agent_device'                : "cuda",   # * cpu or cuda
        'agent_gamma_exp'             : 0.9,      # discount_factor for experience_first_last.. shouldn't matter since step_size is only 1
        'agent_buffer_size'           : 50000,    # size of replay buffer
        'agent_target_net_sync'       : 1000,     # sync TargetNet with weights of DNN every .. iterations
        'agent_simple_eps_start'      : 1.0,      # simpleagent: epsilon start
        'agent_simple_eps_final'      : 0.02,     # simpleagent: epsilon end
        'agent_simple_eps_frames'     : 5*10**5,    # * simpleagent: epsilon frames -> how many frames until 0.02 should be reached .. decay is linear
        'agent_hidden_size'           : 64,        # * Neurons in the hidden layer
        'agent_hidden_layers'         : 1,        # * number of hidden layers
        'agent_dueling_network'       : True,     # enable dueling agent
        'agent_steps_count'           : 1,        # Steps Count

        # bridge
        'bridge_optimizer'            : None,     # Optimizer -> default ist Adam
        'bridge_learning_rate'        : 0.0001,   # learningrate
        'bridge_gamma'                : 0.9,      # discount_factor for reward
        'bridge_initial_population'   : 5000,     # initial number of experiences in buffer
        'bridge_batch_size'           : 32,       # batch_size for training

        # loop control
        'loop_bound_avg_reward'       : 95.0,   # target avg reward
        'loop_logtb'                  : True,     # Log to Tensorboard Logfile
    }),
    'dueling_net_lin_hl_2_hs_32': SimpleNamespace(**{
        # env
        'env_mode_energy_penalty'     : False,    # should there be a -1 point penalty for a used energy unit
        'env_mode_random'             : False,    # does acceleration and decelartion have a random part
        'env_mode_limit_steps'        : True,     # * are the maximum possible steps limited
        'env_mode_time_penalty'       : False,    # Penalty for every timestep
        'env_mode_reward'             : "lin",    # * Reward mode for overshoot or not reaching goal "lin", "exp" or "log"

        # agent
        'agent_type'                  : "s",      # agent type: s=simple, r=rainbow
        'agent_device'                : "cuda",   # * cpu or cuda
        'agent_gamma_exp'             : 0.9,      # discount_factor for experience_first_last.. shouldn't matter since step_size is only 1
        'agent_buffer_size'           : 50000,    # size of replay buffer
        'agent_target_net_sync'       : 1000,     # sync TargetNet with weights of DNN every .. iterations
        'agent_simple_eps_start'      : 1.0,      # simpleagent: epsilon start
        'agent_simple_eps_final'      : 0.02,     # simpleagent: epsilon end
        'agent_simple_eps_frames'     : 5*10**5,    # * simpleagent: epsilon frames -> how many frames until 0.02 should be reached .. decay is linear
        'agent_hidden_size'           : 32,        # * Neurons in the hidden layer
        'agent_hidden_layers'         : 2,        # * number of hidden layers
        'agent_dueling_network'       : True,     # enable dueling agent
        'agent_steps_count'           : 1,        # Steps Count

        # bridge
        'bridge_optimizer'            : None,     # Optimizer -> default ist Adam
        'bridge_learning_rate'        : 0.0001,   # learningrate
        'bridge_gamma'                : 0.9,      # discount_factor for reward
        'bridge_initial_population'   : 5000,     # initial number of experiences in buffer
        'bridge_batch_size'           : 32,       # batch_size for training

        # loop control
        'loop_bound_avg_reward'       : 95.0,   # target avg reward
        'loop_logtb'                  : True,     # Log to Tensorboard Logfile
    }),
    'nstep_3_log': SimpleNamespace(**{
        # env
        'env_mode_energy_penalty'     : False,    # should there be a -1 point penalty for a used energy unit
        'env_mode_random'             : False,    # does acceleration and decelartion have a random part
        'env_mode_limit_steps'        : True,     # * are the maximum possible steps limited
        'env_mode_time_penalty'       : False,    # Penalty for every timestep
        'env_mode_reward'             : "log",    # * Reward mode for overshoot or not reaching goal "lin", "exp" or "log"

        # agent
        'agent_type'                  : "s",      # agent type: s=simple, r=rainbow
        'agent_device'                : "cpu",    # * cpu or cuda
        'agent_gamma_exp'             : 0.9,      # discount_factor for experience_first_last.. shouldn't matter since step_size is only 1
        'agent_buffer_size'           : 50000,    # size of replay buffer
        'agent_target_net_sync'       : 1000,     # sync TargetNet with weights of DNN every .. iterations
        'agent_simple_eps_start'      : 1.0,      # simpleagent: epsilon start
        'agent_simple_eps_final'      : 0.02,     # simpleagent: epsilon end
        'agent_simple_eps_frames'     : 5*10**5,    # * simpleagent: epsilon frames -> how many frames until 0.02 should be reached .. decay is linear
        'agent_hidden_size'           : 128,        # * Neurons in the hidden layer
        'agent_hidden_layers'         : 1,         # * number of hidden layers
        'agent_dueling_network'       : False,     # enable dueling agent
        'agent_steps_count'           : 3,         # Steps Count

        # bridge
        'bridge_optimizer'            : None,     # Optimizer -> default ist Adam
        'bridge_learning_rate'        : 0.0001,   # learningrate
        'bridge_gamma'                : 0.9,      # discount_factor for reward
        'bridge_initial_population'   : 5000,     # initial number of experiences in buffer
        'bridge_batch_size'           : 32,       # batch_size for training

        # loop control
        'loop_bound_avg_reward'       : 95.0,   # target avg reward
        'loop_logtb'                  : True,     # Log to Tensorboard Logfile
    }),


}

def create_control(params: SimpleNamespace, config_name) -> LoopControlV10:

    env = CarEnvV20(mode_energy_penalty= params.env_mode_energy_penalty,
                 mode_random           = params.env_mode_random,
                 mode_limit_steps      = params.env_mode_limit_steps,
                 mode_time_penalty     = params.env_mode_time_penalty,
                 mode_reward           = params.env_mode_reward)

    agent = SimpleAgentV10(env,
                        devicestr  = params.agent_device,
                        gamma           = params.agent_gamma_exp,
                        buffer_size     = params.agent_buffer_size,
                        target_net_sync = params.agent_target_net_sync,
                        eps_start       = params.agent_simple_eps_start,
                        eps_final       = params.agent_simple_eps_final,
                        eps_frames      = params.agent_simple_eps_frames,
                        hidden_size     = params.agent_hidden_size,
                        hidden_layers   = params.agent_hidden_layers,
                        dueling_network = params.agent_dueling_network
                        )

    bridge = SimpleBridgeV10(agent=agent,
                          optimizer          = params.bridge_optimizer,
                          learning_rate      = params.bridge_learning_rate,
                          gamma              = params.bridge_gamma,
                          initial_population = params.bridge_initial_population,
                          batch_size         = params.bridge_batch_size,
                          )


    control = LoopControlV10(
        bridge              = bridge,
        run_name            = config_name,
        bound_avg_reward    = params.loop_bound_avg_reward,
        logtb               = params.loop_logtb,
        logfolder="./../runs/runv40")

    return control

def run_example(config_name: str):
    # get rid of missing metrics warning
    warnings.simplefilter("ignore", category=UserWarning)

    control = create_control(HYPERPARAMS[config_name], config_name)
    control.run()

if __name__ == '__main__':
    #run_example('dueling_net_log_hl_2_hs_64')
    #run_example('dueling_net_lin_hl_2_hs_64')
    #run_example('dueling_net_lin_hl_1_hs_64_cuda')
    #run_example('dueling_net_lin_hl_2_hs_32')
    run_example('nstep_3_log')
