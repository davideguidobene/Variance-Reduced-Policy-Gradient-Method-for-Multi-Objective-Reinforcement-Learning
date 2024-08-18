import torch.nn.functional as F
import torch.optim as optim
import gymnasium as gym
import numpy as np
import mo_gymnasium as mo_gym
from queue_env import get_env

SO_ENVS = ['Acrobot-v1', 'CartPole-v1']
MO_ENVS = ['deep-sea-treasure-v0', 'queue']

def deepsea_proj(x):
    if x[0] < 0:
        x[0] = 0
    if x[1] > 23.7:
        x[1] = 23.7
    if x[1] <= -100:
        x[1] = -100
    if x[1] > 0:
        x[1] = 0    
    return x

def deepsea_der(x, epsilon = 1):
    x = deepsea_proj(x)
    return 0.5 * np.array([x[0] + epsilon, 100 + epsilon + x[1]])**(-0.5)

def deepsea_scalarization(x, epsilon = 1):
    x = deepsea_proj(x)
    return (x[0] + epsilon)**0.5 + (100 + epsilon + x[1])**0.5

def queue_proj(x):
    discount_factor = Config.Queue.gamma
    H = Config.H
    return np.clip(x, a_min=0, a_max=(1 - discount_factor ** H)/(1-discount_factor))

def queue_scalarization(x, epsilon=1):
    x = queue_proj(x)
    H = Config.H
    return np.sum(-H / (x + epsilon))

def queue_der(x, epsilon=1):
    x = queue_proj(x)
    H = Config.H
    return H / ((x + epsilon) ** 2)

class Config:
    alg_name = 'MOTSIVRPG' # choose from 'MOPG', 'MOTSIVRPG'
    environment = 'queue' # choose from SO_ENVS = ['Acrobot-v1', 'CartPole-v1'] or MO_ENVS = ['deep-sea-treasure-v0', 'queue']
    
    num_runs = 8
    parallel =  True
    
    epochs = 2400
    
    checkpoint = False
    checkpoint_interval = 200
    debug = False

    # Server Queues Parameters
    M = 8 # choose from 6, 8, 12, 16, 32, 64
    H = 500 # Truncated Horizon

    def initalize_env():
        # Single-objective
        if Config.environment in SO_ENVS:
            Config.env_name = Config.environment
            Config.env = gym.make(Config.env_name)
            env_class_name = Config.env_name[:-3]
            Config.env_config = getattr(Config, env_class_name)

        # Multi-objective
        elif Config.environment in MO_ENVS:
            if Config.environment == 'deep-sea-treasure-v0':    
                Config.env_name = "deep-sea-treasure-v0"
                Config.env = mo_gym.make(Config.env_name)
                Config.env_config = Config.DeepSeaTreasure
            else:
                Config.env_name = 'queue'

                Ms = [6, 8, 12, 16, 32, 64]
                i = Ms.index(Config.M)
                initial_probs = [0.05, 0.0125, 0.0115, 0.005, 0.00175, 0.00053]
                ds = [0.033, 0.025, 0.01, 0.006, 0.0015, 0.00038]
                ars = [initial_probs[i] + m * ds[i] for m in range(Config.M)]

                Config.env = get_env(max_queue_len=Config.H, H=Config.H, arrival_rates=ars)
                Config.env_config = Config.Queue
    

    class Policy:
        activation = F.relu
        parameterization = F.softmax
        optimizer = optim.Adam
        learning_rate = 3e-4
        hidden_size = 128

    class PGMethods:
        num_episodes_eval = 50
        num_episodes_per_train_step = 288
    
    class MOPG(PGMethods):
        pass
    
    class MOTSIVRPG(PGMethods):
        N = 144
        B = 12
        n_iterations = 12
        delta = 2
        
    class Environment:
        gamma = 0.9999
        scalarization = lambda x: x
        derivative_scalarization = lambda x: np.array([1])
        proj = lambda x: x
        single_objective = True

    class Acrobot(Environment):
        pass

    class CartPole(Environment):
        gamma = 0.9999
        
    class MORLEnvironment(Environment):
        single_objective = False
        scalarization = lambda x: x[0]
        derivative_scalarization = 1

    class DeepSeaTreasure(MORLEnvironment):
        gamma = 1
        proj = deepsea_proj
        scalarization = deepsea_scalarization
        derivative_scalarization = deepsea_der

    class Queue(MORLEnvironment): 
        gamma = 0.9999
        proj = queue_proj
        scalarization = queue_scalarization
        derivative_scalarization = queue_der
