import time
import os
import warnings
warnings.filterwarnings("ignore")

import sys
sys.path.append('HybridPPO')

# from HybridPPO.hybridppo import *
from BusBunchingEnv import Env

import gym
from gym import spaces
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy


if __name__ == '__main__':

    # env = gym.make('Moving-v0')
    # if recording
    # env = gym.wrappers.Monitor(env, "./video", force=True)
    # env.metadata["render.modes"] = ["human", "rgb_array"]
    
    # env.reset()

    # ACTION_SPACE = env.action_space[0].n
    # PARAMETERS_SPACE = env.action_space[1].shape[0]
    # OBSERVATION_SPACE = env.observation_space.shape[0]
    

    mode = 'waiting_time'
    holding_only = False
    config = {"holding_only": holding_only, "mode": mode}
    env = Env(config)

    model_dir = f"models/PPO{mode}"
    logdir = "logs"

    if not os.path.exists(logdir):
        os.makedirs(logdir)

    model = PPO("MlpPolicy", 
                    env, 
                    verbose=1, 
                    batch_size=128, 
                    tensorboard_log=logdir,
                    learning_rate=0.001,
                    gamma=0.99,
                    device='cuda')

    model.learn(total_timesteps=300000)
    model.save(f"model_dir/{mode}")

    del model # remove to demonstrate saving and loading

    # model = PPO.load(f"model_dir/{mode}")

    # obs = env.reset()
    # while True:
    #     action = (0, 0)
    #     obs, rewards, dones, info = env.step(action)
    #     # if rendering
    #     time.sleep(0.1)

    # time.sleep(1)
    # env.close()