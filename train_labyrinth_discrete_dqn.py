import gym
import labyrinth_ctrl

import os
import numpy as np

from matplotlib import pyplot as plt

import tensorflow as tf
import keras

from stable_baselines import DQN
from stable_baselines import results_plotter
from stable_baselines.bench import Monitor
from stable_baselines.results_plotter import load_results, ts2xy
from stable_baselines.common.noise import AdaptiveParamNoiseSpec
from stable_baselines.common.callbacks import BaseCallback, CallbackList
from stable_baselines.common.policies import ActorCriticPolicy
from stable_baselines.common.vec_env import VecVideoRecorder

# Create log dir
log_dir = "tmp_dqn/"
# video_folder = 'logs/videos_dqn/'

net_arch=[dict(pi=[512, 256],vf=[256, 128])]

policy_kwa = dict(act_fun = tf.nn.tanh, net_arch = net_arch)

os.makedirs(log_dir, exist_ok=True)

env = gym.make('labyrinthCtrlDiscrete-v0')

env = Monitor(env, log_dir)
#'CnnPolicy'
model = DQN('CnnPolicy', env, verbose=1, tensorboard_log=log_dir)


time_steps = 100_000
# time_steps = 10_000
model.learn(total_timesteps=int(time_steps))

model.save("dqn_cnn_labyrinthCtrl_discrete_v0")

env.close()

