import gym
import labyrinth_ctrl

import os
import numpy as np

from matplotlib import pyplot as plt

import tensorflow as tf
import keras

from stable_baselines import PPO2
from stable_baselines import results_plotter
from stable_baselines.bench import Monitor
from stable_baselines.results_plotter import load_results, ts2xy
from stable_baselines.common.noise import AdaptiveParamNoiseSpec
from stable_baselines.common.callbacks import BaseCallback
from stable_baselines.common.policies import ActorCriticPolicy
from stable_baselines.common.vec_env import VecVideoRecorder


class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq: (int)
    :param log_dir: (str) Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: (int)
    """
    def __init__(self, check_freq: int, log_dir: str, verbose=1):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, 'best_model')
        self.best_mean_reward = -np.inf

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:

            # Retrieve training reward
            x, y = ts2xy(load_results(self.log_dir), 'timesteps')
            if len(x) > 0:
              # Mean training reward over the last 100 episodes
                mean_reward = np.mean(y[-100:])
                if self.verbose > 0:
                    print("Num timesteps: {}".format(self.num_timesteps))
                    print("Best mean reward: {:.2f} - Last mean reward per episode: {:.2f}".format(self.best_mean_reward, mean_reward))

                # New best model, you could save the agent here
                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    # Example for saving best model
                    if self.verbose > 0:
                        print("Saving new best model to {}".format(self.save_path))
                    self.model.save(self.save_path)

        return True




class KerasPolicy(ActorCriticPolicy):
    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=False, **kwargs):
        super(KerasPolicy, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=reuse, scale=False)

        with tf.variable_scope("model", reuse=reuse):
#             Conv_layer_1 >> Conv --> RelU --> Pool
            cnn1 = tf.keras.layers.Conv2D(32,(3,3), padding="same")(self.processed_obs)
            act1 = tf.keras.layers.Activation('relu')(cnn1)
            bnorm1 = tf.keras.layers.BatchNormalization(axis=2)(act1)
            mpool1 = tf.keras.layers.MaxPooling2D((3,3))(bnorm1)
            drop1 = tf.keras.layers.Dropout(0.25)(mpool1)            
#             Conv_layer_2 >> Conv --> RelU --> Pool
            cnn2 = tf.keras.layers.Conv2D(64,(3,3), padding="same")(drop1)
            act2 = tf.keras.layers.Activation('relu')(cnn2)
            bnorm2 = tf.keras.layers.BatchNormalization(axis=2)(act2)
            mpool2 = tf.keras.layers.MaxPooling2D((3,3))(bnorm2)
            drop2 = tf.keras.layers.Dropout(0.25)(mpool2)
#             Conv_layer_3 >> Conv --> RelU --> Pool
            cnn3 = tf.keras.layers.Conv2D(64,(3,3), padding="same")(drop2)
            act3 = tf.keras.layers.Activation('relu')(cnn3)
            bnorm3 = tf.keras.layers.BatchNormalization(axis=2)(act3)
            mpool3 = tf.keras.layers.MaxPooling2D((3,3))(bnorm3)
            drop3 = tf.keras.layers.Dropout(0.25)(mpool3)    
            
            flat = tf.keras.layers.Flatten()(drop3)

            x = tf.keras.layers.Dense(64, activation="relu", name='pi_fc_0')(flat)
            pi_latent = tf.keras.layers.Dense(64, activation="relu", name='pi_fc_1')(x)

            x1 = tf.keras.layers.Dense(64, activation="relu", name='vf_fc_0')(flat)
            vf_latent = tf.keras.layers.Dense(64, activation="relu", name='vf_fc_1')(x1)

            value_fn = tf.keras.layers.Dense(1, name='vf')(vf_latent)

            self._proba_distribution, self._policy, self.q_value = \
                self.pdtype.proba_distribution_from_latent(pi_latent, vf_latent, init_scale=0.01)

        self._value_fn = value_fn
        self._setup_init()

    def step(self, obs, state=None, mask=None, deterministic=False):
        if deterministic:
            action, value, neglogp = self.sess.run([self.deterministic_action, self.value_flat, self.neglogp],
                                                   {self.obs_ph: obs})
        else:
            action, value, neglogp = self.sess.run([self.action, self.value_flat, self.neglogp],
                                                   {self.obs_ph: obs})
        return action, value, self.initial_state, neglogp

    def proba_step(self, obs, state=None, mask=None):
        return self.sess.run(self.policy_proba, {self.obs_ph: obs})

    def value(self, obs, state=None, mask=None):
        return self.sess.run(self.value_flat, {self.obs_ph: obs})

    
# Create log dir
log_dir = "tmp/"
video_folder = 'logs/videos/'



os.makedirs(log_dir, exist_ok=True)

env = gym.make('labyrinthCtrl-v0')

env = Monitor(env, log_dir)
#'CnnPolicy'
model = PPO2(KerasPolicy, env, verbose=1, tensorboard_log=log_dir)

# Create the callback: check every 1000 steps
callback = SaveOnBestTrainingRewardCallback(check_freq=1000, log_dir=log_dir)

time_steps = 500_000
# time_steps = 10_000
model.learn(total_timesteps=int(time_steps), callback=callback)

model.save("2cnn_labyrinthCtrl_v0")

env.close()

results_plotter.plot_results([log_dir], time_steps, results_plotter.X_TIMESTEPS, "PPO2 labyrinthCtrl")
plt.show()
