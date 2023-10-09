import os
import gym
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3.common import results_plotter
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy, plot_results
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.type_aliases import TrainFreq, TrainFrequencyUnit


class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq:
    :param log_dir: Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: Verbosity level: 0 for no output, 1 for info messages, 2 for debug messages
    """
    def __init__(self, check_freq: int, log_dir: str, verbose: int = 1):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, "best_model")
        self.best_mean_reward = -np.inf

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:

          # Retrieve training reward
          x, y = ts2xy(load_results(self.log_dir), "timesteps")
          if len(x) > 0:
              # Mean training reward over the last 100 episodes
              mean_reward = np.mean(y[-100:])
              if self.verbose >= 1:
                print(f"Num timesteps: {self.num_timesteps}")
                print(f"Best mean reward: {self.best_mean_reward:.2f} - Last mean reward per episode: {mean_reward:.2f}")

              # New best model, you could save the agent here
              if mean_reward > self.best_mean_reward:
                  self.best_mean_reward = mean_reward
                  # Example for saving best model
                  if self.verbose >= 1:
                    print(f"Saving new best model to {self.save_path}")
                  self.model.save(self.save_path)

        return True
    

class NextBestViewCustomCallback(BaseCallback):
    def __init__(self, 
                 output_file, 
                 verify_env, 
                 test_env, 
                 check_freq=10000, 
                 step_size=10, 
                 verbose: int = 1, 
                 check_replay_buffer: bool = False):
        super(NextBestViewCustomCallback, self).__init__(verbose)
        self.output_file = output_file
        self.verify_env = verify_env
        self.test_env = test_env
        self.step_size = step_size
        self.check_freq = check_freq
        self.cnt = 0
        self.best_coverage = 90
        self.check_replay_buffer = check_replay_buffer
        self.need_adjust = True
    
    # check the repaly buffer
    def _init_callback(self) -> None:
        if not self.check_replay_buffer:
            return
        experience = self.model.replay_buffer.sample(32, env=self.model._vec_normalize_env)
        
    def _on_rollout_end(self) -> None:
        if not self.check_replay_buffer:
            return
        experience = self.model.replay_buffer.sample(32, env=self.model._vec_normalize_env)
    
    def _on_step(self) -> bool:
        if self.need_adjust and self.model.num_timesteps > 100000:
            print("[INFO] before change num_timesteps: {}, frequency: {}, unit: {}, check_freq: {}".format(self.model.num_timesteps,
                                                                                                           self.model.train_freq.frequency,
                                                                                                           self.model.train_freq.unit,
                                                                                                           self.check_freq))
            train_freq = (32, TrainFrequencyUnit("step"))
            self.model.train_freq = TrainFreq(*train_freq)
            self.check_freq = 20000
            print("[INFO] after change num_timesteps: {}, frequency: {}, unit: {}, check_freq: {}".format(self.model.num_timesteps,
                                                                                                          self.model.train_freq.frequency,
                                                                                                          self.model.train_freq.unit,
                                                                                                          self.check_freq))
            self.need_adjust = False
        if self.n_calls % self.check_freq == 0:
            with open(self.output_file, "a+", encoding="utf-8") as f:
                f.write("------ {} ------\n".format(self.cnt))
            self.cnt += 1
            # self._caclulate_policy_detail()
            cur_coverage = self._caculate_average_coverage()
            if cur_coverage > self.best_coverage:
                self.best_coverage = cur_coverage
                # TODO: delete this function??
                # self.model.save("best_model")
        return True
    
    def _caclulate_policy_detail(self):
        model_size = self.verify_env.shapenet_reader.model_num
        init_step = 0
        for model_id in range(model_size):
            obs = self.verify_env.reset(init_step=init_step)
            init_step = (init_step + 1) % 33
            with open(self.output_file, "a+", encoding="utf-8") as f:
                f.write("{}: ({}) [0]{:.2f} ".format(self.verify_env.model_name, self.verify_env.current_view, self.verify_env.current_coverage * 100))
            for step_id in range(self.step_size - 1):
                action, _states = self.model.predict(obs, deterministic=True)
                obs, rewards, dones, info = self.verify_env.step(action)
                with open(self.output_file, "a+", encoding="utf-8") as f:
                    f.write("({}) [{}]{:.2f} ".format(action, step_id + 1, info["current_coverage"] * 100))
            with open(self.output_file, "a+", encoding="utf-8") as f:
                f.write("\n")

    def _caculate_average_coverage(self):
        model_size = self.test_env.shapenet_reader.model_num
        init_step = 0
        average_coverage = np.zeros(10)
        for model_id in range(model_size):
            obs = self.test_env.reset(init_step=init_step)
            init_step = (init_step + 1) % 33
            average_coverage[0] += self.test_env.current_coverage
            for step_id in range(self.step_size - 1):
                action, _states = self.model.predict(obs, deterministic=True)
                obs, rewards, dones, info = self.test_env.step(action)
                average_coverage[step_id + 1] += info["current_coverage"]
        average_coverage = average_coverage / model_size
        average_coverage = average_coverage * 100
        with open(self.output_file, "a+", encoding="utf-8") as f:
            f.write("average_coverage: ")
            for i in range(self.step_size):
                f.write("[{}]:{:.2f} ".format(i + 1, average_coverage[i]))
            f.write("\n")
        return average_coverage[9]
