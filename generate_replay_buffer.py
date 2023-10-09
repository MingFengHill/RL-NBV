import envs.rl_nbv_env
import argparse
import logging
from copy import deepcopy
from stable_baselines3.common.buffers import DictReplayBuffer
from stable_baselines3.common.save_util import save_to_pkl, load_from_pkl
from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv
import stable_baselines3
import numpy as np
import os
import random
# nohup ./run_generate_replay_buffer.sh > test.log 2>&1 &


def make_env(data_path, env_id, args, logger):
    def _f():
        if args.env_num == 1:
            env = envs.rl_nbv_env.PointCloudNextBestViewEnv(data_path=data_path,
                                                            view_num=args.view_num,
                                                            observation_space_dim=args.observation_space_dim,
                                                            log_level=logging.INFO,
                                                            is_ratio_reward=False,
                                                            is_reward_with_cur_coverage=False)
            return env
        if args.is_ratio_reward == 1:
            env = envs.rl_nbv_env.PointCloudNextBestViewEnv(data_path=data_path,
                                                            view_num=args.view_num,
                                                            observation_space_dim=args.observation_space_dim,
                                                            env_id=env_id,
                                                            log_level=logging.INFO,
                                                            is_ratio_reward=True)
            logger.info("is_ratio_reward is True")
        else:
            env = envs.rl_nbv_env.PointCloudNextBestViewEnv(data_path=data_path,
                                                            view_num=args.view_num,
                                                            observation_space_dim=args.observation_space_dim,
                                                            env_id=env_id,
                                                            log_level=logging.INFO,
                                                            is_ratio_reward=False)
            logger.info("is_ratio_reward is False")
        return env
    return _f


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--step_size', type=int, required=True)
    parser.add_argument('--view_num', type=int, required=True)
    parser.add_argument('--buffer_size', type=int, required=True)
    parser.add_argument('--env_num', type=int, required=True)
    parser.add_argument('--observation_space_dim', type=int, required=True)
    parser.add_argument('--save_path', type=str, required=True)
    parser.add_argument('--log_path', type=str, default="replay_buffer.log")
    parser.add_argument('--is_load_buffer', type=int, default=0) 
    parser.add_argument('--load_path', type=str, default=None)
    parser.add_argument('--is_add_negative_exp', type=int, default=0)
    parser.add_argument('--negative_exp_factor', type=float, default=0.03)
    parser.add_argument('--is_ratio_reward', type=int, default=1)
    parser.add_argument('--is_reward_with_cur_coverage', type=int, default=1)
    parser.add_argument('--cur_coverage_ratio', type=float, default=1.0)
    args = parser.parse_args()

    # initialize logger
    logger = logging.getLogger(args.log_path)
    logger.setLevel(logging.DEBUG)
    log_format = logging.Formatter('[%(asctime)s] [%(levelname)s] %(message)s', '%Y-%m-%d %H:%M:%S')
    file_handle = logging.FileHandler(args.log_path)
    file_handle.setFormatter(log_format)
    file_handle.setLevel(logging.DEBUG)
    logger.addHandler(file_handle)

    for arg, value in sorted(vars(args).items()):
        logger.info("Argument {}: {}".format(arg, value))

    env_list = []
    for i in range(args.env_num):
        env_list.append(make_env(args.data_path, i, args, logger))
    env_vec = DummyVecEnv(env_list)
    replay_buffer = None
    if args.is_load_buffer == 1:
        replay_buffer = load_from_pkl(args.load_path, verbose=1)
    else:
        replay_buffer = DictReplayBuffer(buffer_size=args.buffer_size,
                                         observation_space=env_vec.observation_space,
                                         action_space=env_vec.action_space,
                                         device='cuda:1',
                                         n_envs=args.env_num)
    
    experience_size = 0
    negative_experience_size = 0
    model_size = env_vec.envs[0].shapenet_reader.model_num
    logger.info("begin execution, model size: {}".format(model_size * args.env_num))
    for model_id in range(model_size):
        logger.info("handle {} model".format(model_id * args.env_num))
        last_obs = env_vec.reset()
        actions = np.zeros((args.env_num,), dtype=np.int32)
        for step in range(args.step_size - 1):
            experience_size += args.env_num
            for env_id in range(args.env_num):
                view_state = last_obs["view_state"][env_id]
                action = 0
                cover_add_max = 0
                # TODO: test adding nagative experience
                if args.is_add_negative_exp == 1:
                    factor = random.random()
                    if factor <= args.negative_exp_factor:
                        negative_experience_size += 1
                        action = random.randint(0, args.view_num - 1)
                        while view_state[action] != 1:
                            action = random.randint(0, args.view_num - 1)
                        actions[env_id] = action
                        continue
                for i in range(args.view_num):
                    if view_state[i] == 1:
                        continue
                    cover_add_cur = env_vec.envs[env_id].try_step(i)
                    if cover_add_cur >= cover_add_max:
                        cover_add_max = cover_add_cur
                        action = i
                    actions[env_id] = action
            obs, reward, done, info = env_vec.step(actions)

            # Avoid modification by reference
            obs_ = deepcopy(obs)
            info_ = deepcopy(info)
            last_obs_ = deepcopy(last_obs)
            actions_ = deepcopy(actions)

            replay_buffer.add(last_obs_,
                              obs_,
                              actions_,
                              reward,
                              done,
                              info_)
            if np.any(np.isnan(last_obs["current_point_cloud"])):
                logger.error("current_point_cloud has nan")
                for env_id in range(args.env_num):
                    logger.error("model name: {}".format(env_vec.envs[env_id].model_name))
            if np.any(np.isnan(last_obs["view_state"])):
                logger.error("view_state has nan")
                for env_id in range(args.env_num):
                    logger.error("model name: {}".format(env_vec.envs[env_id].model_name))
            last_obs = obs

    logger.info("save as pkl file")
    save_to_pkl(args.save_path, replay_buffer, verbose=1)
    logger.info("model size: {}, experiment size: {}, negative experience size: {}".format(model_size, experience_size, negative_experience_size))
        
