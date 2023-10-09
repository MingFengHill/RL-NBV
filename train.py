import envs.rl_nbv_env
import models.pc_nbv
import models.pointnet2_cls_ssg
import numpy as np
import gym
import argparse
import open3d as o3d
import gym.utils.env_checker as gym_env_checker
import stable_baselines3.common.env_checker as sb3_env_checker
import stable_baselines3
import time
import os
import torch, gc
from torchinfo import summary
import copy
import logging
from torch.profiler import profile, record_function, ProfilerActivity
from custom_callback import NextBestViewCustomCallback
from typing import Callable
import optim.adamw
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
# nohup ./run_train.sh > test.log 2>&1 & 


def caclulate_policy_detail(env, model, step_size, output_file):
    model_size = env.shapenet_reader.model_num
    init_step = 0
    for model_id in range(model_size):
        obs = env.reset(init_step=init_step)
        init_step = (init_step + 1) % 33
        with open(output_file, "a+", encoding="utf-8") as f:
            f.write("{}: ({}) [0]{:.2f} ".format(env.model_name, env.current_view, env.current_coverage * 100))
        for step_id in range(step_size - 1):
            action, _states = model.predict(obs, deterministic=True)
            obs, rewards, dones, info = env.step(action)
            with open(output_file, "a+", encoding="utf-8") as f:
                f.write("({}) [{}]{:.2f} ".format(action, step_id + 1, info["current_coverage"] * 100))
        with open(output_file, "a+", encoding="utf-8") as f:
            f.write("\n")


def caculate_average_coverage(env, model, step_size, output_file):
    model_size = env.shapenet_reader.model_num
    average_coverage = np.zeros(10)
    init_step = 0
    for model_id in range(model_size):
        obs = env.reset(init_step=init_step)
        init_step = (init_step + 1) % 33
        average_coverage[0] += env.current_coverage
        for step_id in range(step_size - 1):
            action, _states = model.predict(obs, deterministic=True)
            obs, rewards, dones, info = env.step(action)
            average_coverage[step_id + 1] += info["current_coverage"]
    average_coverage = average_coverage / model_size
    average_coverage = average_coverage * 100
    with open(output_file, "a+", encoding="utf-8") as f:
        f.write("average_coverage: ")
        for i in range(step_size):
            f.write("[{}]:{:.2f} ".format(i + 1, average_coverage[i]))
        f.write("\n")


def make_env(data_path, env_id, args):
    def _f():
        if args.is_ratio_reward == 1:
            env = envs.rl_nbv_env.PointCloudNextBestViewEnv(data_path=data_path,
                                                            view_num=args.view_num,
                                                            observation_space_dim=args.observation_space_dim,
                                                            env_id=env_id,
                                                            log_level=logging.INFO,
                                                            is_ratio_reward=True)
        else:
            env = envs.rl_nbv_env.PointCloudNextBestViewEnv(data_path=data_path,
                                                            view_num=args.view_num,
                                                            observation_space_dim=args.observation_space_dim,
                                                            env_id=env_id,
                                                            log_level=logging.INFO,
                                                            is_ratio_reward=False)
        return env
    return _f


def linear_schedule(initial_value: float) -> Callable[[float], float]:
    """
    Linear learning rate schedule.

    :param initial_value: Initial learning rate.
    :return: schedule that computes
      current learning rate depending on remaining progress
    """
    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0.

        :param progress_remaining:
        :return: current learning rate
        """
        if progress_remaining > 0.05:
            return progress_remaining * initial_value
        else:
            return 0.05 * initial_value

    return func


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--view_num', type=int, required=True)
    parser.add_argument('--verify_data_path', type=str, required=True)
    parser.add_argument('--test_data_path', type=str, required=True)
    parser.add_argument('--observation_space_dim', type=int, required=True)
    parser.add_argument('--step_size', type=int, required=True)
    parser.add_argument('--output_file', type=str, required=True)
    parser.add_argument('--is_save_model', type=int, default=0)
    parser.add_argument('--is_save_replay_buffer', type=int, default=0)
    parser.add_argument('--is_profile', type=int, default=0)
    parser.add_argument('--is_vec_env', type=int, default=0)
    parser.add_argument('--is_transform', type=int, default=0)
    parser.add_argument('--is_freeze_fe', type=int, default=0)
    parser.add_argument('--env_num', type=int, default=8)
    parser.add_argument('--pretrained_model_path', type=str, default="null")
    parser.add_argument('--is_load_replay_buffer', type=int, default=0)
    parser.add_argument('--replay_buffer_path', type=str, default="null")
    parser.add_argument('--is_ratio_reward', type=int, default=1)
    args = parser.parse_args()

    logger = logging.getLogger("train_detail.log")
    logger.setLevel(logging.DEBUG)
    log_format = logging.Formatter('[%(asctime)s] [%(levelname)s] %(message)s', '%Y-%m-%d %H:%M:%S')
    file_handle = logging.FileHandler("train_detail.log")
    file_handle.setFormatter(log_format)
    file_handle.setLevel(logging.DEBUG)
    logger.addHandler(file_handle)

    for arg, value in sorted(vars(args).items()):
        logger.info("Argument {}: {}".format(arg, value))

    if args.is_vec_env:
        env_list = []
        for i in range(args.env_num):
            env_list.append(make_env(args.data_path, i, args))
        train_env = stable_baselines3.common.vec_env.SubprocVecEnv(env_list)
    else:
        train_env = envs.rl_nbv_env.PointCloudNextBestViewEnv(data_path=args.data_path,
                                                              view_num=args.view_num,
                                                              observation_space_dim=args.observation_space_dim,
                                                              log_level=logging.INFO,
                                                              is_ratio_reward=False,
                                                              is_reward_with_cur_coverage=False)
    policy_kwargs = dict(features_extractor_class=models.pointnet2_cls_ssg.PointNetFeatureExtraction,
                         features_extractor_kwargs=dict(features_dim=128),
                         optimizer_class=optim.adamw.AdamW)
    model = stable_baselines3.DQN(policy="MultiInputPolicy",
                                  env=train_env,
                                  policy_kwargs=policy_kwargs,
                                  verbose=1,
                                  device='cuda:1',
                                #   buffer_size=10000,
                                  learning_starts=3000,
                                  batch_size=128,
                                  exploration_fraction=0.5,
                                  exploration_final_eps=0.2,
                                  gradient_steps=1,
                                  learning_rate=linear_schedule(0.001),
                                  train_freq=16,
                                  gamma=0.1)

    if args.is_transform == 1:
        qnet_dict = model.policy.q_net.state_dict()
        logger.info("qnet_dict type: {}".format(type(qnet_dict)))
        update_dict = copy.deepcopy(qnet_dict)
        # logger.info("!original qnet parameters")
        # logger.info('-'*40)
        # for key in sorted(qnet_dict.keys()):
        #     parameter = qnet_dict[key]
        #     logger.info(key)
        #     logger.info(parameter.size())
        #     logger.info(parameter)
        # logger.info('-'*40)

        # load parameters form the pretrained model
        if not os.path.exists(args.pretrained_model_path):
            logger.error("pretrained_model_path: {} is not exists".format(args.pretrained_model_path))
        checkpoint = torch.load(args.pretrained_model_path)
        pretrained_model_state_dict = checkpoint["model_state_dict"]
        logger.info("!pretrained model parameters")
        logger.info('-'*40)
        for key in sorted(pretrained_model_state_dict.keys()):
            parameter = pretrained_model_state_dict[key]
            logger.info(key)
            logger.info(parameter.size())
            logger.info(parameter)
        logger.info('-'*40)

        # Read the parameters required by the feature extractor from the pretrained model
        for key in sorted(pretrained_model_state_dict.keys()):
            if key != "fc3.bias" and key != "fc3.weight":
                key_in_qnet = "features_extractor." + key
                if key_in_qnet in update_dict:
                    update_dict[key_in_qnet] = pretrained_model_state_dict[key]
                    logger.info("key: {} update".format(key_in_qnet))
                else:
                    logger.error("key: {} not in update_dict".format(key_in_qnet))
        model.policy.q_net.load_state_dict(update_dict)
        model.policy.q_net_target.load_state_dict(update_dict)
        model.q_net.load_state_dict(update_dict)
        model.q_net_target.load_state_dict(update_dict)

        qnet_dict = model.policy.q_net.state_dict()
        logger.info("!updated qnet parameters")
        logger.info('-'*40)
        for key in sorted(qnet_dict.keys()):
            parameter = qnet_dict[key]
            logger.info(key)
            logger.info(parameter.size())
            logger.info(parameter)
        logger.info('-'*40)

        if args.is_freeze_fe == 1:
            logger.info("freeze feature extractor")
            for param in model.policy.q_net.features_extractor.sa1.parameters():
                param.requires_grad = False
            for param in model.policy.q_net.features_extractor.sa2.parameters():
                param.requires_grad = False
            for param in model.policy.q_net.features_extractor.sa3.parameters():
                param.requires_grad = False

            for param in model.policy.q_net_target.features_extractor.sa1.parameters():
                param.requires_grad = False
            for param in model.policy.q_net_target.features_extractor.sa2.parameters():
                param.requires_grad = False
            for param in model.policy.q_net_target.features_extractor.sa3.parameters():
                param.requires_grad = False

            for param in model.q_net.features_extractor.sa1.parameters():
                param.requires_grad = False
            for param in model.q_net.features_extractor.sa2.parameters():
                param.requires_grad = False
            for param in model.q_net.features_extractor.sa3.parameters():
                param.requires_grad = False

            for param in model.q_net_target.features_extractor.sa1.parameters():
                param.requires_grad = False
            for param in model.q_net_target.features_extractor.sa2.parameters():
                param.requires_grad = False
            for param in model.q_net_target.features_extractor.sa3.parameters():
                param.requires_grad = False

    if args.is_load_replay_buffer == 1:
        model.load_replay_buffer(args.replay_buffer_path)
        logger.info("load replay buffer: {}".format(args.replay_buffer_path))

    verify_env = envs.rl_nbv_env.PointCloudNextBestViewEnv(data_path=args.verify_data_path,
                                                           view_num=args.view_num,
                                                           observation_space_dim=args.observation_space_dim,
                                                           log_level=logging.INFO)
    test_env = envs.rl_nbv_env.PointCloudNextBestViewEnv(data_path=args.test_data_path,
                                                         view_num=args.view_num,
                                                         observation_space_dim=args.observation_space_dim,
                                                         log_level=logging.INFO)
    custom_callback = NextBestViewCustomCallback(args.output_file, verify_env, test_env, check_freq=15000)

    start_time = time.time()
    if args.is_profile == 0:
        model.learn(500000, callback=custom_callback)
    else:
        with profile(activities=[ProfilerActivity.CUDA], 
                        on_trace_ready=torch.profiler.tensorboard_trace_handler('./log'),
                        profile_memory=True, 
                        record_shapes=True, 
                        with_stack=True, 
                        with_modules=True) as prof:
            with record_function("train"):
                model.learn(2000)
        logger.info(prof.key_averages().table(sort_by="self_cuda_memory_usage", row_limit=400))
        # print(torch.cuda.memory_summary(device=1))
        # gc.collect()
        # torch.cuda.empty_cache()
        # print(torch.cuda.memory_summary(device=1))
    elapsed_time = time.time() - start_time
    print("learn time: ", time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))
    
    if args.is_save_replay_buffer == 1:
        model.save_replay_buffer("dqn_replay_buffer")
    
    # Save the agent
    if args.is_save_model == 1:
        model.save("rl_nbv")
        with open(args.output_file, "a+", encoding="utf-8") as f:
            f.write("------ Before Save ------\n")
        # caclulate_policy_detail(verify_env, model, args.step_size, args.output_file)
        caculate_average_coverage(test_env, model, args.step_size, args.output_file)

        del model  # delete trained model to demonstrate loading
        # Load the trained agent
        # NOTE: if you have loading issue, you can pass `print_system_info=True`
        # to compare the system on which the model was trained vs the current one
        # model = DQN.load("dqn_lunar", env=env, print_system_info=True)
        model = stable_baselines3.DQN.load(path="rl_nbv", 
                                           env=train_env, 
                                           policy_kwargs=policy_kwargs, 
                                           policy="MultiInputPolicy")
        with open(args.output_file, "a+", encoding="utf-8") as f:
            f.write("------ After Save ------\n")
        # caclulate_policy_detail(verify_env, model, args.step_size, args.output_file)
        caculate_average_coverage(test_env, model, args.step_size, args.output_file)

