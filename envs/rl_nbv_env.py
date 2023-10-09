from typing import Optional
import numpy as np
import math
import gym
from gym import spaces
import envs.shapenet_reader as shapenet_reader
import random
import torch
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from distance.chamfer_distance import ChamferDistanceFunction
import logging

os.environ['CUDA_VISIBLE_DEVICES'] = "0"


def resample_pcd(pcd, n, logger, name):
    """Drop or duplicate points so that pcd has exactly n points"""
    if pcd.shape[0] == 0:
        logger.error("obervation space points is 0! model: {}".format(name))
        return np.zeros((n, 3))
    idx = np.random.permutation(pcd.shape[0])
    if idx.shape[0] < n:
        idx = np.concatenate([idx, np.random.randint(pcd.shape[0], size=n-pcd.shape[0])])
    return pcd[idx[:n]]


def normalize_pc(points, logger, name):
    centroid = np.mean(points, axis=0)
    points -= centroid
    furthest_distance = np.max(np.sqrt(np.sum(abs(points)**2,axis=-1)))
    if furthest_distance == 0:
        logger.error("furthest_distance is 0, model: {}".format(name))
        return points
    points /= furthest_distance
    return points


class PointCloudNextBestViewEnv(gym.Env):
    def __init__(self,
                 data_path,
                 view_num=33,
                 begin_view=-1,
                 observation_space_dim=-1,
                 terminated_coverage=0.97,
                 max_step=11,
                 env_id=None,
                 log_level=logging.DEBUG,
                 is_print=False,
                 is_normalize=True,
                 is_ratio_reward=False,
                 is_reward_with_cur_coverage=False,
                 cur_coverage_ratio=1.0):
        self.COVERAGE_THRESHOLD = 0.00005
        self.is_ratio_reward = is_ratio_reward
        self.is_reward_with_cur_coverage = is_reward_with_cur_coverage
        self.cur_coverage_ratio = cur_coverage_ratio
        self.terminated_coverage = terminated_coverage
        self.action_space = spaces.Discrete(view_num)
        self.DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self._init_logger(env_id, log_level, is_print=is_print)
        self.logger.info("PointCloudNextBestViewEnv is ok")
        real_data_path = data_path
        if env_id is not None:
            real_data_path = os.path.join(data_path, str(env_id))
        self.shapenet_reader = shapenet_reader.ShapenetReader(real_data_path, view_num, self.logger, True)
        self.view_state = np.zeros(view_num, dtype=np.int32) 
        self.view_num = view_num
        self.begin_view = begin_view
        self.max_step = max_step
        if self.begin_view == -1:
            self.current_view = random.randint(0, self.view_num - 1)
            self.logger.info("random init view: {}".format(self.current_view))
        else:
            self.current_view = self.begin_view
        self.action_history = [self.current_view]
        self.current_points_cloud = self.shapenet_reader.get_point_cloud_by_view_id(self.current_view)
        self.ground_truth_points_cloud = self.shapenet_reader.ground_truth
        self.ground_truth_points_cloud_size = self.ground_truth_points_cloud.shape[0]
        self.ground_truth_tensor = self.shapenet_reader.ground_truth[np.newaxis, :, :].astype(np.float32)
        self.ground_truth_tensor = torch.tensor(self.ground_truth_tensor).to(self.DEVICE)
        self.view_state[self.current_view] = 1
        self.observation_space_dim = observation_space_dim
        self.is_normalize = is_normalize
        if observation_space_dim == -1:
            # for debug
            self.observation_space = spaces.Dict({
                "current_point_cloud": spaces.Box(low=float("-inf"), 
                                                  high=float("inf"), 
                                                  shape=(512, 3), 
                                                  dtype=np.float64),
                "view_state": spaces.Box(low=0,
                                         high=1,
                                         shape=(view_num,),
                                         dtype=np.int32),
            })
        else:
            if self.is_normalize:
                self.observation_space = spaces.Dict({
                    "current_point_cloud": spaces.Box(low=float("-1"), 
                                                      high=float("1"), 
                                                      shape=(3, observation_space_dim), 
                                                      dtype=np.float64),
                    "view_state": spaces.Box(low=0,
                                             high=1,
                                             shape=(view_num,),
                                             dtype=np.int32),
                })
            else:
                self.observation_space = spaces.Dict({
                    "current_point_cloud": spaces.Box(low=float("-inf"), 
                                                    high=float("inf"), 
                                                    shape=(3, observation_space_dim), 
                                                    dtype=np.float64),
                    "view_state": spaces.Box(low=0,
                                            high=1,
                                            shape=(view_num,),
                                            dtype=np.int32),
                })
        self.current_coverage = self._caculate_current_coverage()
        self.coverage_add = self.current_coverage
        self.step_cnt = 1
        self.model_name = self.shapenet_reader.get_model_info()

    def step(self, action):
        self.action_history.append(action)
        self.step_cnt += 1
        if self.view_state[action] == 1:
            # print("[INFO] action {} is visited".format(action))
            reward = self._get_reward(-0.05, action)
            observation = self._get_observation_space()
            terminated = self._get_terminated()
            info = self._get_info()
            self.logger.debug("[step] action: {:2d}, cover_add: {:.2f}, cur_cover: {:.2f}, step_cnt: {:2d}, terminated: {}".format(action, 0, self.current_coverage * 100, self.step_cnt, terminated))
            return observation, reward, terminated, info
        new_points_cloud = self.shapenet_reader.get_point_cloud_by_view_id(action)
        self.view_state[action] = 1
        new_points_cloud_tensor = new_points_cloud[np.newaxis, :, :].astype(np.float32) 
        new_points_cloud_tensor = torch.tensor(new_points_cloud_tensor).to(self.DEVICE)
        cur_points_cloud_tensor = self.current_points_cloud[np.newaxis, :, :].astype(np.float32) 
        cur_points_cloud_tensor = torch.tensor(cur_points_cloud_tensor).to(self.DEVICE)
        dist1, dist2 =  ChamferDistanceFunction.apply(new_points_cloud_tensor, cur_points_cloud_tensor)  
        dist1 = dist1.cpu().numpy()      
        overlay_flag = dist1 < self.COVERAGE_THRESHOLD
        
        increase_points_cloud = new_points_cloud[~overlay_flag[0, :]]
        if increase_points_cloud.shape[0] == 0:
            # print("[INFO] increase_points_cloud shape is 0")
            reward = self._get_reward(0, action)
            # self._get_debug_info()
            observation = self._get_observation_space()
            terminated = self._get_terminated()
            info = self._get_info()
            self.logger.debug("[step] action: {:2d}, cover_add: {:.2f}, cur_cover: {:.2f}, step_cnt: {:2d}, terminated: {}".format(action, 0, self.current_coverage * 100, self.step_cnt, terminated))
            return observation, reward, terminated, info

        self.current_points_cloud = np.append(self.current_points_cloud, increase_points_cloud, axis=0)
        increase_points_tensor = increase_points_cloud[np.newaxis, :, :].astype(np.float32) 
        increase_points_tensor = torch.tensor(increase_points_tensor).to(self.DEVICE)
        dist1, dist2 = ChamferDistanceFunction.apply(increase_points_tensor, self.ground_truth_tensor)  
        dist2 = dist2.cpu().numpy()      
        cover_flag = dist2 < self.COVERAGE_THRESHOLD
        # cover_flag = cover_flag[0, :]
        cover_add = np.sum(cover_flag == True)
        cover_add = cover_add / self.ground_truth_points_cloud_size
        self.current_coverage += cover_add
        self.coverage_add = cover_add

        # Points that have already been covered will no longer be counted repeatedly
        self.current_points_cloud_from_gt = np.append(self.current_points_cloud_from_gt,
                                                      self.ground_truth_points_cloud[cover_flag[0, :]],
                                                      axis=0)
        self.ground_truth_points_cloud = self.ground_truth_points_cloud[~cover_flag[0, :]]
        self.ground_truth_tensor = self.ground_truth_points_cloud[np.newaxis, :, :].astype(np.float32)
        self.ground_truth_tensor = torch.tensor(self.ground_truth_tensor).to(self.DEVICE)
        
        reward = self._get_reward(cover_add, action)
        observation = self._get_observation_space()
        terminated = self._get_terminated()
        info = self._get_info()

        if cover_add == 1:
            self.logger.error("cover_add is 1")
            self._get_debug_info()
        self.logger.debug("[step] action: {:2d}, cover_add: {:.2f}, cur_cover: {:.2f}, step_cnt: {:2d}, terminated: {}".format(action, cover_add * 100, self.current_coverage * 100, self.step_cnt, terminated))
        return observation, reward, terminated, info
    
    # for greedy policy test
    def try_step(self, action):
        if self.view_state[action] == 1:
            return 0
        new_points_cloud = self.shapenet_reader.get_point_cloud_by_view_id(action)
        new_points_cloud_tensor = new_points_cloud[np.newaxis, :, :].astype(np.float32) 
        new_points_cloud_tensor = torch.tensor(new_points_cloud_tensor).to(self.DEVICE)
        cur_points_cloud_tensor = self.current_points_cloud[np.newaxis, :, :].astype(np.float32) 
        cur_points_cloud_tensor = torch.tensor(cur_points_cloud_tensor).to(self.DEVICE)
        dist1, dist2 =  ChamferDistanceFunction.apply(new_points_cloud_tensor, cur_points_cloud_tensor)  
        dist1 = dist1.cpu().numpy()      
        overlay_flag = dist1 < self.COVERAGE_THRESHOLD
        
        increase_points_cloud = new_points_cloud[~overlay_flag[0, :]]
        if increase_points_cloud.shape[0] == 0:
            return 0

        increase_points_tensor = increase_points_cloud[np.newaxis, :, :].astype(np.float32) 
        increase_points_tensor = torch.tensor(increase_points_tensor).to(self.DEVICE)
        dist1, dist2 = ChamferDistanceFunction.apply(increase_points_tensor, self.ground_truth_tensor)  
        dist2 = dist2.cpu().numpy()      
        cover_flag = dist2 < self.COVERAGE_THRESHOLD
        # cover_flag = cover_flag[0, :]
        cover_add = np.sum(cover_flag == True)
        cover_add = cover_add / self.ground_truth_points_cloud_size
        return cover_add

    def reset(self, init_step=-1):
        self.shapenet_reader.get_next_model()
        self.view_state = np.zeros(self.view_num, dtype=np.int32) 
        self.action_history.clear()
        if self.begin_view == -1:
            self.current_view = random.randint(0, self.view_num - 1)
        else:
            self.current_view = self.begin_view
        if init_step != -1:
            self.current_view = init_step
        self.action_history.append(self.current_view)
        self.current_points_cloud = self.shapenet_reader.get_point_cloud_by_view_id(self.current_view)
        self.view_state[self.current_view] = 1
        self.ground_truth_points_cloud = self.shapenet_reader.ground_truth
        self.ground_truth_points_cloud_size = self.ground_truth_points_cloud.shape[0]
        self.ground_truth_tensor = self.shapenet_reader.ground_truth[np.newaxis, :, :].astype(np.float32)
        self.ground_truth_tensor = torch.tensor(self.ground_truth_tensor).to(self.DEVICE)
        self.current_coverage = self._caculate_current_coverage()
        self.coverage_add = self.current_coverage
        self.step_cnt = 1
        self.model_name = self.shapenet_reader.get_model_info()
        observation = self._get_observation_space()
        info = self._get_info()
        self.logger.debug("[reset] pass, init step: {}".format(self.current_view))
        return observation

    def close(self):
        pass

    def render(sellf):
        pass

    def _caculate_current_coverage(self):
        cur_points_cloud_tensor = self.current_points_cloud[np.newaxis, :, :].astype(np.float32) 
        cur_points_cloud_tensor = torch.tensor(cur_points_cloud_tensor).to(self.DEVICE)
        dist1, dist2 = ChamferDistanceFunction.apply(cur_points_cloud_tensor, self.ground_truth_tensor)  
        dist2 = dist2.cpu().numpy()      
        cover_flag = dist2 < self.COVERAGE_THRESHOLD
        # cover_flag = cover_flag[0, :]
        coverage = np.sum(cover_flag == True)
        coverage = coverage / self.ground_truth_points_cloud_size

        # Points that have already been covered will no longer be counted repeatedly
        self.current_points_cloud_from_gt = self.ground_truth_points_cloud[cover_flag[0, :]]
        self.ground_truth_points_cloud = self.ground_truth_points_cloud[~cover_flag[0, :]]
        self.ground_truth_tensor = self.ground_truth_points_cloud[np.newaxis, :, :].astype(np.float32)
        self.ground_truth_tensor = torch.tensor(self.ground_truth_tensor).to(self.DEVICE)
        return coverage

    def _get_reward(self, cover_add, action):
        if self.is_reward_with_cur_coverage == True:
            if self.step_cnt < 4:
                return cover_add * 10
            else:
                if cover_add <= 0:
                    return cover_add * 10
                else:
                    remain = 1.0 - (self.current_coverage - cover_add)
                    reward = (cover_add / remain) * 5 + cover_add * 5
                    return reward
        elif self.is_ratio_reward:
            if cover_add <= 0:
                return cover_add * 10
            else:
                remain = 1.0 - (self.current_coverage - cover_add)
                reward = (cover_add / remain) * 10
                return reward
        else:
            return cover_add * 10

    def _get_observation_space(self):
        if self.observation_space_dim == -1:
            # do not downsample, just for debug
            cur_pc = self.current_points_cloud_from_gt.T
            return {"current_point_cloud": cur_pc, "view_state": self.view_state}
        else:
            cur_pc = resample_pcd(self.current_points_cloud_from_gt, self.observation_space_dim, self.logger, self.model_name)
            if self.is_normalize:
                cur_pc = normalize_pc(cur_pc, self.logger, self.model_name)
            # for PC_NBV net
            cur_pc = cur_pc.T
            return {"current_point_cloud": cur_pc, "view_state": self.view_state}

    def _get_terminated(self):
        if self.step_cnt > self.max_step:
            return True
        else:
            return False
        
    def _get_info(self):
        return {"cur_points_cloud": self.ground_truth_points_cloud, 
                "model_name": self.model_name, 
                "current_coverage": self.current_coverage}

    def _get_debug_info(self):
        self.logger.info("model name:{}, action history: {}".format(self.model_name, self.action_history))

    def _init_logger(self, env_id, log_level, is_print=False, is_log_file=True):
        log_path = None
        if env_id == None:
            log_path = "env.log"
        else:
            log_path = "env_{}.log".format(env_id)

        self.logger = logging.getLogger(log_path)
        self.logger.setLevel(log_level)
        log_format = logging.Formatter('[%(asctime)s] [%(levelname)s] %(message)s', '%Y-%m-%d %H:%M:%S')

        if is_print:
            shell_handle = logging.StreamHandler()
            shell_handle.setFormatter(log_format)
            shell_handle.setLevel(log_level)
            self.logger.addHandler(shell_handle)
        if is_log_file:
            file_handle = logging.FileHandler(log_path)
            file_handle.setFormatter(log_format)
            file_handle.setLevel(log_level)
            self.logger.addHandler(file_handle)