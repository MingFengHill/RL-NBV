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
from torchinfo import summary
import logging
from typing import Callable
import random
import matplotlib as mpl
import matplotlib.pyplot as plt
from std_msgs.msg import Int32
import std_msgs
import rospy
from roslib import message
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2, PointField
from scipy.spatial.transform import Rotation as R
from threading import Thread, Lock
import optim.adamw
# nohup ./run_benchmark.sh > test.log 2>&1 &

g_dqn_model = None


def RandomExplore(env, logger, args):
    def func(obs):
        action = random.randint(0, args.view_num -1)
        return action
    return func


def IdealExplore(env, logger, args):
    def func(obs):
        view_state = obs["view_state"]
        action = 0
        cover_add_max = 0
        for i in range(args.view_num):
            if view_state[i] == 1:
                continue
            cover_add_cur = env.try_step(i)
            if cover_add_cur >= cover_add_max:
                cover_add_max = cover_add_cur
                action = i
        return action
    return func


def DQNExplore(env, logger, args):
    def func(obs):
        global g_dqn_model
        action, _states = g_dqn_model.predict(obs, deterministic=True)
        return action
    return func


class PCNBVAdapter:
    def __init__(self, logger, args):
        self.logger = logger
        self.args = args
        self.actions = []
        with open(args.pcnbv_action_path) as f:
            for line in f.readlines():
                self.actions.append(int(line.strip('\n')))
        self.cur_step = 0

    def PCNBVExplore(self, obs):
        if self.cur_step >= len(self.actions):
            self.logger.error("pc-nbv actions out of range")
            return 0
        else:
            if self.cur_step == len(self.actions) - 1:
                self.logger.info("last, step. cur step size: {}".format(self.cur_step))
            nbv = self.actions[self.cur_step]
            self.cur_step = self.cur_step + 1
            return nbv


class InformationGainAdapter:
    def __init__(self, env, logger, args):
        rospy.init_node("benchmark")
        self.env = env
        self.logger = logger
        self.args = args
        self.nbv_lock = Lock()
        self.updata_down_lock = Lock()
        self.update_down = False
        self.camera_id_pub = rospy.Publisher('camera_id_inner', Int32, queue_size=10)
        self.point_clouds_pub = rospy.Publisher('/world/pcl_input', PointCloud2, queue_size=10)
        self.nbv_sub = rospy.Subscriber("next_best_view", Int32, self.next_best_view_callback)
        self.octomap_update_sub = rospy.Subscriber("/world/octomap_update_down", Int32, self.octomap_update_callback)
        self.cur_pos = 0
        self.next_best_view = None
        self.camera_poses = np.loadtxt("./poses.txt")
        self.is_first_step = True
        # clear octomap
        self.views_space_clear_pub = rospy.Publisher('/views/clear', Int32, queue_size=10)
        self.octomap_clear_pub = rospy.Publisher('/world/clear', Int32, queue_size=10)
        self.cur_step = 0
        self.octomap_clear_sub = rospy.Subscriber("/world/octomap_clear_down", Int32, self.octomap_clear_callback)
        self.clear_down = False
        self.updata_clear_lock = Lock()
        self.nbv_update = False
        self.camera_id_update_sub = rospy.Subscriber("camera_id_update", Int32, self.camera_id_update_callback)
        self.camera_id_update_lock = Lock()
        self.camera_id_update = False
        
    def camera_id_update_callback(self, data):
        self.camera_id_update_lock.acquire()
        self.camera_id_update = True
        self.camera_id_update_lock.release()
        
    def octomap_update_callback(self, data):
        self.updata_down_lock.acquire()
        self.update_down = True
        self.updata_down_lock.release()
        
    def octomap_clear_callback(self, data):
        self.updata_clear_lock.acquire()
        self.clear_down = True
        self.updata_clear_lock.release()
    
    def next_best_view_callback(self, data):
        self.nbv_lock.acquire()
        if data.data >= 0 and data.data < self.args.view_num:
            self.next_best_view = data.data
            self.nbv_update = True
        self.nbv_lock.release()
        
    def handle_first_step(self, obs):
        view_state = obs["view_state"]
        for i in range(self.args.view_num):
            if view_state[i] == 1:
                self.cur_pos = i
                self.logger.info("first step: {}".format(i))
                break
        self._update_camera_pose()
        self.views_space_clear_pub.publish(1)
        
    def _clear_octomap(self):
        self.updata_clear_lock.acquire()
        self.update_clear = False
        self.updata_clear_lock.release()
        self.octomap_clear_pub.publish(1)
        # Waiting for the map clear
        while True:
            self.updata_clear_lock.acquire()
            token = self.clear_down
            self.updata_clear_lock.release()
            if token:
                break
            time.sleep(1)
        self.logger.info("octomap clear down")
            
    def _update_camera_pose(self):
        self.updata_down_lock.acquire()
        self.update_down = False
        self.updata_down_lock.release()
        self.camera_id_update_lock.acquire()
        self.camera_id_update = False
        self.camera_id_update_lock.release()
        self.camera_id_pub.publish(self.cur_pos)
        while True:
            self.camera_id_update_lock.acquire()
            token = self.camera_id_update
            self.camera_id_update_lock.release()
            if token:
                break
            time.sleep(1)
        pc_array = np.asarray(self.env.shapenet_reader.get_point_cloud_by_view_id(self.cur_pos))
        world_2_camera = self._get_world_2_camera(self.cur_pos)
        pc_array = np.concatenate([pc_array, np.ones((pc_array.shape[0], 1))], axis=1)
        pc_array = pc_array.T
        pc_array = np.matmul(world_2_camera, pc_array)
        pc_array = pc_array.T
        pc_msg = PointCloud2()
        # make point cloud
        pc_msg = pc2.create_cloud_xyz32(pc_msg.header, pc_array[:, :3])
        pc_msg.header.frame_id = "camera"
        self.point_clouds_pub.publish(pc_msg)
        print("[INFO] go to {}".format(self.cur_pos))
        # Waiting for the map update
        while True:
            self.updata_down_lock.acquire()
            token = self.update_down
            self.updata_down_lock.release()
            if token:
                break
            time.sleep(1)
        self.logger.info("arrival {}".format(self.cur_pos))

    def IGExplore(self, obs):
        if self.is_first_step == True:
            self.handle_first_step(obs)
            self.is_first_step = False
        view_state = obs["view_state"]
        next_best_view = None
        self.nbv_lock.acquire()
        self.nbv_update = False
        self.nbv_lock.release()
        while True:
            self.nbv_lock.acquire()
            nbv_update_local = self.nbv_update
            self.nbv_lock.release()
            if nbv_update_local:
                self.nbv_lock.acquire()
                nbv = self.next_best_view
                self.nbv_lock.release()
                if view_state[nbv] == 1:
                    self.logger.error("Go to same view!")
                break
            time.sleep(1)
        self.logger.info("next best view: {}".format(nbv))    
        self.cur_pos = nbv
        # clear octomap
        self.cur_step += 1
        if self.cur_step == self.args.step_size - 1:
            self._update_camera_pose()
            self._clear_octomap()
            self.cur_step = 0
            self.is_first_step = True
            return nbv
        self._update_camera_pose()
        return nbv
    
    def _get_world_2_camera(self, camera_id):
        pose_from_file = self.camera_poses[camera_id]
        position = [pose_from_file[0], pose_from_file[1], pose_from_file[2]]
        quaternion = [pose_from_file[3], pose_from_file[4], pose_from_file[5], pose_from_file[6]]
        r = R.from_quat(quaternion)
        rotation_matrix = r.as_matrix()
        transformation = [[rotation_matrix[0][0], rotation_matrix[0][1], rotation_matrix[0][2], position[0]],
                          [rotation_matrix[1][0], rotation_matrix[1][1], rotation_matrix[1][2], position[1]],
                          [rotation_matrix[2][0], rotation_matrix[2][1], rotation_matrix[2][2], position[2]],
                          [0, 0, 0, 1]]
        world_2_camera = np.linalg.inv(np.asarray(transformation))
        return world_2_camera
    
    def array_to_pointcloud2(cloud_arr, stamp=None, frame_id=None):
        '''Converts a numpy record array to a sensor_msgs.msg.PointCloud2.
        '''
        cloud_arr = np.atleast_2d(cloud_arr)
        cloud_msg = PointCloud2()

        cloud_msg.header = std_msgs.msg.Header()
        if stamp is not None:
            cloud_msg.header.stamp = stamp
        if frame_id is not None:
            cloud_msg.header.frame_id = frame_id
        cloud_msg.height = cloud_arr.shape[0]
        cloud_msg.width = cloud_arr.shape[1]
        # cloud_msg.fields = dtype_to_fields(cloud_arr.dtype)
        cloud_msg.is_bigendian = False # assumption
        cloud_msg.point_step = cloud_arr.dtype.itemsize
        cloud_msg.row_step = cloud_msg.point_step*cloud_arr.shape[1]
        cloud_msg.is_dense = True
        cloud_msg.data = cloud_arr.tostring()
        return cloud_msg 


#  Returns tuple of handles, labels for axis ax, after reordering them to conform to the label order `order`, and if unique is True, after removing entries with duplicate labels.
def reorderLegend(ax=None, order=None):
    if ax is None: ax=plt.gca()
    handles, labels = ax.get_legend_handles_labels()
    labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0])) # sort both labels and handles by labels
    if order is not None: # Sort according to a given list (not necessarily complete)
        keys=dict(zip(order,range(len(order))))
        labels, handles = zip(*sorted(zip(labels, handles), key=lambda t,keys=keys: keys.get(t[0],np.inf)))
    ax.legend(handles, labels, loc=4, fontsize='xx-large')
    return(handles, labels)


def plot_coverage_results(results, title="Similar testing dataset"):
    color_map = {"Ours":"#C4323F", "Random":"#02263E", "AreaFactor":"#FA8600", "ProximityCount":"#73BAD6", "PC-NBV":"#91D542"}
    markerfacecolor_map = {"Ours":"none", "Random":"#02263E", "AreaFactor":"none", "ProximityCount":"#73BAD6", "PC-NBV":"none"}
    marker_map = {"Ours":"^", "Random":"o", "AreaFactor":"s", "ProximityCount":"v", "PC-NBV":"o"}
    fig, ax = plt.subplots(layout='constrained')
    for key, value in results.items():
        value = value / 100
        x = np.arange(start=1, stop=(len(value)+1))
        if key in color_map:
            color = color_map[key]
            ax.plot(x, 
                    value, 
                    linestyle='-', 
                    color=color, 
                    marker=marker_map[key], 
                    markerfacecolor=markerfacecolor_map[key], 
                    label=key)
        else:
            ax.plot(x, value, marker='o', linestyle='-', label=key)
        ax.legend(loc=4, fontsize='xx-large')
    order = ["Random", "PC-NBV",  "AreaFactor", "ProximityCount", "Ours"]
    reorderLegend(ax, order)
    ax.set_xlabel("Number of rounds", fontsize='xx-large')
    ax.set_ylabel("Average surface coverage", fontsize='xx-large')
    # ax.set_title(title, fontsize='xx-large')
    plt.xticks(x, x)
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--view_num', type=int, required=True)
    parser.add_argument('--test_data_path', type=str, required=True)
    parser.add_argument('--observation_space_dim', type=int, required=True)
    parser.add_argument('--step_size', type=int, required=True)
    parser.add_argument('--is_log', type=int, default=0)
    parser.add_argument('--log_path', type=str, required=True)
    parser.add_argument('--test_random', type=int, default=0)
    parser.add_argument('--test_ideal', type=int, default=0)
    parser.add_argument('--test_information_gain', type=int, default=0)
    parser.add_argument('--test_rlnbv', type=int, default=0)
    parser.add_argument('--test_pcnbv', type=int, default=0)
    parser.add_argument('--pcnbv_action_path', type=str, default=None)
    parser.add_argument('--dqn_model_path', type=str, default=None)
    parser.add_argument('--is_load_test_rlt', type=int, default=0)
    parser.add_argument('--test_rlt_path', type=str, default=None)
    parser.add_argument('--is_resume', type=int, default=0)
    parser.add_argument('--save_detail', type=int, default=0)
    parser.add_argument('--detail_fold', type=str, required=True)
    parser.add_argument('--loop_num', type=int, default=1)
    args = parser.parse_args()

    logger = logging.getLogger(args.log_path)
    logger.setLevel(logging.DEBUG)
    log_format = logging.Formatter('[%(asctime)s] [%(levelname)s] %(message)s', '%Y-%m-%d %H:%M:%S')
    shell_handle = logging.StreamHandler()
    shell_handle.setFormatter(log_format)
    shell_handle.setLevel(logging.DEBUG)
    logger.addHandler(shell_handle)
    if args.is_log == 1:
        file_handle = logging.FileHandler(args.log_path)
        file_handle.setFormatter(log_format)
        file_handle.setLevel(logging.DEBUG)
        logger.addHandler(file_handle)

    test_env = envs.rl_nbv_env.PointCloudNextBestViewEnv(data_path=args.test_data_path,
                                                         view_num=args.view_num,
                                                         observation_space_dim=args.observation_space_dim,
                                                         log_level=logging.ERROR)
    if args.test_rlnbv == 1:
        policy_kwargs = dict(features_extractor_class=models.pointnet2_cls_ssg.PointNetFeatureExtraction,
                             features_extractor_kwargs=dict(features_dim=128),
                             optimizer_class=optim.adamw.AdamW)
        g_dqn_model = stable_baselines3.DQN.load(path=args.dqn_model_path, 
                                                 env=test_env, 
                                                 policy_kwargs=policy_kwargs, 
                                                 policy="MultiInputPolicy",
                                                 device='cuda:1')

    results = {}
    algorithms = {}
    ig_adapter = None
    pcnbv_adapter = None
    if args.test_random == 1:
        algorithms["Random"] = RandomExplore(test_env, logger, args)
    if args.test_ideal == 1:
        algorithms["Ideal"] = IdealExplore(test_env, logger, args)
    if args.test_rlnbv == 1:
        algorithms["DQN"] = DQNExplore(test_env, logger, args)
    if args.test_information_gain == 1:
        ig_adapter = InformationGainAdapter(test_env, logger, args)
        algorithms["IG"] = ig_adapter.IGExplore
    if args.test_pcnbv == 1:
        pcnbv_adapter = PCNBVAdapter(logger, args)
        algorithms["PC-NBV"] = pcnbv_adapter.PCNBVExplore

    # test all algorithms
    for algorithm_name, algorithm in algorithms.items():
        detail_results = {}
        logger.info("start algorithm: {}".format(algorithm_name))
        model_size = test_env.shapenet_reader.model_num
        logger.info("model size: {}".format(model_size))
        # for debugging
        # model_size = 5
        average_coverage = np.zeros(args.step_size)
        model_set = set()
        # Counting the time spent per NBV
        total_time = 0.0
        icnt = 0
        if algorithm_name == "IG" and args.is_resume == 1:
            if os.path.exists("./benchmark_ac.txt"):
                with open("./benchmark_ac.txt", "r", encoding="utf-8") as f:
                    lines = f.readlines()
                    for idx in range(args.step_size):
                        average_coverage[idx] = float(lines[idx].strip('\n'))
                        logger.info("history average coverage[{}]: {}".format(idx, average_coverage[idx]))
            if os.path.exists("./benchmark_model.txt"):
                with open("./benchmark_model.txt") as f:
                    for line in f.readlines():
                        model_set.add(line.strip('\n'))
        init_step = 0
        for loop_id in range(args.loop_num):
            logger.info("init_step: {}".format(init_step))
            init_step = 0
            for model_id in range(model_size):
                detail_result = np.zeros(args.step_size)
                obs = test_env.reset(init_step=init_step)
                cur_model_name = test_env.shapenet_reader.cur_model_name
                logger.info("handle model: {}".format(cur_model_name))
                if algorithm_name == "IG" and args.is_resume == 1:
                    if cur_model_name in model_set:
                        logger.info("skip model: {}".format(cur_model_name))
                        init_step = (init_step + 1) % args.view_num
                        continue
                init_step = (init_step + 1) % args.view_num
                average_coverage[0] += test_env.current_coverage
                detail_result[0] = test_env.current_coverage
                for step_id in range(args.step_size - 1):
                    s_time = time.time()
                    action = algorithm(obs)
                    e_time = time.time()
                    total_time += (e_time - s_time)
                    icnt += 1
                    obs, rewards, dones, info = test_env.step(action)
                    average_coverage[step_id + 1] += info["current_coverage"]
                    detail_result[step_id + 1] = info["current_coverage"]
                    if (step_id == args.step_size - 2) and (info["current_coverage"] <= 0.9):
                        logger.error("model name: {}, step: {}, coverage: {}".format(test_env.model_name, args.step_size, info["current_coverage"]))
                if algorithm_name == "IG" and args.is_resume == 1:
                    with open("./benchmark_model.txt", "a+", encoding="utf-8") as f:
                        f.write("{}\n".format(cur_model_name))
                    with open("./benchmark_ac.txt", "w+", encoding="utf-8") as f:
                        for idx in range(args.step_size):
                            f.write("{}\n".format(average_coverage[idx]))
                        f.write("{}\n".format(model_id))
                if loop_id == 0:
                    detail_results[cur_model_name] = detail_result
                else:
                    detail_results[cur_model_name] = detail_results[cur_model_name] + detail_result
        logger.info("{} per NBV time: {:.3f}".format(algorithm_name, total_time/icnt))
        average_coverage = average_coverage / (model_size * args.loop_num)
        average_coverage = average_coverage * 100
        results[algorithm_name] = average_coverage
        
        # save surface coverage each step
        if args.save_detail == 1:
            log_file_name = "./verify_rlt/{}/".format(args.detail_fold) + algorithm_name + "_detail.rlt"
            print("[INFO] Saving: {}".format(log_file_name))
            with open(log_file_name, "a+", encoding="utf-8") as f:
                print("[INFO] Saving: {}".format(log_file_name))
                for key, value in detail_results.items():
                    f.write("{}: ".format(key))
                    for coverage in value:
                        f.write("{:.2f}, ".format(coverage / args.loop_num))
                    f.write("\n")

    for key, value in results.items():
        logger.info("--------- {} ---------".format(key))
        for step, coverage in enumerate(value):
            logger.info("[{}] {:.2f}".format(step, coverage))

    with open("average_coverage.txt", "a+", encoding="utf-8") as f:
        for key, value in results.items():
            f.write("{}: ".format(key))
            for coverage in value:
                f.write("{:.2f}, ".format(coverage))
            f.write("\n")
    
    if args.is_load_test_rlt == 1:
        with open(args.test_rlt_path, 'r') as f:
            lines=f.readlines()
            title = ""
            for index, line in enumerate(lines):
                if index == 0:
                    title = str(line)
                    continue
                line = line.split(':')
                if len(line) < 2:
                    break
                algorithm_name = line[0]
                line = line[1]
                line = line.split(',')
                if len(line) < args.step_size:
                    logger.error("{} step size error".format(algorithm_name))
                average_coverage = np.zeros(args.step_size)
                for i in range(args.step_size):
                    average_coverage[i] = float(line[i])
                results[algorithm_name] = average_coverage
            plot_coverage_results(results, title=title)
    else:
        plot_coverage_results(results)

