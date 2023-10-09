# Author: Wentao Yuan (wyuan1@cs.cmu.edu) 05/31/2018
# Modified by Rui Zeng 07/12/2020

import Imath
import OpenEXR
import argparse
import array
import numpy as np
import os
from open3d import *
# from open3d.io import *
import matplotlib.pyplot as plt
import sys
import pdb
import time

def read_exr(exr_path, height, width):
    file = OpenEXR.InputFile(exr_path)
    depth_arr = array.array('f', file.channel('R', Imath.PixelType(Imath.PixelType.FLOAT)))
    depth = np.array(depth_arr).reshape((height, width))
    depth[depth < 0] = 0
    depth[np.isinf(depth)] = 0
    return depth


def depth2pcd(depth, intrinsics, pose):
    inv_K = np.linalg.inv(intrinsics)
    inv_K[2, 2] = -1
    depth = np.flipud(depth)
    y, x = np.where(depth > 0)
    # image coordinates -> camera coordinates
    points = np.dot(inv_K, np.stack([x, y, np.ones_like(x)] * depth[y, x], 0))
    # camera coordinates -> world coordinates
    points = np.dot(pose, np.concatenate([points, np.ones((1, points.shape[1]))], 0)).T[:, :3]
    return points


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--output_path', type=str, required=True)
    parser.add_argument('--num_scans', type=int, default=33)
    args = parser.parse_args()

    intrinsics = np.loadtxt(os.path.join(args.output_path, 'intrinsics.txt'))
    width = int(intrinsics[0, 2] * 2)
    height = int(intrinsics[1, 2] * 2)

    model_set = set()
    if os.path.exists("./render_pcd.txt"):
        with open("./render_pcd.txt") as f:
            for line in f.readlines():
                model_set.add(line.strip('\n'))

    model_list = os.listdir(args.data_path)
    model_size = len(model_list)
    print("-- begin covert exr to pcd, model size: {} --".format(model_size))
    iter = 0
    start = time.perf_counter()
    for model in model_list:
        # 打印进度条
        percentage = ((iter + 1) / model_size) * 100
        finished = "*" * int(percentage)
        rest = "." * (100 - int(percentage))
        duration = time.perf_counter() - start
        print("\r{:^3.0f}%[{}->{}]{:.2f}s".format(percentage, finished, rest, duration), end = "")
        iter += 1

        # Continue after interruption
        if model in model_set:
            continue
        with open("./render_pcd.txt", "a+", encoding="utf-8") as f:
            f.write("{}\n".format(model))

        pcd_path = os.path.join(args.output_path, model)
        for i in range(args.num_scans):
            exr_path = os.path.join(pcd_path, 'exr', '%d.exr' % i)
            pose_path = os.path.join(pcd_path, 'pose', '%d.txt' % i)   

            depth = read_exr(exr_path, height, width)
            depth_img = open3d.geometry.Image(np.uint16(depth * 100000))
            open3d.io.write_image(os.path.join(pcd_path, '%d.png' % i), depth_img) 

            pose = np.loadtxt(pose_path)
            points = depth2pcd(depth, intrinsics, pose)
            if (points.shape[0] == 0):
                points = np.array([(1.0,1.0,1.0)])
            pcd = open3d.geometry.PointCloud()
            pcd.points = open3d.utility.Vector3dVector(points)
            open3d.io.write_point_cloud(os.path.join(pcd_path, '%d.pcd' % i), pcd)
    print("/n")
