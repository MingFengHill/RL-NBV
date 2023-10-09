import os
import sys
import argparse
import time
import open3d as o3d
import numpy as np

def covert_obj_2_pcd_open3d(model_obj_path, model_pcd_path):
    mesh = o3d.io.read_triangle_mesh(model_obj_path)
    mesh.compute_vertex_normals()
    pcd = mesh.sample_points_uniformly(number_of_points=16384)
    # o3d.visualization.draw_geometries([pcd])
    o3d.io.write_point_cloud(model_pcd_path, pcd)


# https://github.com/isl-org/Open3D/issues/4362
def delete_texture(mode_path):
    file_list = os.listdir(mode_path)
    for file in file_list:
        if not file.endswith(".obj"):
            file_path = os.path.join(mode_path, file)
            if os.path.exists(file_path):
                os.system("rm -rf {}".format(file_path))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--output_path', type=str, required=True)
    parser.add_argument('--use_open3d', type=int, required=True)
    parser.add_argument('--delete_texture', type=int, required=True)
    args = parser.parse_args()
    
    # check input arguments
    if not os.path.exists(args.data_path):
        print("[ERRO] data path is not exist")
        exit
    if not os.path.exists(args.output_path):
        print("[ERRO] output path is not exist")
        exit

    if args.use_open3d == 0:  
        # build the project
        os.system("./build.sh")
        if not os.path.exists("./build/mesh_sampling"):
            print("[ERRO] build failed")
            exit

    # sample models
    model_list = os.listdir(args.data_path)
    model_size = len(model_list)
    print("-- begin sample, model size: {} --".format(model_size))
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

        model_output_path = os.path.join(args.output_path, model)
        if not os.path.exists(model_output_path):
            os.makedirs(model_output_path, exist_ok=True)

        # Open3D reading texture error
        if args.delete_texture == 1:
            model_path = os.path.join(args.data_path, model)
            delete_texture(model_path)
        
        model_obj_path = os.path.join(args.data_path, model, "model.obj")
        model_pcd_path = os.path.join(model_output_path, "model.pcd")
        if args.use_open3d == 1:
            covert_obj_2_pcd_open3d(model_obj_path, model_pcd_path)
        else:  
            os.system("./build/mesh_sampling {} {} -no_vis_result -n_samples 16384 -write_normals -no_vox_filter".format(model_obj_path, model_pcd_path))
    print("/n")

