import bpy
import mathutils
import numpy as np
import os
import sys
import time
import pdb
# Usage: blender -b -P render_depth.py
# nohup ./run.sh > test.log 2>&1 & 


def setup_blender(width, height, focal_length, output_dir):
    # camera
    camera = bpy.data.objects['Camera']
    camera.data.angle = np.arctan(width / 2 / focal_length) * 2
    # camera.data.clip_end = 1.2
    # camera.data.clip_start = 0.2

    # render layer
    scene = bpy.context.scene
    scene.render.filepath = 'buffer'
    scene.render.image_settings.color_depth = '16'
    scene.render.resolution_percentage = 100
    scene.render.resolution_x = width
    scene.render.resolution_y = height

    # compositor nodes
    scene.use_nodes = True
    tree = scene.node_tree
    rl = tree.nodes.new('CompositorNodeRLayers')
    output = tree.nodes.new('CompositorNodeOutputFile')
    output.base_path = ''
    output.format.file_format = 'OPEN_EXR'
    tree.links.new(rl.outputs['Depth'], output.inputs[0])

    # remove default cube
    bpy.data.objects['Cube'].select = True
    bpy.ops.object.delete()

    return scene, camera, output


if __name__ == '__main__':
    data_path = "/home/wang/data/rl_nbv/train/"
    output_path = "/home/wang/data/rl_nbv/output/train/"
    view_space_path = "./viewspace_shapenet_33.txt"
    width = 640
    height = 480
    focal = 476

    # check input arguments
    if not os.path.exists(data_path):
        print("[ERRO] data path is not exist")
        exit
    if not os.path.exists(output_path):
        print("[ERRO] output path is not exist")
        exit

    model_set = set()
    if os.path.exists("./render_exr.txt"):
        with open("./render_exr.txt") as f:
            for line in f.readlines():
                model_set.add(line.strip('\n'))

    viewspace = np.loadtxt(view_space_path)
    scene, camera, output = setup_blender(width, height, focal, output_path)
    intrinsics = np.array([[focal, 0, width / 2], [0, focal, height / 2], [0, 0, 1]])
    open('blender.log', 'w+').close()
    np.savetxt(os.path.join(output_path, 'intrinsics.txt'), intrinsics, '%f')

    model_list = os.listdir(data_path)
    model_size = len(model_list)
    print("-- begin render, model size: {} --".format(model_size))
    iter = 0
    start = time.time()
    for model in model_list:
        # 打印进度条
        percentage = ((iter + 1) / model_size) * 100
        finished = "*" * int(percentage)
        rest = "." * (100 - int(percentage))
        duration = time.time() - start
        print("\r{:^3.0f}%[{}->{}]".format(percentage, finished, rest), time.strftime("%Hh:%Mm:%Ss", time.gmtime(duration)),end = "")
        iter += 1

        # Continue after interruption
        if model in model_set:
            continue
        with open("./render_exr.txt", "a+", encoding="utf-8") as f:
            f.write("{}\n".format(model))

        model_output_path = os.path.join(output_path, model)
        if not os.path.exists(model_output_path):
            print("[ERRO] model output path: {} is not exist".format(model_output_path))
            exit
        exr_dir = os.path.join(model_output_path, 'exr')
        pose_dir = os.path.join(model_output_path, 'pose')
        os.makedirs(exr_dir, exist_ok=True)
        os.makedirs(pose_dir, exist_ok=True)

        # Redirect output to log file
        old_os_out = os.dup(1)
        os.close(1)
        os.open('blender.log', os.O_WRONLY)

        # Import mesh model
        model_obj_path = os.path.join(data_path, model, "model.obj")
        bpy.ops.import_scene.obj(filepath=model_obj_path)
        # Rotate model by 90 degrees around x-axis (z-up => y-up) to match ShapeNet's coordinates
        bpy.ops.transform.rotate(value=-np.pi / 2, axis=(1, 0, 0))  

        # Render
        for i in range(viewspace.shape[0]):
            scene.frame_set(i)
            cam_pose = mathutils.Vector((viewspace[i][0], viewspace[i][1], viewspace[i][2]))
            center_pose = mathutils.Vector((0, 0, 0))
            direct = center_pose - cam_pose
            rot_quat = direct.to_track_quat('-Z', 'Y')
            camera.rotation_euler = rot_quat.to_euler()
            camera.location = cam_pose
            pose_matrix = camera.matrix_world
            output.file_slots[0].path = os.path.join(exr_dir, '#.exr')
            bpy.ops.render.render(write_still=True)
            np.savetxt(os.path.join(pose_dir, '%d.txt' % i), pose_matrix, '%f')
        
        # Clean up
        bpy.ops.object.delete()
        for m in bpy.data.meshes:
            bpy.data.meshes.remove(m)
        for m in bpy.data.materials:
            m.user_clear()
            bpy.data.materials.remove(m)    

        # Show time
        os.close(1)
        os.dup(old_os_out)
        os.close(old_os_out)


