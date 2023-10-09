import bpy
import mathutils
import numpy as np
import os
import sys
import time
import pdb
# Usage: blender -b -P generate_viewspace_poses.py


if __name__ == '__main__':
    view_space_path = "./viewspace_shapenet_33.txt"
    viewspace = np.loadtxt(view_space_path)
    quat_array = []

    for i in range(viewspace.shape[0]):
        cam_pose = mathutils.Vector((viewspace[i][0], viewspace[i][1], viewspace[i][2]))
        center_pose = mathutils.Vector((0, 0, 0))
        # direct = center_pose - cam_pose
        direct = cam_pose - center_pose
        rot_quat = direct.to_track_quat('-Z', 'Y')
        quat_array.append([rot_quat.x, rot_quat.y, rot_quat.z, rot_quat.w])
    
    quat_array = np.asarray(quat_array)
    poses = np.concatenate([viewspace, quat_array], axis=1)

    np.savetxt("poses.txt", poses)
    