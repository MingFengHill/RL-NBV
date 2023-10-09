# **RL-NBV**
This repository hosts the implementation of RL-NBV which is a deep reinforcement learning based Next-Best-View method for unknown object reconstruction.
<p>    
<img  src="./img/drl_framework.jpg"  width="300" />
</p>

## Training Data
The training data contains the complete point cloud of the model and partial point clouds captured in candidate views.

1.The complete point cloud is generated using the tool in the `./tools/sample/` directory.
```
./tools/sample/run.sh
```

2.Partial point clouds are generated using the tool in the `./tools/render/` directory
```
blender -b -P generate_render_exr.py
python generate_render_pcd.py --data_path [data path]\
                              --output_path [output path]\
```

3.We provide two testing datasets, **complex shaped object dataset** and **mechanical components dataset** in the `./data` directory.
## Train
```
./run_train.sh
```
## Test
```
./run_benchmark.sh
```

## Acknowledgment
This library was inspired by the following project:

[PC-NBV](https://github.com/Smile2020/PC-NBV)

[rpg_ig_active_reconstruction](https://github.com/uzh-rpg/rpg_ig_active_reconstruction)

[PointNet++ implementation](https://github.com/yanx27/Pointnet_Pointnet2_pytorch)

[gym](https://github.com/openai/gym)

[stable-baselines3](https://github.com/DLR-RM/stable-baselines3)

