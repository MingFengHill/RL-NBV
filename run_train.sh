rm -rf env_*log
rm -rf model_detail.log
python train.py --data_path /home/wang/data/rl_nbv/output/train\
                --verify_data_path /home/wang/data/verify/output\
                --test_data_path /home/wang/data/rl_nbv/output/novel\
                --output_file train_result.txt\
                --view_num 33\
                --observation_space_dim 1024\
                --step_size 10\
                --is_profile 0\
                --is_vec_env 0\
                --is_transform 1\
                --pretrained_model_path ./models/pretrained/pointnet2_ssg_wo_normals/checkpoints/best_model.pth\
                --is_freeze_fe 0\
                --is_save_model 1\
                --is_save_replay_buffer 0\
                --is_load_replay_buffer 1\
                --replay_buffer_path ideal_policy_d1024_s10_1e_nra_3997\
                --is_ratio_reward 1