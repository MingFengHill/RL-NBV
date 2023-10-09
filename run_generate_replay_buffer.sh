rm -rf env_*.log
python generate_replay_buffer.py --data_path /home/wang/data/rl_nbv/output/train\
                                 --view_num 33\
                                 --observation_space_dim 1024\
                                 --step_size 10\
                                 --buffer_size 1000000\
                                 --save_path "ideal_policy"\
                                 --env_num 1\
                                 --is_add_negative_exp 1\
                                 --is_ratio_reward 0