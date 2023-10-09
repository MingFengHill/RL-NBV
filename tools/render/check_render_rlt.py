import argparse
import numpy as np
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True)
    args = parser.parse_args()

    if not os.path.exists(args.data_path):
        print("[ERRO] data path is not exist")
        exit

    model_list = os.listdir(args.data_path)
    is_pass = True
    failed_model_set = set()

    for model in model_list:
        if model == "intrinsics.txt":
            continue
    
        model_path = os.path.join(args.data_path, model)
        if not os.path.isdir(model_path):
            print("[ERRO] model path: {} is not dir".format(model_path))
            is_pass = False
            failed_model_set.add(model)
            continue

        file_list = os.listdir(model_path)
        file_set = set(file_list)

        if "model.pcd" not in file_set:
            print("[ERRO] model path: {} miss model.pcd".format(model_path))
            is_pass = False
            failed_model_set.add(model)
        for i in range(33):
            if "{}.pcd".format(i) not in file_set:
                failed_model_set.add(model)
                print("[ERRO] model path: {}.pcd miss model.pcd".format(i))
                is_pass = False
    if is_pass is True:
        print("[INFO] data check pass")
    else:
        print("[INFO] data check failed")
        print("[ERRO] failed model set: {}".format(failed_model_set))

