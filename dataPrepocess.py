import os
import random
import cv2
import shutil
import pandas as pd

def get_files(folder):
    real_file_list = []
    replay_file_list = []
    for dir_, _, files in os.walk(folder):
        for file_name in files:
            rel_dir = os.path.relpath(dir_, folder)
            rel_file = os.path.join(folder,rel_dir, file_name)
            if 'real' in rel_dir:
                real_file_list.append(rel_file)
            else:
                replay_file_list.append(rel_file)
    return real_file_list, replay_file_list

def split_train_test(file_list, ratio=0.75):
    num_test = int(len(file_list)*ratio) 
    random.shuffle(file_list)
    train_files = file_list[:num_test]
    test_files = file_list[num_test:]
    return test_files, train_files

def create_csv(data_path, train_csv_name, test_csv_name):
    if not os.path.exists(data_path):
        print(f'{data_path} not existed!')
        return
    real_files, replay_files = get_files(os.path.join(data_path))
    real_test_files, real_train_files = split_train_test(real_files)
    replay_test_files, replay_train_files = split_train_test(replay_files)
    train_files = real_train_files + replay_train_files
    test_files = real_test_files + replay_test_files
    print('Number of train images: ',len(train_files))
    print('Number of test images: ', len(test_files))
    train_data = []
    test_data = []
    for f in train_files:
        if 'real' in f:
            data = [f, 1.0]
        else:
            data = [f, 0.0]
        train_data.append(data)
    for f in test_files:
        if 'real' in f:
            data = [f, 1.0]
        else:
            data = [f, 0.0]
        test_data.append(data)
    random.shuffle(train_data)
    random.shuffle(test_data)
    train_df = pd.DataFrame(train_data, columns=['name', 'label'])
    test_df = pd.DataFrame(test_data, columns=['name', 'label'])
    train_df.to_csv(train_csv_name)
    test_df.to_csv(test_csv_name)


    # print(data_folder)
    # print(real_files[:5])
    # print(replay_files[:5])

create_csv('./new_compose', 'train_compose.csv', 'test_compose.csv')

