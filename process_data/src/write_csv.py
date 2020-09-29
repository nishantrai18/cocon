import os
import csv
import glob
import sys

import pandas as pd

from joblib import delayed, Parallel
from tqdm import tqdm


def str2bool(s):
    """Convert string to bool (in argparse context)."""
    if s.lower() not in ['true', 'false']:
        raise ValueError('Need bool; got %r' % s)
    return {'true': True, 'false': False}[s.lower()]


def write_list(data_list, path, ):
    with open(path, 'w') as f:
        writer = csv.writer(f, delimiter=',')
        for row in data_list:
            if row: writer.writerow(row)
    print('split saved to %s' % path)


def main_UCF101(f_root, splits_root, csv_root='../data/ucf101/'):
    '''generate training/testing split, count number of available frames, save in csv'''
    if not os.path.exists(csv_root): os.makedirs(csv_root)
    for which_split in [1,2,3]:
        train_set = []
        test_set = []
        train_split_file = os.path.join(splits_root, 'trainlist%02d.txt' % which_split)
        with open(train_split_file, 'r') as f:
            for line in f:
                vpath = os.path.join(f_root, line.split(' ')[0][0:-4]) + '/'
                train_set.append([vpath, len(glob.glob(os.path.join(vpath, '*.jpg')))])

        test_split_file = os.path.join(splits_root, 'testlist%02d.txt' % which_split)
        with open(test_split_file, 'r') as f:
            for line in f:
                vpath = os.path.join(f_root, line.rstrip()[0:-4]) + '/'
                test_set.append([vpath, len(glob.glob(os.path.join(vpath, '*.jpg')))])

        write_list(train_set, os.path.join(csv_root, 'train_split%02d.csv' % which_split))
        write_list(test_set, os.path.join(csv_root, 'test_split%02d.csv' % which_split))


def main_HMDB51(f_root, splits_root, csv_root='../data/hmdb51/'):
    '''generate training/testing split, count number of available frames, save in csv'''
    if not os.path.exists(csv_root): os.makedirs(csv_root)
    for which_split in [1,2,3]:
        train_set = []
        test_set = []
        split_files = sorted(glob.glob(os.path.join(splits_root, '*_test_split%d.txt' % which_split)))
        assert len(split_files) == 51
        for split_file in split_files:
            action_name = os.path.basename(split_file)[0:-16]
            with open(split_file, 'r') as f:
                for line in f:
                    video_name = line.split(' ')[0]
                    _type = line.split(' ')[1]
                    vpath = os.path.join(f_root, action_name, video_name[0:-4]) + '/'
                    if _type == '1':
                        train_set.append([vpath, len(glob.glob(os.path.join(vpath, '*.jpg')))])
                    elif _type == '2':
                        test_set.append([vpath, len(glob.glob(os.path.join(vpath, '*.jpg')))])

        write_list(train_set, os.path.join(csv_root, 'train_split%02d.csv' % which_split))
        write_list(test_set, os.path.join(csv_root, 'test_split%02d.csv' % which_split))


def main_JHMDB(f_root, splits_root, csv_root='../data/jhmdb/'):
    '''generate training/testing split, count number of available frames, save in csv'''
    if not os.path.exists(csv_root): os.makedirs(csv_root)
    for which_split in [1,2,3]:
        train_set = []
        test_set = []
        split_files = sorted(glob.glob(os.path.join(splits_root, '*_test_split%d.txt' % which_split)))
        assert len(split_files) == 21
        for split_file in split_files:
            action_name = os.path.basename(split_file)[0:-16]
            with open(split_file, 'r') as f:
                for line in f:
                    video_name = line.split(' ')[0]
                    _type = line.split(' ')[1].strip('\n')
                    vpath = os.path.join(f_root, action_name, video_name[0:-4]) + '/'
                    if _type == '1':
                        train_set.append([vpath, len(glob.glob(os.path.join(vpath, '*.jpg')))])
                    elif _type == '2':
                        test_set.append([vpath, len(glob.glob(os.path.join(vpath, '*.jpg')))])

        write_list(train_set, os.path.join(csv_root, 'train_split%02d.csv' % which_split))
        write_list(test_set, os.path.join(csv_root, 'test_split%02d.csv' % which_split))


### For Kinetics ###
def get_split(root, split_path, mode):
    print('processing %s split ...' % mode)
    print('checking %s' % root)
    split_list = []
    split_content = pd.read_csv(split_path).iloc[:,0:4]
    split_list = Parallel(n_jobs=64)\
                 (delayed(check_exists)(row, root) \
                 for i, row in tqdm(split_content.iterrows(), total=len(split_content)))
    return split_list

missedCnt = 0

def check_exists(row, root):
    global missedCnt

    dirname = '_'.join([row['youtube_id'], '%06d' % row['time_start'], '%06d' % row['time_end']])
    full_dirname = os.path.join(root, row['label'], dirname)
    # replace spaces with underscores
    full_dirname = full_dirname.replace(' ', '_')
    if os.path.exists(full_dirname):
        n_frames = len(glob.glob(os.path.join(full_dirname, '*.jpg')))
        return [full_dirname, n_frames]
    else:
        missedCnt += 1
        return None

def main_Kinetics400(mode, k400_path, f_root, csv_root='../data/kinetics400'):
    global missedCnt
    missedCnt = 0

    train_split_path = os.path.join(k400_path, 'kinetics-400_train.csv')
    val_split_path = os.path.join(k400_path, 'kinetics-400_val.csv')
    test_split_path = os.path.join(k400_path, 'kinetics-400_test.csv')

    if not os.path.exists(csv_root):
        os.makedirs(csv_root)

    if mode == 'train':
        train_split = get_split(os.path.join(f_root, 'train'), train_split_path, 'train')
        write_list(train_split, os.path.join(csv_root, 'train_split.csv'))
    elif mode == 'val':
        val_split = get_split(os.path.join(f_root, 'val'), val_split_path, 'val')
        write_list(val_split, os.path.join(csv_root, 'val_split.csv'))
    elif mode == 'test':
        test_split = get_split(f_root, test_split_path, 'test')
        write_list(test_split, os.path.join(csv_root, 'test_split.csv'))
    else:
        raise IOError('wrong mode')

    print("Total files missed:", missedCnt)


import argparse

if __name__ == '__main__':
    # f_root is the frame path
    # edit 'your_path' here: 

    #dataset_dir = '/vision/u/nishantr/data/'
    # dataset_dir = '/scr/nishantr/data/'

    parser = argparse.ArgumentParser()
    parser.add_argument('--ucf101', default=False, type=str2bool)
    parser.add_argument('--jhmdb', default=False, type=str2bool)
    parser.add_argument('--hmdb51', default=False, type=str2bool)
    parser.add_argument('--kinetics', default=False, type=str2bool)
    parser.add_argument('--dataset_path', default='/scr/nishantr/data', type=str)
    args = parser.parse_args()

    dataset_path = args.dataset_path

    if args.ucf101:
        main_UCF101(f_root=dataset_path + 'ucf101/frame',
                    splits_root=dataset_path + 'ucf101/splits_classification')

    if args.jhmdb:
        main_JHMDB(f_root=dataset_path + 'jhmdb/frame', splits_root=dataset_path + 'jhmdb/splits')

    if args.hmdb51:
        main_HMDB51(f_root=dataset_path + 'hmdb/frame', splits_root=dataset_path + 'hmdb/splits/')

    if args.kinetics:
        main_Kinetics400(
            mode='train', # train or val or test
            k400_path=dataset_path + 'kinetics/splits',
            f_root=dataset_path + 'kinetics/frame256',
            csv_root='../data/kinetics400_256',
        )

        main_Kinetics400(
            mode='val', # train or val or test
            k400_path=dataset_path + 'kinetics/splits',
            f_root=dataset_path + 'kinetics/frame256',
            csv_root='../data/kinetics400_256',
        )

    # main_Kinetics400(mode='train', # train or val or test
    #                  k400_path='your_path/Kinetics',
    #                  f_root='your_path/Kinetics400_256/frame',
    #                  csv_root='../data/kinetics400_256')
