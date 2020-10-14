import torch
from torch.utils import data
from torchvision import transforms
import os
import sys
import time
import pickle
import glob
import csv
import scipy.io
import pandas as pd
import numpy as np
import cv2
import random

import model_utils as mu

sys.path.append('../utils')

from copy import deepcopy
from augmentation import *
from tqdm import tqdm
from joblib import Parallel, delayed


def pil_loader(path):
    img = Image.open(path)
    return img.convert('RGB')


toTensor = transforms.ToTensor()
toPILImage = transforms.ToPILImage()
def flow_loader(path):
    try:
        img = Image.open(path)
    except:
        return None
    return toTensor(img)


class BaseDataloader(data.Dataset):

    def __init__(
        self,
        mode,
        transform,
        seq_len,
        num_seq,
        downsample,
        which_split,
        vals_to_return,
        sampling_method,
        dataset,
        debug=False,
        postfix=''
    ):
        super(BaseDataloader, self).__init__()

        self.dataset = dataset
        self.mode = mode
        self.debug = debug
        self.transform = transform
        self.seq_len = seq_len
        self.num_seq = num_seq
        self.downsample = downsample
        self.which_split = which_split
        # Describes which particular items to return e.g. ["imgs", "poses", "labels"]
        self.vals_to_return = set(vals_to_return)
        self.sampling_method = sampling_method
        self.num_classes = mu.get_num_classes(self.dataset if not postfix else '-'.join((self.dataset, postfix)))

        assert not ((self.dataset == "hmdb51") and ("poses" in self.vals_to_return)), \
            "HMDB51 does not support poses yet"

        assert not ((self.dataset == "jhmdb") and ("flow" in self.vals_to_return)), \
            "JHMDB does not support flow yet"

        if self.sampling_method == "random":
            assert "imgs" not in self.vals_to_return, \
                "Invalid sampling method provided for imgs: {}".format(self.sampling_method)

        # splits
        mode_str = "test" if ((mode == 'val') or (mode == 'test')) else mode
        mode_split_str = '/' + mode_str + '_split%02d.csv' % self.which_split

        if "kinetics400" in dataset:
            mode_str = "val" if ((mode == 'val') or (mode == 'test')) else mode
            mode_split_str = '/' + mode_str + '_split.csv'

        split = '../data/' + self.dataset + mode_split_str

        if "panasonic" in dataset:
            # FIXME: change when access is changed
            split = os.path.join('../data', '{}_split{}.csv'.format(mode, '_' + postfix if postfix else ''))
            # maximum 15 values
            video_info = pd.read_csv(split, header=None, names=list(range(20)))
        else:
            video_info = pd.read_csv(split, header=None)

        # poses_mat_dict: vpath to poses_mat
        self.poses_dict = {}

        # get action list
        self.action_dict_encode = {}
        self.action_dict_decode = {}

        action_file = os.path.join('../data/' + self.dataset, 'classInd{}.txt'.format('_' + postfix if postfix else ''))
        action_df = pd.read_csv(action_file, sep=' ', header=None)
        for _, row in action_df.iterrows():
            act_id, act_name = row
            act_id = int(act_id) - 1 # let id start from 0
            assert 0 <= act_id < self.num_classes, "Incorrect class_id: {}".format(act_id)
            self.action_dict_decode[act_id] = act_name
            self.action_dict_encode[act_name] = act_id

        drop_idx = set()
        # track duplicate categories
        dup_cat_dict = dict()
        # filter out too short videos:
        for idx, row in tqdm(video_info.iterrows(), total=len(video_info)):
            # FIXME: make dataloader more modular. This only works for panasonic data
            num_views = int(len([i for i in np.array(row) if i == i]) / 4)
            # drop indices with no ego-view
            view_names = [row[i * 4].split('/')[-1].split('_')[2] for i in range(num_views)]
            if not 'v000' in view_names:
                drop_idx.add(idx)
                continue
            # drop indices with only a single view
            if num_views < 2:
                drop_idx.add(idx)
                continue
            # drop indices with multiple categories
            p, r, _, a = row[0].split('/')[-1].split('_')
            s = row[1]
            e = row[2]
            key = (p, r, a, s, e)
            if key in dup_cat_dict:
                # drop duplicates
                drop_idx.add(idx)
                drop_idx.add(dup_cat_dict[key])
            dup_cat_dict[key] = idx

            # FIXME: dropping indices with > 1 categories here for now. Might need to change

            vpath, vstart, vend, vname = row[:4]
            vlen = int(vend - vstart + 1)
            if self.sampling_method == 'disjoint':
                if vlen-self.num_seq*self.seq_len*self.downsample <= 0:
                    drop_idx.add(idx)
            else:
                if vlen <= 0:
                    drop_idx.add(idx)
        self.video_info = video_info.drop(list(drop_idx), axis=0)

        # FIXME: panasonic data don't need val sampling here. Try making this more modular!
        # if self.debug:
        #     self.video_info = self.video_info.sample(frac=0.0025, random_state=42)
        # elif self.mode == 'val':
        #     self.video_info = self.video_info.sample(frac=0.3)
        #     # self.video_info = self.video_info.head(int(0.3 * len(self.video_info)))

        self.idx_sampler = None
        if self.sampling_method == "dynamic":
            self.idx_sampler = self.idx_sampler_dynamic
        if self.sampling_method == "disjoint":
            self.idx_sampler = self.idx_sampler_disjoint
        elif self.sampling_method == "random":
            self.idx_sampler = self.idx_sampler_random

        if self.mode == 'test':
            self.idx_sampler = self.idx_sampler_test

        if mu.FlowMode in self.vals_to_return:
            self.setup_flow_modality()

        # shuffle not required due to external sampler

    def setup_flow_modality(self):
        '''Can be overriden in the derived classes'''
        vpath, _ = self.video_info.iloc[0]
        vpath = vpath.rstrip('/')
        base_dir = vpath.split(self.dataset)[0]
        print("Base dir for flow:", base_dir)
        self.flow_base_path = os.path.join(base_dir, 'flow', self.dataset + '_flow/')

    def idx_sampler_test(self, seq_len, num_seq, vlen, vpath):
        '''
        sample index uniformly from a video
        '''

        downsample = self.downsample
        if (vlen - (num_seq * seq_len * self.downsample)) <= 0:
            downsample = ((vlen - 1) / (num_seq * seq_len * 1.0)) * 0.9

        seq_idx = np.expand_dims(np.arange(num_seq), -1) * downsample * seq_len
        seq_idx_block = seq_idx + np.expand_dims(np.arange(seq_len), 0) * downsample
        seq_idx_block = seq_idx_block.astype(int)

        return [seq_idx_block, vpath]

    def idx_sampler_dynamic(self, seq_len, num_seq, vlen, vpath, idx_offset=0, start_idx=-1):
        '''sample index from a video'''
        downsample = self.downsample
        if (vlen - (num_seq * seq_len * self.downsample)) <= 0:
            downsample = ((vlen - 1) / (num_seq * seq_len * 1.0)) * 0.9

        n = 1
        if start_idx < 0:
            try:
                start_idx = np.random.choice(range(vlen - int(num_seq * seq_len * downsample)), n)
            except:
                print("Error!", vpath, vlen, num_seq, seq_len, downsample, n)

        seq_idx = np.expand_dims(np.arange(num_seq), -1) * downsample * seq_len + start_idx + idx_offset
        seq_idx_block = seq_idx + np.expand_dims(np.arange(seq_len), 0) * downsample
        seq_idx_block = seq_idx_block.astype(int)

        return [seq_idx_block, vpath], start_idx

    def idx_sampler_disjoint(self, seq_len, num_seq, vlen, vpath):
        '''sample index from a video'''

        if (vlen - (num_seq * seq_len * self.downsample)) <= 0:
            return None

        n = 1
        if self.mode == 'test':
            seq_idx_block = np.arange(0, vlen, self.downsample) # all possible frames with downsampling
            return [seq_idx_block, vpath]

        start_idx = np.random.choice(range(vlen - (num_seq * seq_len * self.downsample)), n)
        seq_idx = np.expand_dims(np.arange(num_seq), -1) * self.downsample * seq_len + start_idx
        # Shape num_seq x seq_len
        seq_idx_block = seq_idx + np.expand_dims(np.arange(seq_len), 0) * self.downsample

        return [seq_idx_block, vpath]

    def idx_sampler_random(self, seq_len, num_seq, vlen, vpath):
        '''sample index from a video'''

        # Here we compute the max downsampling we could perform
        max_ds = ((vlen - 1) // seq_len)

        if max_ds <= 0:
            return None

        if self.mode == 'test':
            seq_idx_block = np.arange(0, vlen, self.downsample)
            # all possible frames with downsampling
            return [seq_idx_block, vpath]

        seq_idx_block = []
        for i in range(num_seq):
            rand_ds = random.randint(1, max_ds)
            start_idx = random.randint(0, vlen - (seq_len * rand_ds) - 1)
            seq_idx = np.arange(start=start_idx, stop=(start_idx + (seq_len*rand_ds)), step=rand_ds)
            seq_idx_block.append(seq_idx)

        seq_idx_block = np.array(seq_idx_block)

        return [seq_idx_block, vpath]

    def fetch_imgs_seq(self, vpath, seq_len, idx_block):
        '''Can be overriden in the derived classes'''
        img_list = [os.path.join(vpath, 'image_%05d.jpg' % (i + 1)) for i in idx_block]
        seq = [pil_loader(f) for f in img_list]
        img_t_seq = self.transform["imgs"](seq)  # apply same transform
        (IC, IH, IW) = img_t_seq[0].size()
        img_t_seq = torch.stack(img_t_seq, 0)
        img_t_seq = img_t_seq.view(self.num_seq, seq_len, IC, IH, IW).transpose(1, 2)
        return img_t_seq

    @staticmethod
    def fill_nones(l):
        l = [l[i - 1] if l[i] is None else l[i] for i in range(len(l))]
        l = [l[i - 1] if l[i] is None else l[i] for i in range(len(l))]
        try:
            nonNoneL = [item for item in l if item is not None][0]
        except:
            nonNoneL = torch.zeros((1, 256, 256))
        return [torch.zeros(nonNoneL.shape) if l[i] is None else l[i] for i in range(len(l))]

    def get_u_flow_path_list(self, vpath, idx_block):
        vid_name = os.path.basename(os.path.normpath(vpath))
        return [os.path.join(self.flow_base_path, 'u', vid_name, 'frame%06d.jpg' % (i + 1)) for i in idx_block]

    def get_v_flow_path_list(self, vpath, idx_block):
        vid_name = os.path.basename(os.path.normpath(vpath))
        return [os.path.join(self.flow_base_path, 'v', vid_name, 'frame%06d.jpg' % (i + 1)) for i in idx_block]

    def fetch_flow_seq(self, vpath, seq_len, idx_block):
        '''
        Can be overriden in the derived classes
            - TODO: implement and experiment with stack flow, later on
        '''

        u_flow_list = self.get_u_flow_path_list(vpath, idx_block)
        v_flow_list = self.get_v_flow_path_list(vpath, idx_block)

        u_seq = self.fill_nones([flow_loader(f) for f in u_flow_list])
        v_seq = self.fill_nones([flow_loader(f) for f in v_flow_list])

        seq = [toPILImage(torch.cat([u, v])) for u, v in zip(u_seq, v_seq)]
        flow_t_seq = self.transform["flow"](seq)

        (FC, FH, FW) = flow_t_seq[0].size()
        flow_t_seq = torch.stack(flow_t_seq, 0)
        flow_t_seq = flow_t_seq.view(self.num_seq, seq_len, FC, FH, FW).transpose(1, 2)

        if flow_t_seq.mean() > 0.3:
            flow_t_seq -= 0.5

        return flow_t_seq

    def fetch_fnb_flow_seq(self, vpath, seq_len, idx_block):
        pass

    def get_class_vid(self, vpath):
        return os.path.normpath(vpath).split('/')[-2:]

    def load_detectron_feature(self, fdir, idx, opt):
        # opt is either hm or seg

        shape = (192, 256)

        def load_feature(path):
            try:
                x = np.load(path)[opt]
            except:
                x = np.zeros((0, 0, 0))

            # Match non-existent values
            if x.shape[1] == 0:
                num_channels = 17 if opt == 'hm' else 1
                x = np.zeros((num_channels, shape[0], shape[1]))

            x = torch.tensor(x, dtype=torch.float) / 255.0

            # Add extra channel in case it's not present
            if len(x.shape) < 3:
                x = x.unsqueeze(0)
            return x

        suffix = 'heatmap' if opt == 'hm' else 'segmask'
        fpath = os.path.join(fdir, suffix + '_%05d.npz' % idx)
        if os.path.isfile(fpath):
            return load_feature(fpath)
        else:
            # We do not have results lower than idx=2
            idx = max(3, idx)
            # We assume having all results for every two frames
            fpath0 = os.path.join(fdir, suffix + '_%05d.npz' % (idx - 1))
            fpath1 = os.path.join(fdir, suffix + '_%05d.npz' % (idx + 1))
            # This is not guaranteed to exist
            if not os.path.isfile(fpath1):
                fpath1 = fpath0
            a0, a1 = load_feature(fpath0), load_feature(fpath1)
            try:
                a_avg = (a0 + a1) / 2.0
            except:
                a_avg = None
            return a_avg

    def fetch_kp_heatmap_seq(self, vpath, seq_len, idx_block):
        assert '/frame/' in vpath, "Incorrect vpath received: {}".format(vpath)

        feature_vpath = vpath.replace('/frame/', '/heatmaps/')
        seq = self.fill_nones([self.load_detectron_feature(feature_vpath, idx, opt='hm') for idx in idx_block])

        if len(set([x.shape for x in seq])) > 1:
            # We now know the invalid paths, so no need to print them
            # print("Invalid path:", vpath)
            seq = [seq[len(seq) // 2] for _ in seq]

        hm_t_seq = self.transform[mu.KeypointHeatmap](seq)  # apply same transform
        (IC, IH, IW) = hm_t_seq[0].size()

        hm_t_seq = hm_t_seq.view(self.num_seq, seq_len, IC, IH, IW).transpose(1, 2)
        return hm_t_seq

    def fetch_seg_mask_seq(self, vpath, seq_len, idx_block):
        assert '/frame/' in vpath, "Incorrect vpath received: {}".format(vpath)

        feature_vpath = vpath.replace('/frame/', '/segmasks/')
        seq = self.fill_nones([self.load_detectron_feature(feature_vpath, idx, opt='seg') for idx in idx_block])

        seg_t_seq = self.transform[mu.SegMask](seq)  # apply same transform
        (IC, IH, IW) = seg_t_seq[0].size()

        seg_t_seq = seg_t_seq.view(self.num_seq, seq_len, IC, IH, IW).transpose(1, 2)
        return seg_t_seq

    def __getitem__(self, index):
        vpath, vlen = self.video_info.iloc[index]
        # Remove trailing backslash if any
        vpath = vpath.rstrip('/')

        seq_len = self.seq_len
        if "tgt" in self.vals_to_return:
            seq_len = 2 * self.seq_len

        items = self.idx_sampler(seq_len, self.num_seq, vlen, vpath)
        if items is None:
            print(vpath)

        idx_block, vpath = items
        assert idx_block.shape == (self.num_seq, seq_len)
        idx_block = idx_block.reshape(self.num_seq * seq_len)

        vals = {}

        # Populate return list
        if mu.ImgMode in self.vals_to_return:
            img_t_seq = self.fetch_imgs_seq(vpath, seq_len, idx_block)
            vals[mu.ImgMode] = img_t_seq
        if mu.FlowMode in self.vals_to_return:
            flow_t_seq = self.fetch_flow_seq(vpath, seq_len, idx_block)
            vals[mu.FlowMode] = flow_t_seq
        if mu.FnbFlowMode in self.vals_to_return:
            fnb_flow_t_seq = self.fetch_fnb_flow_seq(vpath, seq_len, idx_block)
            vals[mu.FnbFlowMode] = fnb_flow_t_seq
        if mu.KeypointHeatmap in self.vals_to_return:
            hm_t_seq = self.fetch_kp_heatmap_seq(vpath, seq_len, idx_block)
            vals[mu.KeypointHeatmap] = hm_t_seq
        if mu.SegMask in self.vals_to_return:
            seg_t_seq = self.fetch_seg_mask_seq(vpath, seq_len, idx_block)
            vals[mu.SegMask] = seg_t_seq

        # Process double length target results
        if "tgt" in self.vals_to_return:
            orig_keys = list(vals.keys())
            for k in orig_keys:
                full_x = vals[k]
                vals[k] = full_x[:, :self.seq_len, ...]
                vals["tgt_" + k] = full_x[:, self.seq_len:, ...]
        if "labels" in self.vals_to_return:
            try:
                vname = vpath.split('/')[-3]
                vid = self.encode_action(vname)
            except:
                vname = vpath.split('/')[-2]
                vid = self.encode_action(vname)
            label = torch.LongTensor([vid])
            vals["labels"] = label

        # Add video index field
        vals["vnames"] = torch.LongTensor([index])

        return vals

    def __len__(self):
        return len(self.video_info)

    def encode_action(self, action_name):
        '''give action name, return category'''
        return self.action_dict_encode[action_name]

    def decode_action(self, action_code):
        '''give action code, return action name'''
        return self.action_dict_decode[action_code]


class Kinetics_3d(BaseDataloader):

    def setup_flow_modality(self):
        '''Can be overriden in the derived classes'''
        self.flow_base_path = '/data/nishantr/kinetics/fnb_frames/'

    def get_u_flow_path_list(self, vpath, idx_block):
        v_class, v_name = self.get_class_vid(vpath)
        return [os.path.join(self.flow_base_path, self.mode, v_class, v_name, 'flow_x_%05d.jpg' % (i + 1)) for i in idx_block]

    def get_v_flow_path_list(self, vpath, idx_block):
        v_class, v_name = self.get_class_vid(vpath)
        return [os.path.join(self.flow_base_path, self.mode, v_class, v_name, 'flow_y_%05d.jpg' % (i + 1)) for i in idx_block]

    def __init__(
        self,
        mode='train',
        transform=None,
        seq_len=5,
        num_seq=6,
        downsample=3,
        which_split=1,
        vals_to_return=["imgs"],
        sampling_method="dynamic",
        use_big=False,
    ):
        dataset = "kinetics400"
        if use_big:
            dataset += "_256"
        super(Kinetics_3d, self).__init__(
            mode,
            transform,
            seq_len,
            num_seq,
            downsample,
            which_split,
            vals_to_return,
            sampling_method,
            dataset=dataset
        )

        self.vid_shapes = {}

        if mu.FlowMode in self.vals_to_return:
            self.setup_flow_modality()

    def get_vid_shape(self, vpath, idx_block):
        v_class, v_name = self.get_class_vid(vpath)

        if (v_class, v_name) not in self.vid_shapes:
            img_list = [os.path.join(vpath, 'image_%05d.jpg' % (i + 1)) for i in idx_block[0:1]]
            seq = [pil_loader(f) for f in img_list]
            self.vid_shapes[(v_class, v_name)] = seq[0].size

        return self.vid_shapes[(v_class, v_name)]

    def fetch_flow_seq(self, vpath, seq_len, idx_block):
        '''
        Can be overriden in the derived classes
        '''

        shape = self.get_vid_shape(vpath, idx_block)

        def reshape_flow(img):
            new_img = img.resize((shape[0], shape[1]))
            assert new_img.size == shape, "Shape mismatch: {}, {}".format(new_img.shape, shape)
            return new_img

        def fill_nones(l):
            if l[0] is None:
                l[0] = torch.zeros((1, 128, 128))
            for i in range(1, len(l)):
                if l[i] is None:
                    l[i] = l[i-1]
            return l

        u_flow_list = self.get_u_flow_path_list(vpath, idx_block)
        v_flow_list = self.get_v_flow_path_list(vpath, idx_block)

        u_seq = fill_nones([flow_loader(f) for f in u_flow_list])
        v_seq = fill_nones([flow_loader(f) for f in v_flow_list])

        seq = [reshape_flow(toPILImage(torch.cat([u, v]))) for u, v in zip(u_seq, v_seq)]
        flow_t_seq = self.transform["flow"](seq)

        (FC, FH, FW) = flow_t_seq[0].size()
        flow_t_seq = torch.stack(flow_t_seq, 0)
        flow_t_seq = flow_t_seq.view(self.num_seq, seq_len, FC, FH, FW).transpose(1, 2)

        # Subract the mean to get interpretable optical flow
        if flow_t_seq.mean() > 0.3:
            flow_t_seq -= 0.5

        return flow_t_seq


class UCF101_3d(BaseDataloader):

    def __init__(
        self,
        mode='train',
        transform=None,
        seq_len=5,
        num_seq=6,
        downsample=3,
        which_split=1,
        vals_to_return=["imgs"],
        sampling_method="dynamic",
        debug=False,
    ):
        super(UCF101_3d, self).__init__(
            mode,
            transform,
            seq_len,
            num_seq,
            downsample,
            which_split,
            vals_to_return,
            sampling_method,
            dataset="ucf101",
            debug=debug
        )

        self.vid_shapes = {}
        self.fnb_flow_base_path = '/data/nishantr/ucf101/fnb_frames/'

    def get_fnb_u_flow_path_list(self, vpath, idx_block):
        v_class, v_name = self.get_class_vid(vpath)
        return [os.path.join(self.fnb_flow_base_path, v_class, v_name, 'flow_x_%05d.jpg' % (i + 1)) for i in idx_block]

    def get_fnb_v_flow_path_list(self, vpath, idx_block):
        v_class, v_name = self.get_class_vid(vpath)
        return [os.path.join(self.fnb_flow_base_path, v_class, v_name, 'flow_y_%05d.jpg' % (i + 1)) for i in idx_block]

    def get_vid_shape(self, vpath, idx_block):
        v_class, v_name = self.get_class_vid(vpath)

        if (v_class, v_name) not in self.vid_shapes:
            img_list = [os.path.join(vpath, 'image_%05d.jpg' % (i + 1)) for i in idx_block[0:1]]
            seq = [pil_loader(f) for f in img_list]
            self.vid_shapes[(v_class, v_name)] = seq[0].size

        return self.vid_shapes[(v_class, v_name)]

    def fetch_fnb_flow_seq(self, vpath, seq_len, idx_block):
        shape = self.get_vid_shape(vpath, idx_block)

        def reshape_flow(img):
            new_img = img.resize((shape[0], shape[1]))
            assert new_img.size == shape, "Shape mismatch: {}, {}".format(new_img.shape, shape)
            return new_img

        def fill_nones(l):
            if l[0] is None:
                l[0] = torch.zeros((1, 128, 128))
            for i in range(1, len(l)):
                if l[i] is None:
                    l[i] = l[i-1]
            return l

        u_flow_list = self.get_fnb_u_flow_path_list(vpath, idx_block)
        v_flow_list = self.get_fnb_v_flow_path_list(vpath, idx_block)

        u_seq = fill_nones([flow_loader(f) for f in u_flow_list])
        v_seq = fill_nones([flow_loader(f) for f in v_flow_list])

        seq = [reshape_flow(toPILImage(torch.cat([u, v]))) for u, v in zip(u_seq, v_seq)]
        flow_t_seq = self.transform["flow"](seq)

        (FC, FH, FW) = flow_t_seq[0].size()
        flow_t_seq = torch.stack(flow_t_seq, 0)
        flow_t_seq = flow_t_seq.view(self.num_seq, seq_len, FC, FH, FW).transpose(1, 2)

        # Subract the mean to get interpretable optical flow
        if flow_t_seq.mean() > 0.3:
            flow_t_seq -= 0.5

        return flow_t_seq


class BaseDataloaderHMDB(BaseDataloader):

    def __init__(
        self,
        mode,
        transform,
        seq_len,
        num_seq,
        downsample,
        which_split,
        vals_to_return,
        sampling_method,
        dataset
    ):
        super(BaseDataloaderHMDB, self).__init__(
            mode,
            transform,
            seq_len,
            num_seq,
            downsample,
            which_split,
            vals_to_return,
            sampling_method,
            dataset=dataset
        )


class HMDB51_3d(BaseDataloaderHMDB):
    def __init__(
        self,
        mode='train',
        transform=None,
        seq_len=5,
        num_seq=6,
        downsample=1,
        which_split=1,
        vals_to_return=["imgs"],
        sampling_method="dynamic"
    ):
        super(HMDB51_3d, self).__init__(
            mode,
            transform,
            seq_len,
            num_seq,
            downsample,
            which_split,
            vals_to_return,
            sampling_method,
            dataset="hmdb51"
        )


class JHMDB_3d(BaseDataloaderHMDB):
    def __init__(
        self,
        mode='train',
        transform=None,
        seq_len=5,
        num_seq=6,
        downsample=1,
        which_split=1,
        vals_to_return=["imgs"],
        sampling_method="dynamic"
    ):
        super(JHMDB_3d, self).__init__(
            mode,
            transform,
            seq_len,
            num_seq,
            downsample,
            which_split,
            vals_to_return,
            sampling_method,
            dataset="jhmdb"
        )


class Panasonic_3d(BaseDataloader):

    def __init__(
        self,
        mode='train',
        transform=None,
        seq_len=5,
        num_seq=6,
        downsample=3,
        which_split=1,
        vals_to_return=["imgs"],
        sampling_method="dynamic",
        debug=False,
        dataset="panasonic",
        postfix=''
    ):
        super(Panasonic_3d, self).__init__(
            mode,
            transform,
            seq_len,
            num_seq,
            downsample,
            which_split,
            vals_to_return,
            sampling_method,
            dataset=dataset,
            debug=debug,
            postfix=postfix
        )


    def __getitem__(self, index):
        row = np.array(self.video_info.iloc[index]).tolist()
        num_views = int(len([i for i in row if i == i]) / 4)
        # FIXME: randomly sample indices
        i0 = [i for i in range(num_views) if 'v000' in row[i * 4]][0]
        i1 = np.random.choice(np.setdiff1d(range(num_views), [i0]))
        vpath0, vstart0, vend0, vname, vpath1, vstart1, vend1, _ = row[4*i0 : 4*i0+4] + row[4*i1 : 4*i1+4]
        # FIXME: make sure the first frame is synchronized
        vstart = max(vstart0, vstart1)
        vend = min(vend0, vend1)
        vlen = int(vend - vstart + 1)

        # Remove trailing backslash if any
        vpath0 = vpath0.rstrip('/')
        vpath1 = vpath1.rstrip('/')

        seq_len = self.seq_len
        if "tgt" in self.vals_to_return:
            seq_len = 2 * self.seq_len

        items0, start_idx = self.idx_sampler(seq_len, self.num_seq, vlen, vpath0, idx_offset=vstart)
        items1, _ = self.idx_sampler(seq_len, self.num_seq, vlen, vpath1, idx_offset=vstart, start_idx=start_idx)
        if items0 is None or items1 is None:
            print(vpath)

        idx_block0, vpath0 = items0
        assert idx_block0.shape == (self.num_seq, seq_len)
        idx_block0 = idx_block0.reshape(self.num_seq * seq_len)

        idx_block1, vpath1 = items1
        assert idx_block1.shape == (self.num_seq, seq_len)
        idx_block1 = idx_block1.reshape(self.num_seq * seq_len)

        vals = {}

        # FIXME: make more general
        vals_to_return = np.unique([i.split('-')[0] for i in self.vals_to_return])
        # Populate return list
        if mu.ImgMode in vals_to_return:
            img_t_seq0 = self.fetch_imgs_seq(vpath0, seq_len, idx_block0)
            # 0 stands for the ego-view while 1 stands for the third-person view
            vals['{}-0'.format(mu.ImgMode)] = img_t_seq0
            img_t_seq1 = self.fetch_imgs_seq(vpath1, seq_len, idx_block1)
            vals['{}-1'.format(mu.ImgMode)] = img_t_seq1

        if mu.FlowMode in self.vals_to_return:
            flow_t_seq = self.fetch_flow_seq(vpath, seq_len, idx_block)
            vals[mu.FlowMode] = flow_t_seq
        if mu.FnbFlowMode in self.vals_to_return:
            fnb_flow_t_seq = self.fetch_fnb_flow_seq(vpath, seq_len, idx_block)
            vals[mu.FnbFlowMode] = fnb_flow_t_seq
        if mu.KeypointHeatmap in self.vals_to_return:
            hm_t_seq = self.fetch_kp_heatmap_seq(vpath, seq_len, idx_block)
            vals[mu.KeypointHeatmap] = hm_t_seq
        if mu.SegMask in self.vals_to_return:
            seg_t_seq = self.fetch_seg_mask_seq(vpath, seq_len, idx_block)
            vals[mu.SegMask] = seg_t_seq

        # Process double length target results
        if "tgt" in self.vals_to_return:
            orig_keys = list(vals.keys())
            for k in orig_keys:
                full_x = vals[k]
                vals[k] = full_x[:, :self.seq_len, ...]
                vals["tgt_" + k] = full_x[:, self.seq_len:, ...]
        if "labels" in self.vals_to_return:
            vid = self.encode_action(vname)
            label = torch.LongTensor([vid])
            vals["labels"] = label

        # Add video index field
        vals["vnames"] = torch.LongTensor([index])

        return vals