import argparse
from collections import namedtuple

import data_utils
import os


from torch.utils import data
from tensorboardX import SummaryWriter
from torchvision import transforms
from copy import deepcopy
from collections import defaultdict

from dataset_3d import *

sys.path.append('../utils')
from utils import AverageMeter

sys.path.append('../backbone')
from resnet_2d3d import neq_load_customized


# Constants for the framework
eps = 1e-7

CPCLoss = "cpc"
CooperativeLoss = "coop"

# Losses for mode sync
ModeSim = "sim"
CosSimLoss = "cossim"
CorrLoss = "corr"
DenseCosSimLoss = "dcssim"
DenseCorrLoss = "dcrr"

# Sets of different losses
LossList = [CPCLoss, CosSimLoss, CorrLoss, DenseCorrLoss, DenseCosSimLoss, CooperativeLoss]
ModeSyncLossList = [CosSimLoss, CorrLoss, DenseCorrLoss, DenseCosSimLoss]

ImgMode = "imgs"
FlowMode = "flow"
FnbFlowMode = "farne"
KeypointHeatmap = "kphm"
SegMask = "seg"
# FIXME: enable multiple views from the same modality
ModeList = [ImgMode, FlowMode, KeypointHeatmap, SegMask, FnbFlowMode, 'imgs-0', 'imgs-1', 'imgs-2', 'imgs-3', 'imgs-4']

ModeParams = namedtuple('ModeParams', ['mode', 'img_fet_dim', 'img_fet_segments', 'final_dim'])


def str2bool(s):
    """Convert string to bool (in argparse context)."""
    if s.lower() not in ['true', 'false']:
        raise ValueError('Need bool; got %r' % s)
    return {'true': True, 'false': False}[s.lower()]


def str2list(s):
    """Convert string to list of strs, split on _"""
    return s.split('_')


def get_multi_modal_model_train_args():
    parser = argparse.ArgumentParser()

    # General global training parameters
    parser.add_argument('--save_dir', default='', type=str, help='dir to save intermediate results')
    parser.add_argument('--dataset', default='ucf101', type=str)
    parser.add_argument('--ds', default=3, type=int, help='frame downsampling rate')
    parser.add_argument('--seq_len', default=5, type=int, help='number of frames in each video block')
    parser.add_argument('--num_seq', default=8, type=int, help='number of video blocks')
    parser.add_argument('--pred_step', default=3, type=int)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--epochs', default=10, type=int, help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, help='manual epoch number (useful on restarts)')
    parser.add_argument('--print_freq', default=5, type=int, help='frequency of printing output during training')
    parser.add_argument('--num_workers', default=2, type=int, help='Number of workers for dataloader')
    parser.add_argument('--reset_lr', action='store_true', help='Reset learning rate when resume training?')
    parser.add_argument('--notes', default="", type=str, help='Additional notes')
    parser.add_argument('--vis_log_freq', default=100, type=int, help='Visualization frequency')

    # Evaluation specific flags
    parser.add_argument('--ft_freq', default=10, type=int, help='frequency to perform finetuning')

    # Global network and model details. Can be overriden using specific flags
    parser.add_argument('--net', default='resnet18', type=str)
    parser.add_argument('--train_what', default='all', type=str)
    parser.add_argument('--img_dim', default=128, type=int)
    parser.add_argument('--sampling', default="dynamic", type=str, help='sampling method (disjoint, random, dynamic)')
    parser.add_argument('--l2_norm', default=True, type=str2bool, help='Whether to perform L2 normalization')
    parser.add_argument('--temp', default=0.07, type=float, help='Temperature to use with L2 normalization')

    # Training specific flags
    parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
    parser.add_argument('--wd', default=1e-5, type=float, help='weight decay')
    parser.add_argument('--losses', default="cpc", type=str2list, help='Losses to use (CPC, Align, Rep, Sim)')
    parser.add_argument('--dropout', default=0.3, type=float, help='Dropout to use for supervised training')
    parser.add_argument('--tune_bb', default=-1.0, type=float,
                        help='Fine-tune back-bone lr degradation. Useful for pretrained weights. (0.5, 0.1, 0.05)')
    parser.add_argument('--tune_imgs_bb', default=-1.0, type=float, help='Fine-tune imgs back-bone lr degradation.')

    # Hyper-parameters
    parser.add_argument('--msync_wt', default=10.0, type=float, help='Loss weight to use for mode sync loss')
    parser.add_argument('--dot_wt', default=10.0, type=float, help='Loss weight to use for mode sync loss')

    # Multi-modal related flags
    parser.add_argument('--data_sources', default='imgs', type=str2list, help='data sources separated by _')
    parser.add_argument('--modalities', default="imgs", type=str2list, help='Modalitiles to consider. Separate by _')

    # Checkpoint flags
    parser.add_argument('--imgs_restore_ckpt', default=None, type=str, help='Restore checkpoint for imgs')
    parser.add_argument('--flow_restore_ckpt', default=None, type=str, help='Restore checkpoint for flow')
    parser.add_argument('--farne_restore_ckpt', default=None, type=str, help='Restore checkpoint for farne flow')
    parser.add_argument('--kphm_restore_ckpt', default=None, type=str, help='Restore checkpoint for kp heatmap')
    parser.add_argument('--seg_restore_ckpt', default=None, type=str, help='Restore checkpoint for seg')

    # TODO: Flags to be fixed/revamped
    # Need to change restore for each ckpt

    # Flags which need not be touched
    parser.add_argument('--resume', default='', type=str, help='path of model to resume')
    parser.add_argument('--pretrain', default='', type=str, help='path of pretrained model')
    parser.add_argument('--prefix', default='22Mar', type=str, help='prefix of checkpoint filename')

    # Extra arguments
    parser.add_argument('--debug', default=False, type=str2bool, help='Reduces latency for data ops')

    return parser


def get_num_classes(dataset):
    if 'kinetics' in dataset:
        return 400
    elif dataset == 'ucf101':
        return 101
    elif dataset == 'jhmdb':
        return 21
    elif dataset == 'hmdb51':
        return 51
    elif dataset == 'panasonic':
        return 75
    else:
        return None


def get_transforms(args):
    return {
        ImgMode: get_imgs_transforms(args),
        FlowMode: get_flow_transforms(args),
        FnbFlowMode: get_flow_transforms(args),
        KeypointHeatmap: get_heatmap_transforms(args),
        SegMask: get_segmask_transforms(args),
    }


def get_test_transforms(args):
    return {
        ImgMode: get_imgs_test_transforms(args),
        FlowMode: get_flow_test_transforms(args),
        FnbFlowMode: get_flow_test_transforms(args),
        KeypointHeatmap: get_heatmap_test_transforms(args),
        SegMask: get_segmask_test_transforms(args),
    }


def convert_to_dict(args):
    if type(args) != dict:
        args_dict = vars(args)
    else:
        args_dict = args
    return args_dict


def get_imgs_test_transforms(args):
    args_dict = convert_to_dict(args)

    transform = transforms.Compose([
        CenterCrop(size=224, consistent=True),
        Scale(size=(args_dict["img_dim"], args_dict["img_dim"])),
        ToTensor(),
        Normalize()
    ])

    return transform


def get_flow_test_transforms(args):
    args_dict = convert_to_dict(args)
    dim = min(128, args_dict["img_dim"])

    center_crop_size = 224
    if args_dict["dataset"] == 'kinetics':
        center_crop_size = 128

    transform = transforms.Compose([
        CenterCrop(size=center_crop_size, consistent=True),
        Scale(size=(dim, dim)),
        ToTensor(),
    ])

    return transform


def get_heatmap_test_transforms(_):
    transform = transforms.Compose([
        CenterCropForTensors(size=192),
        ScaleForTensors(size=(64, 64)),
    ])
    return transform


def get_segmask_test_transforms(_):
    transform = transforms.Compose([
        CenterCropForTensors(size=192),
        ScaleForTensors(size=(64, 64)),
    ])
    return transform


def get_imgs_transforms(args):

    args_dict = convert_to_dict(args)
    transform = None

    if args_dict["debug"]:
        return transforms.Compose([
            CenterCrop(size=224, consistent=True),
            Scale(size=(args_dict["img_dim"], args_dict["img_dim"])),
            ToTensor(),
            Normalize()
        ])

    # designed for ucf101, short size=256, rand crop to 224x224 then scale to 128x128
    if args_dict["dataset"] == 'ucf101':
        transform = transforms.Compose([
            RandomHorizontalFlip(consistent=True),
            RandomCrop(size=224, consistent=True),
            Scale(size=(args_dict["img_dim"], args_dict["img_dim"])),
            RandomGray(consistent=False, p=0.5),
            ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.25, p=1.0),
            ToTensor(),
            Normalize()
        ])
    elif (args_dict["dataset"] == 'jhmdb') or (args_dict["dataset"] == 'hmdb51'):
        transform = transforms.Compose([
            RandomHorizontalFlip(consistent=True),
            RandomCrop(size=224, consistent=True),
            Scale(size=(args_dict["img_dim"], args_dict["img_dim"])),
            RandomGray(consistent=False, p=0.5),
            ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.25, p=1.0),
            ToTensor(),
            Normalize()
        ])
    # designed for kinetics400, short size=150, rand crop to 128x128
    elif args_dict["dataset"] == 'kinetics':
        transform = transforms.Compose([
            RandomSizedCrop(size=args_dict["img_dim"], consistent=True, p=1.0),
            RandomHorizontalFlip(consistent=True),
            RandomGray(consistent=False, p=0.5),
            ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.25, p=1.0),
            ToTensor(),
            Normalize()
        ])
    elif args_dict["dataset"] == 'panasonic':
        transform = transforms.Compose([
            RandomHorizontalFlip(consistent=True),
            PadToSize(size=(256, 256)),
            RandomCrop(size=224, consistent=True),
            Scale(size=(args_dict["img_dim"], args_dict["img_dim"])),
            RandomGray(consistent=False, p=0.5),
            ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.25, p=1.0),
            ToTensor(),
            Normalize()
        ])

    return transform


def get_flow_transforms(args):
    # TODO: Add random horizontal flip

    args_dict = convert_to_dict(args)
    dim = min(128, args_dict["img_dim"])
    transform = None

    if args_dict["debug"]:
        return transforms.Compose([
            Scale(size=(dim, dim)),
            ToTensor(),
        ])

    # designed for ucf101, short size=256, rand crop to 224x224 then scale to 128x128
    if (args_dict["dataset"] == 'ucf101') or (args_dict["dataset"] == 'jhmdb') or (args_dict["dataset"] == 'hmdb51'):
        transform = transforms.Compose([
            RandomIntensityCropForFlow(size=224),
            Scale(size=(dim, dim)),
            ToTensor(),
        ])
    # designed for kinetics400, short size=150, rand crop to 128x128
    elif args_dict["dataset"] == 'kinetics':
        transform = transforms.Compose([
            RandomIntensityCropForFlow(size=dim),
            ToTensor(),
        ])

    return transform


def get_heatmap_transforms(_):
    crop_size = int(192 * 0.8)
    transform = transforms.Compose([
        RandomIntensityCropForTensors(size=crop_size),
        ScaleForTensors(size=(64, 64)),
    ])
    return transform


def get_segmask_transforms(_):
    crop_size = int(192 * 0.8)
    transform = transforms.Compose([
        RandomIntensityCropForTensors(size=crop_size),
        ScaleForTensors(size=(64, 64)),
    ])
    return transform


def get_poses_transforms():
    return transforms.Compose([pu.RandomShift(), pu.Rescale()])


def get_writers(img_path):

    try: # old version
        writer_val = SummaryWriter(log_dir=os.path.join(img_path, 'val'))
        writer_train = SummaryWriter(log_dir=os.path.join(img_path, 'train'))
    except: # v1.7
        writer_val = SummaryWriter(logdir=os.path.join(img_path, 'val'))
        writer_train = SummaryWriter(logdir=os.path.join(img_path, 'train'))

    return writer_train, writer_val


def get_dataset_loaders(args, transform, mode='train'):
    print('Loading data for "%s" ...' % mode)

    if type(args) != dict:
        args_dict = deepcopy(vars(args))
    else:
        args_dict = args

    if args_dict['debug']:
        orig_mode = mode
        mode = 'train'

    use_big_K400 = False
    if args_dict["dataset"] == 'kinetics':
        use_big_K400 = args_dict["img_dim"] > 150
        dataset = Kinetics_3d(
            mode=mode,
            transform=transform,
            seq_len=args_dict["seq_len"],
            num_seq=args_dict["num_seq"],
            downsample=args_dict["ds"],
            vals_to_return=args_dict["data_sources"].split('_'),
            use_big=use_big_K400,
        )
    elif args_dict["dataset"] == 'ucf101':
        dataset = UCF101_3d(
                    mode=mode,
                    transform=transform,
                    seq_len=args_dict["seq_len"],
                    num_seq=args_dict["num_seq"],
                    downsample=args_dict["ds"],
                    vals_to_return=args_dict["data_sources"].split('_'),
                    debug=args_dict["debug"])
    elif args_dict["dataset"] == 'jhmdb':
        dataset = JHMDB_3d(mode=mode,
                            transform=transform,
                            seq_len=args_dict["seq_len"],
                            num_seq=args_dict["num_seq"],
                            downsample=args_dict["ds"],
                            vals_to_return=args_dict["data_sources"].split('_'),
                            sampling_method=args_dict["sampling"])
    elif args_dict["dataset"] == 'hmdb51':
        dataset = HMDB51_3d(mode=mode,
                           transform=transform,
                           seq_len=args_dict["seq_len"],
                           num_seq=args_dict["num_seq"],
                           downsample=args_dict["ds"],
                           vals_to_return=args_dict["data_sources"].split('_'),
                           sampling_method=args_dict["sampling"])
    elif args_dict["dataset"] == 'panasonic':
        dataset = Panasonic_3d(
                    mode=mode,
                    transform=transform,
                    seq_len=args_dict["seq_len"],
                    num_seq=args_dict["num_seq"],
                    downsample=args_dict["ds"],
                    vals_to_return=args_dict["data_sources"].split('_'),
                    debug=args_dict["debug"])
    else:
        raise ValueError('dataset not supported')

    val_sampler = data.SequentialSampler(dataset)
    if use_big_K400:
        train_sampler = data.RandomSampler(dataset, replacement=True, num_samples=int(0.2 * len(dataset)))
    else:
        train_sampler = data.RandomSampler(dataset)

    if args_dict["debug"]:
        if orig_mode == 'val':
            train_sampler = data.RandomSampler(dataset, replacement=True, num_samples=200)
        else:
            train_sampler = data.RandomSampler(dataset, replacement=True, num_samples=2000)
        val_sampler = data.RandomSampler(dataset)
        # train_sampler = data.RandomSampler(dataset, replacement=True, num_samples=100)

    data_loader = None
    if mode == 'train':
        data_loader = data.DataLoader(dataset,
                                      batch_size=args_dict["batch_size"],
                                      sampler=train_sampler,
                                      shuffle=False,
                                      num_workers=args_dict["num_workers"],
                                      collate_fn=data_utils.individual_collate,
                                      pin_memory=True,
                                      drop_last=True)
    elif mode == 'val':
        data_loader = data.DataLoader(dataset,
                                      sampler=val_sampler,
                                      batch_size=args_dict["batch_size"],
                                      shuffle=False,
                                      num_workers=args_dict["num_workers"],
                                      collate_fn=data_utils.individual_collate,
                                      pin_memory=True,
                                      drop_last=True)
    elif mode == 'test':
        data_loader = data.DataLoader(dataset,
                                      sampler=val_sampler,
                                      batch_size=args_dict["batch_size"],
                                      shuffle=False,
                                      num_workers=args_dict["num_workers"],
                                      collate_fn=data_utils.individual_collate,
                                      pin_memory=True,
                                      drop_last=False)

    print('"%s" dataset size: %d' % (mode, len(dataset)))
    return data_loader


def set_multi_modal_path(args):
    if args.resume:
        exp_path = os.path.dirname(os.path.dirname(args.resume))
    else:
        args.modes_str = '_'.join(args.modes)
        args.l2norm_str = str(args.l2_norm)
        exp_path = 'logs/{args.prefix}/{args.dataset}-{args.img_dim}_{0}_' \
                   'bs{args.batch_size}_seq{args.num_seq}_pred{args.pred_step}_len{args.seq_len}_ds{args.ds}_' \
                   'train-{args.train_what}{1}_modes-{args.modes_str}_l2norm' \
                   '_{args.l2norm_str}_{args.notes}'.format(
                        'r%s' % args.net[6::],
                        '_pt=%s' % args.pretrain.replace('/','-') if args.pretrain else '',
                        args=args
                    )
        exp_path = os.path.join(args.save_dir, exp_path)

    img_path = os.path.join(exp_path, 'img')
    model_path = os.path.join(exp_path, 'model')
    if not os.path.exists(img_path): os.makedirs(img_path)
    if not os.path.exists(model_path): os.makedirs(model_path)
    return img_path, model_path


def process_output(mask):
    '''task mask as input, compute the target for contrastive loss'''
    # dot product is computed in parallel gpus, so get less easy neg, bounded by batch size in each gpu'''
    # mask meaning: -2: omit, -1: temporal neg (hard), 0: easy neg, 1: pos, -3: spatial neg
    (B, NP, SQ, B2, NS, _) = mask.size() # [B, P, SQ, B, N, SQ]
    target = mask == 1
    target.requires_grad = False
    return target, (B, B2, NS, NP, SQ)


def check_name_to_be_avoided(k):
    # modules_to_avoid = ['.agg.', '.network_pred.']
    modules_to_avoid = []
    for m in modules_to_avoid:
        if m in k:
            return True
    return False


def load_model(model, model_path):
    if os.path.isfile(model_path):
        print("=> loading resumed checkpoint '{}'".format(model_path))
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
        model = neq_load_customized(model, checkpoint['state_dict'])
        print("=> loaded resumed checkpoint '{}' (epoch {})".format(model_path, checkpoint['epoch']))
    else:
        print("[WARNING] no checkpoint found at '{}'".format(model_path))
    return model


def get_stats_dict(losses_dict, stats, eval=False):
    postfix_dict = {}

    # Populate accuracies
    for loss in stats.keys():
        for mode in stats[loss].keys():
            for stat, meter in stats[loss][mode].items():
                val = meter.avg if eval else meter.local_avg
                postfix_dict[loss[:3] + '_' + mode[:3] + "_" + str(stat)] = round(val, 3)

    # Populate losses
    for loss in losses_dict.keys():
        for key, meter in losses_dict[loss].items():
            key_str = "l_{}_{}".format(loss, key[:3])
            val = meter.avg if eval else meter.local_avg
            postfix_dict[key_str] = round(val, 3)

    return postfix_dict


def init_loggers(losses):
    losses_dict = {l: defaultdict(lambda: AverageMeter()) for l in losses}

    stats = {}
    for loss in losses:
        # Creates a nested default dict
        stats[loss] = defaultdict(lambda: defaultdict(lambda: AverageMeter()))

    return losses_dict, stats
