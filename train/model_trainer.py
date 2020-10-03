import sys
import time
import os
import torch
import random

import numpy as np
import torch.nn as nn
import torchvision.utils as vutils
import torch.optim as optim
import finetune_utils as ftu
import model_utils as mu
import model_3d as m3d
import sim_utils as su
import mask_utils as masku

sys.path.append('../backbone')
from model_3d import DpcRnn
from tqdm import tqdm
from copy import deepcopy

sys.path.append('../utils')
from utils import AverageMeter, save_checkpoint, denorm, calc_topk_accuracy


def get_modality_list(modalities):
    modes = []
    for m in mu.ModeList:
        if m in modalities:
            modes.append(m)
    return modes


def get_modality_restore_ckpts(args):
    ckpts = {}
    for m in mu.ModeList:
        ckpt = getattr(args, m.split('-')[0] + "_restore_ckpt")
        ckpts[m] = ckpt
        if ckpt is not None:
            print("Mode: {} is being restored from: {}".format(m, ckpt))
    return ckpts


class ModalitySyncer(nn.Module):

    def get_feature_extractor_based_on_mode(self, mode_params):
        if mode_params.mode.split('-')[0] in [mu.ImgMode, mu.FlowMode, mu.KeypointHeatmap, mu.SegMask]:
            return m3d.ImageFetCombiner(mode_params.img_fet_dim, mode_params.img_fet_segments)
        else:
            assert False, "Invalid mode provided: {}".format(mode_params)

    def __init__(self, args):
        super(ModalitySyncer, self).__init__()

        self.losses = args["losses"]

        self.mode0_dim = args["mode_0_params"].final_dim
        self.mode1_dim = args["mode_1_params"].final_dim
        self.mode0_fet_extractor = self.get_feature_extractor_based_on_mode(args["mode_0_params"])
        self.mode1_fet_extractor = self.get_feature_extractor_based_on_mode(args["mode_1_params"])

        self.instance_mask = args["instance_mask"]

        self.common_dim = min(self.mode0_dim, self.mode1_dim) // 2
        # input is B x dim0, B x dim1
        self.mode1_to_common = nn.Sequential()
        self.mode0_to_common = nn.Sequential()

        self.mode_losses = [mu.DenseCosSimLoss]

        self.simHandler = nn.ModuleDict(
            {
                mu.CosSimLoss: su.CosSimHandler(),
                mu.CorrLoss: su.CorrSimHandler(),
                mu.DenseCorrLoss: su.DenseCorrSimHandler(self.instance_mask),
                mu.DenseCosSimLoss: su.DenseCosSimHandler(self.instance_mask),
            }
        )

        # Perform initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def get_total_loss(self, mode0, mode1):
        loss = torch.tensor(0.0).to(mode0.device)

        stats = {}
        for lossKey in self.mode_losses:
            lossVal0, stat = self.simHandler[lossKey](mode0, mode1)
            lossVal1, _ = self.simHandler[lossKey](mode1, mode0)
            lossVal = (lossVal0 + lossVal1) * 0.5
            loss += lossVal
            stats[lossKey] = stat

        return loss, stats

    def forward(self, input0, input1):
        assert len(input0.shape) == 5, "{}".format(input0.shape)
        # inputs are B, N, D, S, S
        y0 = self.mode0_fet_extractor(input0)
        y1 = self.mode1_fet_extractor(input1)

        # outputs are B * N, dim
        B, N, _ = y1.shape
        y0_in_space_common = self.mode0_to_common(y0.view(B * N, -1)).view(B, N, -1)
        y1_in_space_common = self.mode1_to_common(y1.view(B * N, -1)).view(B, N, -1)

        return self.get_total_loss(y0_in_space_common, y1_in_space_common)


class MultiModalModelTrainer(nn.Module):

    def addImgGrid(self, data):
        '''
        Plots image frames in different subsections
        '''
        # shape of data[mode] batch, num_seq, seq_len, IC, IH, IW
        for mode in self.modes:
            IC = data[mode].shape[3]
            if IC == 3:
                images = data[mode].cpu()[:, 0, 0, ...]
            else:
                # Plot the summation instead
                images = data[mode].cpu()[:, 0, 0, :, :, :].mean(dim=1, keepdim=True)
            # Shape is batch, IC, IH, IW
            grid = vutils.make_grid(images, nrow=int(np.sqrt(images.shape[0])))
            self.writer_train.add_image('images/frames/{}'.format(mode), grid, 0)

    def addDotLossDistribution(self, cdots_mode_dict, cdots_sets_mode_dict, iter):
        lim = 1024

        shouldFlush = False
        for mode in self.modes:
            cdots = cdots_mode_dict[mode].detach()
            feature_sets = cdots_sets_mode_dict[mode]

            if feature_sets is None:
                cdots_sets_mode_dict[mode] = cdots
            elif feature_sets.shape[0] < lim:
                B_ = feature_sets.shape[2]
                cdots = cdots[:, :, :B_, :]
                cdots_sets_mode_dict[mode] = torch.cat([feature_sets, cdots], dim=0)
            else:
                shouldFlush = True

        if shouldFlush:
            for mode in self.modes:
                self.writer_train.add_histogram(
                    'cdots/individual/{}'.format(mode), cdots_sets_mode_dict[mode].detach().cpu(), iter)

            for m0, m1 in self.mode_pairs:
                diff = (cdots_sets_mode_dict[m0] - cdots_sets_mode_dict[m1]).abs()
                self.writer_train.add_histogram(
                    'cdots/diff/{}'.format(self.get_tuple_name(m0, m1)), diff.detach().cpu(), iter)

            # Reset the feature set
            for mode in self.modes:
                cdots_sets_mode_dict[mode] = None

        return cdots_sets_mode_dict

    def addModesCossimDistribution(self, features_mode_dict, labels, feature_sets_mode_dict, iter, tag):
        '''
        Updates and plots cossim scores of features with different combinations
        features: Batch x Dim
        Plot,
            - Positive scores (In the same class)
            - Negative scores (Of other classes)
        '''

        lim = 1024
        if feature_sets_mode_dict["labels"] is None:
            feature_sets_mode_dict["labels"] = labels
        elif feature_sets_mode_dict["labels"].shape[0] < lim:
            feature_sets_mode_dict["labels"] = torch.cat([feature_sets_mode_dict["labels"], labels], dim=0)

        shouldFlush = False
        for mode in self.modes:
            features = features_mode_dict[mode].detach()
            feature_sets = feature_sets_mode_dict[mode]

            if tag == 'gtf':
                features = features.mean(-1).mean(-1)

            if feature_sets is None:
                feature_sets_mode_dict[mode] = features
            elif feature_sets.shape[0] < lim:
                feature_sets_mode_dict[mode] = torch.cat([feature_sets, features], dim=0)
            else:
                shouldFlush = True

        if shouldFlush:
            self.addModesAggHistogram(feature_sets_mode_dict, iter, tag)
            self.addInterModesAggHistogram(feature_sets_mode_dict, iter, tag)

            # Reset the feature set
            for mode in self.modes:
                feature_sets_mode_dict[mode] = None
            feature_sets_mode_dict["labels"] = None

        return feature_sets_mode_dict

    def addModesAggHistogram(self, feature_sets_mode_dict, iter, tag):

        tot_labels = feature_sets_mode_dict["labels"]
        for mode in self.modes:
            feature_sets = feature_sets_mode_dict[mode]

            # Generate class ref features
            class_wise_idx = {cls_id: [] for cls_id in range(self.num_classes)}
            for idx in range(tot_labels.shape[0]):
                cls_id = tot_labels[idx].item()
                class_wise_idx[cls_id].append(idx)

            # Generate positive and overall scores
            pos_scores = []
            for cls_id in range(self.num_classes):
                class_fet = feature_sets[class_wise_idx[cls_id]]
                pos_score = self.cosSimHandler.get_feature_cross_pair_score(
                    self.cosSimHandler.l2NormedVec(class_fet), self.cosSimHandler.l2NormedVec(class_fet)
                )
                pos_scores.append(pos_score.flatten())
            pos_scores = torch.cat(pos_scores).flatten()

            ovr_scores = self.cosSimHandler.get_feature_cross_pair_score(
                self.cosSimHandler.l2NormedVec(feature_sets), self.cosSimHandler.l2NormedVec(feature_sets)
            )

            self.writer_train.add_histogram('cossim/{}/pos/{}'.format(tag, mode), pos_scores.detach().cpu(), iter)
            self.writer_train.add_histogram('cossim/{}/ovr/{}'.format(tag, mode), ovr_scores.detach().cpu(), iter)

    def addInterModesAggHistogram(self, feature_sets_mode_dict, iter, tag):

        tot_labels = feature_sets_mode_dict["labels"]
        for m0, m1 in self.mode_pairs:
            feature_sets0 = feature_sets_mode_dict[m0]
            feature_sets1 = feature_sets_mode_dict[m1]

            # Generate class ref features
            class_wise_idx = {cls_id: [] for cls_id in range(self.num_classes)}
            for idx in range(tot_labels.shape[0]):
                cls_id = tot_labels[idx].item()
                class_wise_idx[cls_id].append(idx)

            # Generate positive and overall scores
            direct_pos_scores = []
            pos_scores = []
            for cls_id in range(self.num_classes):
                class_fet0 = feature_sets0[class_wise_idx[cls_id]]
                class_fet1 = feature_sets1[class_wise_idx[cls_id]]
                pos_score = self.cosSimHandler.get_feature_cross_pair_score(
                    self.cosSimHandler.l2NormedVec(class_fet0), self.cosSimHandler.l2NormedVec(class_fet1)
                )
                pos_scores.append(pos_score.flatten())
                direct_pos_score = self.cosSimHandler.get_feature_pair_score(
                    self.cosSimHandler.l2NormedVec(class_fet0), self.cosSimHandler.l2NormedVec(class_fet1)
                )
                direct_pos_scores.append(direct_pos_score.flatten())
            pos_scores = torch.cat(pos_scores).flatten()
            direct_pos_scores = torch.cat(direct_pos_scores).flatten()

            ovr_scores = self.cosSimHandler.get_feature_cross_pair_score(
                self.cosSimHandler.l2NormedVec(feature_sets0), self.cosSimHandler.l2NormedVec(feature_sets1)
            )

            self.writer_train.add_histogram(
                'cossim/{}/pos/{}'.format(tag, self.get_tuple_name(m0, m1)), pos_scores.detach().cpu(), iter)
            self.writer_train.add_histogram(
                'cossim/{}/dir-pos/{}'.format(tag, self.get_tuple_name(m0, m1)), direct_pos_scores.detach().cpu(), iter)
            self.writer_train.add_histogram(
                'cossim/{}/ovr/{}'.format(tag, self.get_tuple_name(m0, m1)), ovr_scores.detach().cpu(), iter)

    def get_modality_feature_extractor(self, final_feature_size, last_size, mode):
        if mode.split('-')[0] in [mu.ImgMode, mu.FlowMode, mu.KeypointHeatmap, mu.SegMask]:
            return m3d.ImageFetCombiner(final_feature_size, last_size)
        else:
            assert False, "Invalid mode provided: {}".format(mode)

    def __init__(self, args):
        super(MultiModalModelTrainer, self).__init__()

        self.args = args

        self.modes = args["modalities"]
        self.models = nn.ModuleDict(args["models"])
        self.num_classes = args["num_classes"]
        self.data_sources = args["data_sources"]
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # Gradient accumulation step interval
        self.grad_step_interval = 4

        self.vis_log_freq = args["vis_log_freq"]

        self.losses = [mu.CPCLoss, mu.DenseCosSimLoss, mu.CooperativeLoss]

        # Model log writers
        self.img_path = args["img_path"]
        self.model_path = args["model_path"]
        self.writer_train, self.writer_val = mu.get_writers(self.img_path)

        print("Model path is:", self.model_path, self.img_path)

        self.shouldFinetune = args["ft_freq"] > 0

        transform = mu.get_transforms(args)
        self.train_loader = mu.get_dataset_loaders(args, transform, 'train')
        self.val_loader = mu.get_dataset_loaders(args, transform, 'val')
        self.num_classes = args["num_classes"]
        self.l2_norm = args["l2_norm"]
        self.temp = args["temp"] if self.l2_norm else 1.0

        self.common_dim = 128
        self.cpc_projections = nn.ModuleDict({m: nn.Sequential() for m in self.modes})

        self.denormalize = denorm()

        self.val_criterion_base = nn.CrossEntropyLoss()
        self.val_criterion = lambda x, y: self.val_criterion_base(x, y.float().argmax(dim=1))

        self.criterias = {
            mu.CPCLoss: self.val_criterion,
            mu.CooperativeLoss: nn.L1Loss(),
        }

        self.CooperativeLossLabel = mu.CooperativeLoss

        self.compiled_features = {m: self.get_modality_feature_extractor(
            self.models[m].final_feature_size, self.models[m].last_size, m) for m in self.modes}
        self.interModeDotHandler = su.InterModeDotHandler(last_size=None)

        self.B0 = self.B1 = self.args["batch_size"]

        self.standard_grid_mask = {
            m: masku.process_mask(
                masku.get_standard_grid_mask(
                    self.B0,
                    self.B1,
                    self.args["pred_step"],
                    self.models[m].last_size,
                    device=self.device
                )
            ) for m in self.modes
        }
        self.standard_instance_mask = masku.process_mask(
            masku.get_standard_instance_mask(self.B0, self.B1, self.args["pred_step"], device=self.device)
        )

        self.modeSyncers = nn.ModuleDict()

        self.mode_pairs = [(m0, m1) for m0 in self.modes for m1 in self.modes if m0 < m1]

        self.losses.append(mu.ModeSim)
        self.sync_wt = self.args["msync_wt"]

        for (m0, m1) in self.mode_pairs:
            num_seq = self.args["num_seq"]
            instance_mask_m0_m1 = masku.process_mask(
                masku.get_standard_instance_mask(self.B0, self.B1, num_seq, self.device)
            )
            mode_align_args = {
                "losses": self.losses,
                "mode_0_params": self.get_mode_params(m0),
                "mode_1_params": self.get_mode_params(m1),
                "dim_layer_1": 64,
                "instance_mask": instance_mask_m0_m1,
            }
            # Have to explicitly send these to the GPU as they're present in a dict
            self.modeSyncers[self.get_tuple_name(m0, m1)] = ModalitySyncer(mode_align_args).to(self.device)

        print("[NOTE] Losses used: ", self.losses)

        self.cosSimHandler = su.CosSimHandler()
        print("Using CosSim Dot Losses!")
        self.dot_wt = args["dot_wt"]

        # Use a smaller learning rate if the backbone is already trained
        degradeBackboneLR = 1.0
        if args["tune_bb"] > 0:
            degradeBackboneLR = args["tune_bb"]
        backboneLr = {
            'params':
                [p for k in self.models.keys() for p in self.models[k].parameters()],
            'lr': args["lr"] * degradeBackboneLR
        }
        imgsBackboneLr = {'params': [], 'lr': args["lr"]}
        if args["tune_imgs_bb"] > 0:
            imgsBackboneLr = {
                'params': [p for p in self.models[mu.ImgMode].parameters()],
                'lr': args["lr"] * args["tune_imgs_bb"]
            }
            backboneLr = {
                'params':
                    [p for k in (set(self.models.keys()).difference([mu.ImgMode])) for p in self.models[k].parameters()],
                'lr': args["lr"] * degradeBackboneLR
            }
        miscLr = {
            'params': [p for model in list(self.modeSyncers.values()) for p in model.parameters()],
            'lr': args["lr"]
        }

        self.optimizer = optim.Adam(
            [miscLr, backboneLr, imgsBackboneLr],
            lr=args["lr"],
            weight_decay=args["wd"]
        )

        patience = 10
        self.lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, verbose=True, patience=patience, min_lr=1e-5
        )

        self.model_finetuner = ftu.QuickSupervisedModelTrainer(self.num_classes, self.modes)

        self.iteration = 0
        self.accuracyKList = [1, 3]

    @staticmethod
    def get_tuple_name(m0, m1):
        return "{}|{}".format(m0[:1], m1[:1])

    def get_mode_params(self, mode):
        if mode.split('-')[0] in [mu.ImgMode, mu.FlowMode, mu.KeypointHeatmap, mu.SegMask]:
            return mu.ModeParams(
                mode,
                self.models[mode].param['feature_size'],
                self.models[mode].last_size,
                self.models[mode].param['feature_size']
            )
        else:
            assert False, "Incorrect mode: {}".format(mode)

    def get_feature_pair_score(self, pred_features, gt_features):
        """
            (pred/gt)features: [B, N, D, S, S]
            Special case for a few instances would be with S=1
            Returns 6D pair score tensor
        """
        B1, N1, D1, S1, S1 = pred_features.shape
        B2, N2, D2, S2, S2 = gt_features.shape
        assert (D1, S1) == (D2, S2), \
            "Mismatch between pred and gt features: {}, {}".format(pred_features.shape, gt_features.shape)

        # dot product D dimension in pred-GT pair, get a 6d tensor. First 3 dims are from pred, last 3 dims are from GT.
        preds = pred_features.permute(0, 1, 3, 4, 2).contiguous().view(B1 * N1 * S1 * S1, D1) / self.temp
        gts = gt_features.permute(0, 1, 3, 4, 2).contiguous().view(B2 * N2 * S2 * S2, D2).transpose(0, 1) / self.temp

        # Get the corresponding scores of each region in the matrix with each other region i.e.
        # total last_size ** 4 combinations. Note that we have pred_step ** 2 such pairs as well
        score = torch.matmul(preds, gts).view(B1, N1, S1*S1, B2, N2, S2*S2)

        return score

    def log_visuals(self, input_seq, mode):
        if input_seq.size(0) > 5:
            input_seq = input_seq[:5]

        ic = 3
        if not mode.startswith("imgs"):
            assert input_seq.shape[2] in [1, 2, 3, 17], "Invalid shape: {}".format(input_seq.shape)
            input_seq = torch.abs(input_seq)
            input_seq = input_seq.sum(dim=2, keepdim=True)
            input_seq = input_seq / 0.25
            input_seq[input_seq > 1] = 1.0
            input_seq[input_seq != input_seq] = 0.0
            ic = 1

        img_dim = input_seq.shape[-1]
        assert img_dim in [64, 128, 224], "imgs_dim: {}".format(img_dim)
        grid_img = vutils.make_grid(
            input_seq.transpose(2, 3).contiguous().view(-1, ic, img_dim, img_dim),
            nrow=self.args["num_seq"] * self.args["seq_len"]
        )

        if mode.startswith("imgs"):
            denormed_img = self.denormalize(grid_img)
        else:
            denormed_img = grid_img

        self.writer_train.add_image('input_seq/{}'.format(mode), denormed_img, self.iteration)

    def log_metrics(self, losses_dict, stats, writer, prefix):
        for loss in self.losses:
            for mode in losses_dict[loss].keys():
                writer.add_scalar(
                    prefix + '/losses/' + loss + '/' + str(mode),
                    losses_dict[loss][mode].val,
                    self.iteration
                )
        for loss in stats.keys():
            for mode in stats[loss].keys():
                for stat in stats[loss][mode].keys():
                    writer.add_scalar(
                        prefix + '/stats/' + loss + '/' + str(mode) + '/' + str(stat),
                        stats[loss][mode][stat].val,
                        self.iteration
                    )

    def perform_forward_passes(self, data, Xs):

        B, NS = self.args["batch_size"], self.args["pred_step"]

        pred_features, gt_all_features, agg_features = {}, {}, {}
        flat_scores = {}
        for mode in self.modes:
            if not mode in data.keys():
                continue

            input_seq = data[mode].to(self.device)
            assert input_seq.shape[0] == self.args["batch_size"]
            SQ = self.models[mode].last_size ** 2

            pred_features[mode], gt_features, gt_all_features[mode], X = \
                self.models[mode](input_seq, ret_rep=True)

            gt_all_features[mode] = self.cpc_projections[mode](gt_all_features[mode])
            pred_features[mode] = self.cpc_projections[mode](pred_features[mode])

            # score is a 6d tensor: [B, P, SQ, B', N', SQ]
            score_ = self.get_feature_pair_score(pred_features[mode], gt_features)
            flat_scores[mode] = score_.view(B * NS * SQ, -1)

            if self.shouldFinetune:
                Xs[mode].append(X.reshape(X.shape[0], -1).detach().cpu())

            del input_seq

        return pred_features, gt_all_features, agg_features, flat_scores, Xs

    def update_metrics(self, gt_all_features, flat_scores, stats, data):

        B, NS, N = self.args["batch_size"], self.args["pred_step"], self.args["num_seq"]
        loss_dict = {k: {} for k in self.losses}

        contexts = {}

        for mode in flat_scores.keys():
            SQ = self.models[mode].last_size ** 2

            target_flattened = self.standard_grid_mask[mode].view(self.B0 * NS * SQ, self.B1 * NS * SQ)

            # CPC loss
            if True:

                score_flat = flat_scores[mode]
                target = target_flattened

                target_lbl = target.float().argmax(dim=1)

                # Compute and log performance metrics
                topKs = calc_topk_accuracy(score_flat, target_lbl, self.accuracyKList)
                for i in range(len(self.accuracyKList)):
                    stats[mu.CPCLoss][mode]["acc" + str(self.accuracyKList[i])].update(topKs[i].item(), B)

                # Compute CPC loss for independent model training
                loss_dict[mu.CPCLoss][mode] = self.criterias[mu.CPCLoss](score_flat, target)

        for (m0, m1) in self.mode_pairs:
            if (m0 not in flat_scores.keys()) or (m1 not in flat_scores.keys()):
                continue

            tupName = self.get_tuple_name(m0, m1)

            # Cdot related losses
            if True:

                comp_gt_all0 = self.compiled_features[m0](gt_all_features[m0]).unsqueeze(3).unsqueeze(3)
                comp_gt_all1 = self.compiled_features[m1](gt_all_features[m1]).unsqueeze(3).unsqueeze(3)
                cdot0 = self.interModeDotHandler.get_cluster_dots(comp_gt_all0)
                cdot1 = self.interModeDotHandler.get_cluster_dots(comp_gt_all1)

                B, NS, B2, NS = cdot0.shape

                assert cdot0.shape == cdot1.shape == (B, NS, B2, NS), \
                    "Invalid shapes: {}, {}, {}".format(cdot0.shape, cdot1.shape, (B, NS, B2, NS))

                cos_sim_dot_loss = self.criterias[self.CooperativeLossLabel](cdot0, cdot1)
                loss_dict[self.CooperativeLossLabel][tupName] = self.dot_wt * cos_sim_dot_loss

            # Modality sync loss
            if True:

                sync_loss, mode_stats = self.modeSyncers[tupName](gt_all_features[m0], gt_all_features[m1])

                # stats: dict modeLoss -> specificStat
                for modeLoss in mode_stats.keys():
                    for stat in mode_stats[modeLoss].keys():
                        stats[modeLoss][tupName][stat].update(mode_stats[modeLoss][stat].item(), B)

                loss_dict[mu.ModeSim][tupName] = self.sync_wt * sync_loss

        return loss_dict, stats

    def pretty_print_stats(self, stats):
        grouped_stats = {}
        for k, v in stats.items():
            middle = k.split('_')[-1]
            if middle not in grouped_stats:
                grouped_stats[middle] = []
            grouped_stats[middle].append((k, v))
        for v in grouped_stats.values():
            print(sorted(v))

    def train_epoch(self, epoch):

        self.train()
        print("Model path is:", self.model_path)

        for mode in self.models.keys():
            self.models[mode].train()
        for mode in self.modeSyncers.keys():
            self.modeSyncers[mode].train()

        losses_dict, stats = mu.init_loggers(self.losses)

        B = self.args["batch_size"]
        trainX, trainY = {m: [] for m in self.modes}, []

        tq = tqdm(self.train_loader, desc="Train progress: Ep {}".format(epoch), position=0)
        self.optimizer.zero_grad()

        for idx, data in enumerate(tq):
            trainY.append(data["labels"])

            _, gt_all_features, agg_features, flat_scores, trainX = \
                self.perform_forward_passes(data, trainX)

            loss_dict, stats = \
                self.update_metrics(gt_all_features, flat_scores, stats, data)

            loss = torch.tensor(0.0).to(self.device)
            for l in loss_dict.keys():
                for v in loss_dict[l].keys():
                    losses_dict[l][v].update(loss_dict[l][v].item(), B)
                    loss += loss_dict[l][v]

            loss.backward()

            if idx % self.grad_step_interval:
                self.optimizer.step()
                self.optimizer.zero_grad()

            tq_stats = mu.get_stats_dict(losses_dict, stats)
            tq.set_postfix(tq_stats)

            del loss

            # Perform logging
            if self.iteration % self.vis_log_freq == 0:
                for mode in self.modes:
                    if mode in data.keys():
                        self.log_visuals(data[mode], mode)

            if idx % self.args["print_freq"] == 0:
                self.log_metrics(losses_dict, stats, self.writer_train, prefix='local')

            self.iteration += 1

        print("Overall train stats:")
        self.pretty_print_stats(tq_stats)

        if self.shouldFinetune:
            trainX = {k: torch.cat(v) for k, v in trainX.items()}
            trainY = torch.cat(trainY).reshape(-1).detach()

        return losses_dict, stats, {"X": trainX, "Y": trainY}

    def validate_epoch(self, epoch):

        self.eval()

        for mode in self.models.keys():
            self.models[mode].eval()
        for mode in self.modeSyncers.keys():
            self.modeSyncers[mode].eval()

        losses_dict, stats = mu.init_loggers(self.losses)
        overall_loss = AverageMeter()

        B = self.args["batch_size"]
        tq_stats = {}
        valX, valY = {m: [] for m in self.modes}, []

        with torch.no_grad():
            tq = tqdm(self.val_loader, desc="Val progress: Ep {}".format(epoch), position=0)

            for idx, data in enumerate(tq):
                valY.append(data["labels"])

                _, gt_all_features, agg_features, flat_scores, valX = self.perform_forward_passes(data, valX)
                loss_dict, stats = self.update_metrics(gt_all_features, flat_scores, stats, data)

                # Perform logging
                if self.iteration % self.vis_log_freq == 0:
                    for mode in self.modes:
                        self.log_visuals(data[mode], mode)

                loss = torch.tensor(0.0).to(self.device)
                for l in loss_dict.keys():
                    for v in loss_dict[l].keys():
                        losses_dict[l][v].update(loss_dict[l][v].item(), B)
                        loss += loss_dict[l][v]

                overall_loss.update(loss.item(), B)

                tq_stats = mu.get_stats_dict(losses_dict, stats)
                tq.set_postfix(tq_stats)

        print("Overall val stats:")
        self.pretty_print_stats(tq_stats)

        if self.shouldFinetune:
            valX = {k: torch.cat(v) for k, v in valX.items()}
            valY = torch.cat(valY).reshape(-1)

        return overall_loss, losses_dict, stats, {"X": valX, "Y": valY}

    def train_module(self):

        best_acc = {m: 0.0 for m in self.modes}

        for epoch in range(self.args["start_epoch"], self.args["epochs"]):
            train_losses, train_stats, trainD = self.train_epoch(epoch)
            ovr_loss, val_losses, val_stats, valD = self.validate_epoch(epoch)
            self.lr_scheduler.step(ovr_loss.avg)

            # Log fine-tune performance
            if self.shouldFinetune:
                if epoch % self.args["ft_freq"] == 0:
                    self.model_finetuner.evaluate_classification(trainD, valD)
                    if not self.args["debug"]:
                        self.model_finetuner.evaluate_clustering(trainD, tag='train')
                        # self.model_finetuner.evaluate_clustering(valD, tag='val')

            # save curve
            self.log_metrics(train_losses, train_stats, self.writer_train, prefix='global')
            self.log_metrics(val_losses, val_stats, self.writer_val, prefix='global')

            # save check_point for each mode individually
            for modality in self.modes:
                loss = mu.CPCLoss

                is_best = val_stats[loss][modality][1].avg > best_acc[modality]
                best_acc[modality] = max(val_stats[loss][modality][1].avg, best_acc[modality])

                state = {
                    'epoch': epoch + 1,
                    'mode': modality,
                    'net': self.args["net"],
                    'state_dict': self.models[modality].state_dict(),
                    'best_acc': best_acc[modality],
                    'optimizer': self.optimizer.state_dict(),
                    'iteration': self.iteration
                }

                save_checkpoint(
                    state=state,
                    mode=modality,
                    is_best=is_best,
                    filename=os.path.join(
                        self.model_path, 'mode_' + modality + '_epoch%s.pth.tar' % str(epoch + 1)
                    ),
                    gap=3,
                    keep_all=False
                )

        print('Training from ep %d to ep %d finished' % (self.args["start_epoch"], self.args["epochs"]))

        total_stats = {
            'train': {
                'losses': train_losses,
                'stats': train_stats,
            },
            'val': {
                'losses': val_losses,
                'stats': val_stats,
            }
        }

        return total_stats

    def _initialize_weights(self, module):
        for name, param in module.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.orthogonal_(param, 1)
        # other resnet weights have been initialized in resnet itself


def run_multi_modal_training(args):

    torch.manual_seed(0)
    np.random.seed(0)

    # Update the batch size according to the number of GPUs
    if torch.cuda.is_available():
        args.batch_size *= torch.cuda.device_count()
        args.num_workers *= int(np.sqrt(2 * torch.cuda.device_count()))

    args.num_classes = mu.get_num_classes(args.dataset)
    args.device = torch.device('cuda') if torch.cuda.is_available() else torch.device("cpu")

    # FIXME: support multiple views from the same modality
    args.modes = get_modality_list(args.modalities)
    args.restore_ckpts = get_modality_restore_ckpts(args)
    args.old_lr = None

    args.img_path, args.model_path = mu.set_multi_modal_path(args)

    models = {}
    for mode in args.modes:
        tmp_args = deepcopy(args)
        tmp_args.mode = mode

        # NOTE: Only image modality is allowed to go above 128x128 resolution
        # Remove this statement if you have different sized modalities
        if mode != mu.ImgMode:
            tmp_args.net = 'resnet18'
            tmp_args.img_dim = min(tmp_args.img_dim, 128)

        # NOTE: We only had 64x64 images for segmasks and keypoints
        if mode in [mu.KeypointHeatmap, mu.SegMask]:
            tmp_args.img_dim = 64

        tmp_args_dict = deepcopy(vars(tmp_args))

        model = DpcRnn(tmp_args_dict)

        if args.restore_ckpts[mode] is not None:
            # First try to load it hoping it's stored without the dataParallel
            print('Model saved in dataParallel form')
            model = m3d.get_parallel_model(model)
            model = mu.load_model(model, args.restore_ckpts[mode])
        else:
            model = m3d.get_parallel_model(model)

        # Freeze the required layers
        if args.train_what == 'last':
            for name, param in model.resnet.named_parameters():
                param.requires_grad = False

        print('\n=========Check Grad: {}============'.format(mode))
        for name, param in model.named_parameters():
            if param.requires_grad == False:
                print(name, param.requires_grad)
        print('=================================\n')

        models[mode] = model.to(args.device)

    # Restore from an earlier checkpoint
    args_dict = deepcopy(vars(args))

    args_dict["models"] = models
    args_dict["data_sources"] = '_'.join(args_dict["modes"]) + "_labels"

    model_trainer = MultiModalModelTrainer(args_dict)
    model_trainer = model_trainer.to(args.device)

    stats = model_trainer.train_module()

    return model_trainer, stats


if __name__ == '__main__':

    parser = mu.get_multi_modal_model_train_args()
    args = parser.parse_args()

    run_multi_modal_training(args)
