import sys
import math
import torch

import torch.nn as nn
import sim_utils as su
import model_utils as mu
import torch.nn.functional as F
sys.path.append('../backbone')

from select_backbone import select_resnet
from convrnn import ConvGRU


eps = 1e-7
INF = 25.0


class MyDataParallel(torch.nn.DataParallel):
    """
    Allow nn.DataParallel to call model's attributes.
    """
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)


def get_parallel_model(model):
    if torch.cuda.is_available():
        dev_count = torch.cuda.device_count()
        print("Using {} GPUs".format(dev_count))
        model = MyDataParallel(model, device_ids=list(range(dev_count)))
    return model


def get_num_channels(modality):
    if modality.startswith(mu.ImgMode):
        return 3
    elif modality == mu.FlowMode:
        return 2
    elif modality == mu.FnbFlowMode:
        return 2
    elif modality == mu.KeypointHeatmap:
        return 17
    elif modality == mu.SegMask:
        return 1
    else:
        assert False, "Invalid modality: {}".format(modality)


class ImageFetCombiner(nn.Module):

    def __init__(self, img_fet_dim, img_segments):
        super(ImageFetCombiner, self).__init__()

        # Input feature dimension is [B, dim, s, s]
        self.dim = img_fet_dim
        self.s = img_segments
        self.flat_dim = self.dim * self.s * self.s

        layers = []
        if self.s == 7:
            layers.append(nn.MaxPool2d(2, 2, padding=1))
            layers.append(nn.MaxPool2d(2, 2))
            layers.append(nn.AvgPool2d(2, 2))
        if self.s == 4:
            layers.append(nn.MaxPool2d(2, 2))
            layers.append(nn.AvgPool2d(2, 2))
        elif self.s == 2:
            layers.append(nn.AvgPool2d(2, 2))

        # input is B x dim x s x s
        self.feature = nn.Sequential(*layers)
        # TODO: Normalize
        # Output is B x dim

    def forward(self, input: torch.Tensor):
        # input is B, N, D, s, s
        B, N, D, s, s = input.shape
        input = input.view(B * N, D, s, s)
        y = self.feature(input)
        y = y.reshape(B, N, -1)
        return y


class IdentityFlatten(nn.Module):

    def __init__(self):
        super(IdentityFlatten, self).__init__()

    def forward(self, input: torch.Tensor):
        # input is B, N, D, s, s
        B, N, D, s, s = input.shape
        return input.reshape(B, N, -1)


class DpcRnn(nn.Module):

    def get_modality_feature_extractor(self):
        if self.mode.split('-')[0] in [mu.ImgMode, mu.FlowMode, mu.KeypointHeatmap, mu.SegMask]:
            return ImageFetCombiner(self.final_feature_size, self.last_size)
        else:
            assert False, "Invalid mode provided: {}".format(self.mode)

    '''DPC with RNN'''
    def __init__(self, args):
        super(DpcRnn, self).__init__()

        torch.cuda.manual_seed(233)

        print('Using DPC-RNN model for mode: {}'.format(args["mode"]))
        self.num_seq = args["num_seq"]
        self.seq_len = args["seq_len"]
        self.pred_step = args["pred_step"]
        self.sample_size = args["img_dim"]
        self.is_supervision_enabled = mu.SupervisionLoss in args["losses"]
        self.last_duration = int(math.ceil(self.seq_len / 4))
        self.last_size = int(math.ceil(self.sample_size / 32))
        print('final feature map has size %dx%d' % (self.last_size, self.last_size))
        print('Supervision enabled: {}'.format(self.is_supervision_enabled))

        self.mode = args["mode"]
        self.in_channels = get_num_channels(self.mode)
        self.l2_norm = True

        track_running_stats = True
        print("Track running stats: {}".format(track_running_stats))
        self.backbone, self.param = select_resnet(
            args["net"], track_running_stats=track_running_stats, in_channels=self.in_channels
        )

        # params for GRU
        self.param['num_layers'] = 1
        self.param['hidden_size'] = self.param['feature_size']

        # param for current model
        self.final_feature_size = self.param["feature_size"]
        # self.final_feature_size = self.param['hidden_size'] * (self.last_size ** 2)
        self.total_feature_size = self.param['hidden_size'] * (self.last_size ** 2)

        self.agg = ConvGRU(input_size=self.param['feature_size'],
                               hidden_size=self.param['hidden_size'],
                               kernel_size=1,
                               num_layers=self.param['num_layers'])
        self.network_pred = nn.Sequential(
                                nn.Conv2d(self.param['feature_size'], self.param['feature_size'], kernel_size=1, padding=0),
                                nn.ReLU(inplace=True),
                                nn.Conv2d(self.param['feature_size'], self.param['feature_size'], kernel_size=1, padding=0)
                        )

        self.compiled_features = self.get_modality_feature_extractor()
        self.interModeDotHandler = su.InterModeDotHandler(self.last_size)
        self.cosSimHandler = su.CosSimHandler()

        self.mask = None
        # self.relu = nn.ReLU(inplace=False)
        self._initialize_weights(self.agg)
        self._initialize_weights(self.network_pred)

        if self.is_supervision_enabled:
            self.initialize_supervised_inference_layers()

    def initialize_supervised_inference_layers(self):
        self.final_bn = nn.BatchNorm1d(self.param['feature_size'])
        self.final_bn.weight.data.fill_(1)
        self.final_bn.bias.data.zero_()

        # Update the number of classes here
        self.num_classes = 75
        self.dropout = 0.5
        self.final_fc = nn.Sequential(
            nn.Dropout(self.dropout),
            nn.Linear(self.param['feature_size'], self.num_classes),
        )

        self._initialize_weights(self.final_fc)

    def get_representation(self, block, detach=False):

        (B, N, C, SL, H, W) = block.shape
        block = block.view(B*N, C, SL, H, W)
        feature = self.backbone(block)

        del block
        feature = F.relu(feature)

        feature = F.avg_pool3d(feature, (self.last_duration, 1, 1), stride=1)
        feature = feature.view(B, N, self.param['feature_size'], self.last_size, self.last_size) # [B*N,D,last_size,last_size]
        context, _ = self.agg(feature)
        context = context[:,-1,:].unsqueeze(1)
        context = F.avg_pool3d(context, (1, self.last_size, self.last_size), stride=1).squeeze(-1).squeeze(-1)
        del feature

        if self.l2_norm:
            context = self.cosSimHandler.l2NormedVec(context, dim=2)

        # Return detached version if required
        if detach:
            return context.detach()
        else:
            return context

    def compute_cdot_features(self, feature):
        comp_feature = self.compiled_features(feature).unsqueeze(3).unsqueeze(3)
        cdot, cdot_fet = self.interModeDotHandler(comp_fet=comp_feature)
        return cdot, cdot_fet

    def forward(self, block, ret_rep=False):
        # ret_cdot values: [c, z, zt]

        # block: [B, N, C, SL, W, H]
        # B: Batch, N: Number of sequences per instance, C: Channels, SL: Sequence Length, W, H: Dims

        ### extract feature ###
        (B, N, C, SL, H, W) = block.shape

        block = block.view(B*N, C, SL, H, W)
        feature = self.backbone(block)

        del block

        feature = F.avg_pool3d(feature, (self.last_duration, 1, 1), stride=(1, 1, 1))

        if self.l2_norm:
            feature = self.cosSimHandler.l2NormedVec(feature, dim=1)

        # before ReLU, (-inf, +inf)
        feature_inf_all = feature.view(B, N, self.param['feature_size'], self.last_size, self.last_size)
        feature = feature.view(B, N, self.param['feature_size'], self.last_size, self.last_size)

        # Generate feature for future frames
        feature_inf = feature_inf_all[:, N - self.pred_step::, :].contiguous()

        del feature_inf_all

        # aggregate and predict overall context
        if self.is_supervision_enabled:
            context, _ = self.agg(feature)
            context = context[:, -1, :].unsqueeze(1)
            context = F.avg_pool3d(context, (1, self.last_size, self.last_size), stride=1).squeeze(-1).squeeze(-1)

            # [B,N,C] -> [B,C,N] -> BN() -> [B,N,C], because BN operates on id=1 channel.
            context = self.final_bn(context.transpose(-1, -2)).transpose(-1,-2)
            probabilities = self.final_fc(context).view(B, self.num_classes)

        ### aggregate, predict future ###
        # Generate inferred future (stored in feature_inf) through the initial frames
        _, hidden = self.agg(feature[:, 0:N-self.pred_step, :].contiguous())

        if self.l2_norm:
            hidden = self.cosSimHandler.l2NormedVec(hidden, dim=2)

        # Get the last hidden state, this gives us the predicted representation
        # after tanh, (-1,1). get the hidden state of last layer, last time step
        hidden = hidden[:, -1, :]

        # Predict next pred_step time steps for this instance
        pred = []
        for i in range(self.pred_step):
            # sequentially pred future based on the hidden states
            p_tmp = self.network_pred(hidden)

            if self.l2_norm:
                p_tmp = self.cosSimHandler.l2NormedVec(p_tmp, dim=1)

            pred.append(p_tmp)
            _, hidden = self.agg(p_tmp.unsqueeze(1), hidden.unsqueeze(0))

            if self.l2_norm:
                hidden = self.cosSimHandler.l2NormedVec(hidden, dim=2)

            hidden = hidden[:, -1, :]
        # Contains the representations for each of the next pred steps
        pred = torch.stack(pred, 1) # B, pred_step, xxx

        # Both are of the form [B, pred_step, D, s, s]
        return pred, feature_inf, feature, probabilities, hidden

    def _initialize_weights(self, module):
        for name, param in module.named_parameters():
            if 'weight' in name:
                nn.init.orthogonal_(param, 1)
        # other resnet weights have been initialized in resnet itself

    def reset_mask(self):
        self.mask = None
