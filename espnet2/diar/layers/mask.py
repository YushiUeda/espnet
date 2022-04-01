# Implementation of the TCN proposed in
# Luo. et al.  "Conv-tasnet: Surpassing ideal time–frequency
# magnitude masking for speech separation."
#
# The code is based on:
# https://github.com/kaituoxu/Conv-TasNet/blob/master/src/conv_tasnet.py
#

from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from espnet2.diar.layers.abs_mask import AbsMask


class Mask(AbsMask):
    def __init__(
        self,
        input_dim: int,
        bottleneck_dim: int,
        max_num_spk: int = 3,
        mask_nonlinear="relu"
    ):
        """Basic Module of tasnet.

        Args:
            input_dim: Number of filters in autoencoder
            bottleneck_dim: Number of channels in bottleneck 1 * 1-conv block
            max_num_spk: Number of mask_conv1x1 modules (= Max number of speakers in the dataset)
            mask_nonlinear: use which non-linear function to generate mask
        """
        super().__init__()
        # Hyper-parameter
        self._max_num_spk = max_num_spk
        self.mask_nonlinear = mask_nonlinear
        self._output_size = bottleneck_dim
        # [M, B, K] -> [M, C*N, K]
        self.mask_conv1x1 = nn.ModuleList()
        for z in range(1, max_num_spk+1):
            self.mask_conv1x1.append(nn.Conv1d(bottleneck_dim, z * input_dim, 1, bias=False))

    @property
    def max_num_spk(self) -> int:
        return self._max_num_spk

    @property
    def output_size(self) -> int:
        return self._output_size

    def forward(self, input, ilens, bottleneck_feat, num_spk):
        """Keep this API same with TasNet.

        Args:
            input: [M, K, N], M is batch size
            ilens (torch.Tensor): (M,)
            bottleneck_feat: [M, K, B]
            num_spk: number of speakers

        Returns:
            masked (List[Union(torch.Tensor, ComplexTensor)]): [(M, K, N), ...]
            ilens (torch.Tensor): (M,)
            others predicted data, e.g. masks: OrderedDict[
                'mask_spk1': torch.Tensor(Batch, Frames, Freq),
                'mask_spk2': torch.Tensor(Batch, Frames, Freq),
                ...
                'mask_spkn': torch.Tensor(Batch, Frames, Freq),
            ]

        """
        M, K, N = input.size()
        bottleneck_feat = bottleneck_feat.transpose(1,2) #[M, B, K]
        score = self.mask_conv1x1[num_spk - 1](bottleneck_feat) # [M, B, K] -> [M, num_spk*N, K]
        # add other outputs of the module list with factor 0.0 to enable distributed training
        for z in range(self._max_num_spk):
            if z != num_spk - 1:
                score += 0.0 * F.interpolate(self.mask_conv1x1[z](bottleneck_feat).transpose(1,2), size=num_spk*N).transpose(1,2)
        score = score.view(M, num_spk, N, K)  # [M, num_spk*N, K] -> [M, num_spk, N, K]
        if self.mask_nonlinear == "softmax":
            est_mask = F.softmax(score, dim=1)
        elif self.mask_nonlinear == "relu":
            est_mask = F.relu(score)
        elif self.mask_nonlinear == "sigmoid":
            est_mask = torch.sigmoid(score)
        elif self.mask_nonlinear == "tanh":
            est_mask = torch.tanh(score)
        else:
            raise ValueError("Unsupported mask non-linear function")
        
        masks = est_mask.transpose(2, 3) # [M, num_spk, K, N]
        masks = masks.unbind(dim=1)  # List[M, K, N]

        masked = [input * m for m in masks]

        others = OrderedDict(
            zip(["mask_spk{}".format(i + 1) for i in range(len(masks))], masks)
        )

        return masked, ilens, others