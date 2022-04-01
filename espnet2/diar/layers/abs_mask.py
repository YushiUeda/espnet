from abc import ABC
from abc import abstractmethod
from collections import OrderedDict
from typing import Tuple

import torch


class AbsMask(torch.nn.Module, ABC):
    @property
    @abstractmethod
    def max_num_spk(self) -> int:
        raise NotImplementedError

    @property
    @abstractmethod
    def output_size(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def forward(
        self, 
        input, 
        ilens, 
        bottleneck_feat, 
        num_spk,
        ):

        raise NotImplementedError
        