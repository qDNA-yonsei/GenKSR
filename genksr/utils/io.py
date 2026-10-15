from __future__ import annotations

import os
from collections import OrderedDict
from fnmatch import fnmatch
from typing import TYPE_CHECKING, OrderedDict as OrderedDict_T

if TYPE_CHECKING:
    import torch


def process_weight_names(ckpt: OrderedDict_T[str, torch.Tensor]) -> OrderedDict_T[str, torch.Tensor]:
    return OrderedDict([(str(k).replace('module.', ''), v) for k, v in ckpt.items()])


def filter_folder_for_files(folder, file_pattern):
    return [os.path.join(folder, fn) for fn in os.listdir(folder) if fnmatch(fn, file_pattern)]
