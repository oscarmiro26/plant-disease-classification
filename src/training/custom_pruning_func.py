from typing import Optional, Callable, cast
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.utils import parametrize
from torch.nn.utils.parametrize import ParametrizationList
from torch.ao.pruning._experimental.pruner.parametrization import FakeStructuredSparsity
from torch.ao.pruning._experimental.pruner.prune_functions import (
    _get_adjusted_next_layer_bias,
    _prune_conv2d_helper,
    prune_conv2d_padded,
    _prune_module_bias,
    _propagate_module_bias,
)
from torch_pruning.pruner.function import (
     prune_batchnorm_out_channels,
)


def prune_conv2d_bn2d(
          conv2d_1: nn.Conv2d,
          bn2d: nn.BatchNorm2d,
     ):
          """
          Fusion Pattern for conv2d -> bn2d layers
          """
          mask = _prune_conv2d_helper(conv2d_1)
          pruned_idxs = (~mask.bool()).nonzero(as_tuple=False).view(-1).tolist()
          if not pruned_idxs:
               return

          prune_batchnorm_out_channels(bn2d, pruned_idxs)

          return mask    

def prune_conv2d_bn2d_conv2d(
          conv2d_1: nn.Conv2d,
          bn2d: nn.BatchNorm2d,
          conv2d_2: nn.Conv2d,
     ):
     prune_conv2d_bn2d_relu_conv2d(
          conv2d_1=conv2d_1,
          bn2d=bn2d,
          relu=None,
          conv2d_2=conv2d_2,
     )

def prune_conv2d_bn2d_relu_conv2d(
          conv2d_1: nn.Conv2d,
          bn2d: nn.BatchNorm2d,
          relu: Optional[nn.ReLU] = None, 
          conv2d_2: Optional[nn.Conv2d] = None,
     ):
     """
     Fusion Pattern for conv2d -> bn2d -> relu -> conv2d layers
     """
     parametrization_dict = cast(nn.ModuleDict, conv2d_1.parametrizations)
     weight_parameterizations = cast(ParametrizationList, parametrization_dict.weight)
     for p in weight_parameterizations:
          if isinstance(p, FakeStructuredSparsity):
               mask = cast(Tensor, p.mask)
     prune_bias = getattr(conv2d_1, "prune_bias", False)
     if (
          hasattr(conv2d_2, "padding")
          and cast(tuple[int], conv2d_2.padding) > (0, 0)
          and (conv2d_1.bias is not None or getattr(conv2d_1, "_bias", None) is not None)
     ):
          prune_conv2d_padded(conv2d_1)
     else:
          mask = prune_conv2d_bn2d(
               conv2d_1, bn2d
          )

          if prune_bias:
               _prune_module_bias(conv2d_1, mask)
          else:
               pruned_biases = _propagate_module_bias(conv2d_1, mask)
               if pruned_biases is not None and relu is not None:
                    pruned_biases = relu(pruned_biases)
                    conv2d_2.bias = _get_adjusted_next_layer_bias(
                         conv2d_2, pruned_biases, mask
                    )

          if (not ( hasattr(conv2d_2, "padding") and cast(tuple[int], conv2d_2.padding) > (0, 0)) or conv2d_1.bias is None):
               with torch.no_grad():
                    if parametrize.is_parametrized(conv2d_2):
                         parametrization_dict = cast(
                         nn.ModuleDict, conv2d_2.parametrizations
                    )
                         weight_parameterizations = cast(
                         ParametrizationList, parametrization_dict.weight
                         )
                         weight_parameterizations.original = nn.Parameter(
                         weight_parameterizations.original[:, mask]
                         )
                         conv2d_2.in_channels = weight_parameterizations.original.shape[1]
                    else:
                         conv2d_2.weight = nn.Parameter(conv2d_2.weight[:, mask])
                         conv2d_2.in_channels = conv2d_2.weight.shape[1]

def prune_conv2d_bn2d_add_relu_conv2d(
     conv2d_1: nn.Conv2d,
     bn2d: nn.BatchNorm2d,
     add: Optional[Callable] = None,
     relu: nn.ReLU = None,
     conv2d_3: nn.Conv2d = None,
):
     """
     Fusion Pattern for conv2d -> add -> relu -> conv2d layers
     """
     _ = prune_conv2d_bn2d_relu_conv2d(conv2d_1, bn2d, relu, conv2d_3)

def prune_conv2d_bn2d_add_relu_pool(
          conv2d: nn.Conv2d,
          bn2d: nn.BatchNorm2d,
          add: Optional[Callable] = None,
          relu: nn.ReLU = None,
          pool: Optional[nn.AdaptiveAvgPool2d] = None,
     ):
     """
     Fusion Pattern for conv2d -> add -> relu -> pool layers
     """
     _ = prune_conv2d_bn2d_relu_conv2d(conv2d, relu, bn2d)

def prune_conv2d_bn2d_ds_add_relu_conv2d(
    conv2d_1:  nn.Conv2d,
    bn2d_1:    nn.BatchNorm2d,
    conv2d_2:  nn.Conv2d,
    bn2d_2:    nn.BatchNorm2d,
    add_fn:    Optional[Callable] = None,
    relu:      nn.ReLU       = None,
    conv2d_3:  nn.Conv2d     = None,
):
    """
    Fusion Pattern for conv2d_1 -> bn2d_1 -> downsample.conv -> downsample.bn
                   -> add -> relu -> conv2d_3
    
    1) prune conv2d_1 + bn2d_1
    2) apply same mask to downsample path (conv2d_2 + bn2d_2)
    3) adjust conv2d_3.in_channels to match pruned output
    """
    # 1) prune first conv+bn
    mask = prune_conv2d_bn2d(conv2d_1, bn2d_1)
    if mask is None:
        return

    pruned_idxs = (~mask.bool()).nonzero(as_tuple=False).view(-1).tolist()
    if not pruned_idxs:
        return mask

    # 2) prune downsample Conv2d
    with torch.no_grad():
        conv2d_2.weight = nn.Parameter(conv2d_2.weight[mask])
        conv2d_2.out_channels = conv2d_2.weight.shape[0]
        if conv2d_2.bias is not None:
            conv2d_2.bias = nn.Parameter(conv2d_2.bias[mask])

    # prune downsample BatchNorm2d
    prune_batchnorm_out_channels(bn2d_2, pruned_idxs)

    # 3) adjust next conv’s input channels
    if conv2d_3 is not None:
        with torch.no_grad():
            if parametrize.is_parametrized(conv2d_3):
                pdict = cast(nn.ModuleDict, conv2d_3.parametrizations)
                wpars = cast(ParametrizationList, pdict.weight)
                wpars.original = nn.Parameter(wpars.original[:, mask])
                conv2d_3.in_channels = wpars.original.shape[1]
            else:
                conv2d_3.weight = nn.Parameter(conv2d_3.weight[:, mask])
                conv2d_3.in_channels = conv2d_3.weight.shape[1]

    return mask

