# Based on a script from Zennit:
# https://github.com/chr5tphr/zennit/blob/master/src/zennit/torchvision.py
# The changes inspired by F. Pahde, G. Ü. Yolcu, A. Binder, W. Samek, and S. Lapuschkin,
# “Optimizing Explanations by Network Canonization and Hyperparameter Search.”

import torch
import torchvision
from torch import Tensor
from torchvision.models.efficientnet import FusedMBConv, MBConv
from zennit.canonizers import (AttributeCanonizer, CompositeCanonizer,
                               SequentialMergeBatchNorm)
from zennit.layer import Sum
from zennit.types import SubclassMeta


class SqueezeExcitation(metaclass=SubclassMeta):
    '''Abstract base class that describes squeeze excitation modules.'''
    __subclass__ = (
        torchvision.ops.SqueezeExcitation,
    )


class EfficientNetMBConvCanonizer(AttributeCanonizer):
    '''Canonizer specifically for MBConv of torchvision.models.efficientnet* type models.'''
    def __init__(self):
        super().__init__(self._attribute_map)

    @classmethod
    def _attribute_map(cls, name, module):
        '''Create a forward function and a Sum module to overload as new attributes for module.
        Parameters
        ----------
        name : string
            Name by which the module is identified.
        module : obj:`torch.nn.Module`
            Instance of a module. If this is a MBConv layer, the appropriate attributes
            to overload are returned.
        Returns
        -------
        None or dict
            None if `module` is not an instance of MBConv, otherwise the appropriate
            attributes to overload onto the module instance.
        '''
        if isinstance(module, MBConv):
            attributes = {
                'forward': cls.forward.__get__(module),
                'canonizer_sum': Sum(),
            }
            return attributes
        return None

    @staticmethod
    def forward(self, input: Tensor) -> Tensor:
        '''Modified MBConv forward for EfficientNet.'''
        identity = input
        result = self.block(input)
        if self.use_res_connect:
            result = self.stochastic_depth(result)
            result = torch.stack([identity, result], dim=-1)
            result = self.canonizer_sum(result)
        return result


class EfficientNetFusedMBConvCanonizer(AttributeCanonizer):
    '''Canonizer specifically for FusedMBConv of torchvision.models.efficientnet* type models.'''
    def __init__(self):
        super().__init__(self._attribute_map)

    @classmethod
    def _attribute_map(cls, name, module):
        '''Create a forward function and a Sum module to overload as new attributes for module.
        Parameters
        ----------
        name : string
            Name by which the module is identified.
        module : obj:`torch.nn.Module`
            Instance of a module. If this is a FusedMBConv layer, the appropriate attributes
            to overload are returned.
        Returns
        -------
        None or dict
            None if `module` is not an instance of FusedMBConv, otherwise the appropriate
            attributes to overload onto the module instance.
        '''
        if isinstance(module, FusedMBConv):
            attributes = {
                'forward': cls.forward.__get__(module),
                'canonizer_sum': Sum(),
            }
            return attributes
        return None

    @staticmethod
    def forward(self, input: Tensor) -> Tensor:
        '''Modified FusedMBConv forward for EfficientNet.'''
        identity = input
        result = self.block(input)
        if self.use_res_connect:
            result = self.stochastic_depth(result)
            result = torch.stack([identity, result], dim=-1)
            result = self.canonizer_sum(result)
        return result


class EfficientNetCanonizer(CompositeCanonizer):
    '''Canonizer for torchvision.models.efficientnet* type models. This applies
    SequentialMergeBatchNorm, as well as add a Sum module to the MBConv modules and
    overload their forward method to use the Sum module instead of
    simply adding two tensors, such that forward and backward hooks may be applied.'''
    def __init__(self):
        super().__init__((
            SequentialMergeBatchNorm(),
            EfficientNetMBConvCanonizer(),
            EfficientNetFusedMBConvCanonizer(),
        ))
