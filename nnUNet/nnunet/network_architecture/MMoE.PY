import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
# from nnunet.utilities.nd_softmax import softmax_helper
from torch import nn
from nnunet.network_architecture.initialization import InitWeights_He
from nnunet.network_architecture.neural_network import SegmentationNetwork
import torch.nn.functional

class AbsArchitecture(nn.Module):
    r"""An abstract class for MTL architectures.
    Args:
        task_name (list): A list of strings for all tasks.
        encoder_class (class): A neural network class.
        decoders (dict): A dictionary of name-decoder pairs of type (:class:`str`, :class:`torch.nn.Module`).
        rep_grad (bool): If ``True``, the gradient of the representation for each task can be computed.
        multi_input (bool): Is ``True`` if each task has its own input data, otherwise is ``False``. 
        device (torch.device): The device where model and data will be allocated. 
        kwargs (dict): A dictionary of hyperparameters of architectures.
     
    """
    def __init__(self, multi_input, device, **kwargs):
        super(AbsArchitecture, self).__init__()
        
        self.multi_input = multi_input
        self.device = device
        self.kwargs = kwargs
        
    
        


class MMoE(SegmentationNetwork):
    r"""Multi-gate Mixture-of-Experts (MMoE).
    
    This method is proposed in `Modeling Task Relationships in Multi-task Learning with Multi-gate Mixture-of-Experts (KDD 2018) <https://dl.acm.org/doi/10.1145/3219819.3220007>`_ \
    and implemented by us.
    Args:
        img_size (list): The size of input data. For example, [3, 244, 244] denotes input images with size 3x224x224.
        num_experts (int): The number of experts shared for all tasks. Each expert is an encoder network.
    """
    def __init__(self, task_name, encoder_class, decoders, rep_grad, multi_input, device, **kwargs):
        super(MMoE, self).__init__(task_name, encoder_class, decoders, rep_grad, multi_input, device, **kwargs)
        
        self.img_size = self.kwargs['img_size']
        self.input_size = np.array(self.img_size, dtype=int).prod()
        self.num_experts = self.kwargs['num_experts'][0]
        self.experts_shared = nn.ModuleList([encoder_class() for _ in range(self.num_experts)])
        self.gate_specific = nn.ModuleDict({task: nn.Sequential(nn.Linear(self.input_size, self.num_experts),
                                                                nn.Softmax(dim=-1)) for task in self.task_name})
        
    def forward(self, inputs, task_name=None):
        experts_shared_rep = torch.stack([e(inputs) for e in self.experts_shared])
        out = {}
        for task in self.task_name:
            if task_name is not None and task != task_name:
                continue
            selector = self.gate_specific[task](torch.flatten(inputs, start_dim=1)) 
            gate_rep = torch.einsum('ij..., ji -> j...', experts_shared_rep, selector)
            gate_rep = self._prepare_rep(gate_rep, task, same_rep=False)
            out[task] = self.decoders[task](gate_rep)
        return out
    
    def get_share_params(self):
        return self.experts_shared.parameters()

    def zero_grad_share_params(self):
        self.experts_shared.zero_grad()
        
    def _prepare_rep(self, rep, task, same_rep=None):
        if self.rep_grad:
            if not same_rep:
                self.rep[task] = rep
            else:
                self.rep = rep
            self.rep_tasks[task] = rep.detach().clone()
            self.rep_tasks[task].requires_grad = True
            return self.rep_tasks[task]
        else:
            return rep