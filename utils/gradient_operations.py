import torch
from typing import Dict
import torch.nn as nn
from torch.autograd import Variable

def normalize_grad(g: torch.Tensor, scale: float = 0.1):
    return g / torch.norm(g) * scale


def save_and_clear_grad(model: nn.Module, normalize: bool = False, scale: float = 0.1):
    grad_dict: Dict[str, torch.Tensor] = {}
    flatten_grad = []
    for name, param in model.named_parameters():
        if param.grad is not None:
            if normalize:
                grad_dict[name] = normalize_grad(param.grad.detach().clone(), scale)
                flatten_grad.append(Variable(normalize_grad(param.grad.data.clone().flatten()), requires_grad=False))
            else:
                grad_dict[name] = param.grad.detach().clone()
                flatten_grad.append(Variable(param.grad.data.clone().flatten(), requires_grad=False))
            param.grad = torch.zeros_like(param.grad).to(param.device)
    return grad_dict, flatten_grad


def reattach_grad(model: nn.Module, gdict: Dict[str, torch.Tensor]):
    # check if all parameters are in gdict
    for name, param in model.named_parameters():
        if name not in gdict:
            raise ValueError(f'{name} not in gdict')

    # attach gradients
    for name, param in model.named_parameters():
        if param.grad is not None:
            param.grad = gdict[name]


def merge_gradients(gdict1: Dict[str, torch.Tensor], gdict2: Dict[str, torch.Tensor], weight):
    #if gdict1.keys() != gdict2.keys():
        #raise ValueError('gdict1 and gdict2 have different keys')
    for name, grad in gdict1.items():
        if name in gdict2.keys():
            gdict2[name] = weight[0] * grad + weight[1] * gdict2[name]
        else:
            gdict2[name] = weight[0] * grad #+ weight[1] * gdict2[name]
        #gdict2[name] += grad
    return gdict2


def zeroing_grad(model):
    for name, param in model.named_parameters():
        if param.grad is not None:
            param.grad = torch.zeros_like(param.grad).to(param.device)