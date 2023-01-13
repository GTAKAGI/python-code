# -*- coding: utf-8 -*-

import numpy
import torch

dtype = torch.float

class SurrGradSpike(torch.autograd.Function):
    """Step function"""
    
    scale = 100.0 # controls steepness of surrogate gradient
    @staticmethod
    def forward(ctx, input):
        """
        In the forward pass we compute a step function of the input Tensor
        and return it. ctx is a context object that we use to stash information which 
        we need to later backpropagate our error signals. To achieve this we use the 
        ctx.save_for_backward method.
        """
        ctx.save_for_backward(input)
        out = torch.zeros_like(input)
        out[input > 0] = 1.0
        return out
        
    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor we need to compute the 
        surrogate gradient of the loss with respect to the input. 
        Here we use the normalized negative part of a fast sigmoid 
        as this was done in Zenke & Ganguli (2018).
        """
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad = grad_input/(SurrGradSpike.scale*torch.abs(input)+1.0)**2
        return grad

spike_fn = SurrGradSpike.apply

"""
def spike_fn(x):
    out = torch.zeros_like(x)

    out[x>0]=1.0
    return out
"""
