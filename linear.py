import torch
import torch.nn as nn
import numpy as np
import math

class Linear(nn.Module):
    def __init__(self, in_channels, out_channels, bias=True):
        super(Linear, self).__init__()
        """
        An implementation of a Linear layer.

        Parameters:
        - weight: the learnable weights of the module of shape (in_channels, out_channels).
        - bias: the learnable bias of the module of shape (out_channels).
        """

        # Weight matrix has shape [out_channels, in_channels]
        # Weights sampled from uniform dist. U(-sqrt(k), sqrt(k)), where
        # k = 1/in_channels
        k = 1/in_channels
        self.w = torch.rand(out_channels, in_channels)*2*math.sqrt(k) - math.sqrt(k)
        # Bias should have shape [out_channels]
        self.bias = bias
        self.b = None
        if self.bias: 
            # biases initialized similarly to weights
            self.b = torch.rand(out_channels)*2*math.sqrt(k) - math.sqrt(k)
        
        self.in_channels = in_channels
        self.out_channels = out_channels


    def forward(self, x):
        """
        Input:
        - x: Input data of shape (N, *, H) where * means any number of additional
        dimensions and H = in_channels
        Output:
        - out: Output data of shape (N, *, H') where * means any number of additional
        dimensions and H' = out_channels
        """

        if self.bias:
            if len(x.size()) > 2:
                b_shape = []
                b_shape.extend(list(x.size()[1:-1]))
                b_shape.append(self.out_channels)
                b = np.broadcast_to(self.b, shape=b_shape)
                b = torch.tensor(b)
                out = torch.matmul(x, self.w.t()) + b
            else:
                out = torch.matmul(x, self.w.t()) + self.b
        else:
            out = torch.matmul(x, self.w.t())
        
        return out