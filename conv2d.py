import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

class Conv2d(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 bias=True):

        super(Conv2d, self).__init__()
        """
        An implementation of a convolutional layer.

        The input consists of N data points, each with C channels, height H and
        width W. We convolve each input with F different filters, where each filter
        spans all C channels and has height HH and width WW.

        Parameters:
        - w: Filter weights of shape (F, C, HH, WW)
        - b: Biases, of shape (F,)
        - kernel_size: Size of the convolving kernel
        - stride: The number of pixels between adjacent receptive fields in the
            horizontal and vertical directions.
        - padding: The number of pixels that will be used to zero-pad the input.
        """
        if isinstance(kernel_size, tuple):
            kernel_x, kernel_y = kernel_size
        elif isinstance(kernel_size, int):
            kernel_x = kernel_y = kernel_size
        
        # Weights have shape [out_channels, in_channels, kernel_x, kernel_y]
        # Weights sampled from uniform dist. U(-sqrt(k), sqrt(k)), where
        # k = 1/(in_channels*kernel_x*kernel_y)
        k = 1/(in_channels*kernel_x*kernel_y)
        self.w = torch.rand(out_channels, in_channels, kernel_x, kernel_y)*2*math.sqrt(k) - math.sqrt(k)
        # Bias should have shape [out_channels]
        self.bias = bias
        self.b = None
        if self.bias: 
            # biases initialized similarly to weights
            self.b = torch.rand(out_channels)*2*math.sqrt(k) - math.sqrt(k)
        self.F = out_channels
        self.C = in_channels
        self.kernel_size = (kernel_x, kernel_y)

        if isinstance(stride, tuple):
            self.stride = stride
        elif isinstance(stride, int):
            self.stride = (stride, stride)

        if isinstance(padding, tuple):
            self.padding = padding
        elif isinstance(padding, int):
            self.padding = (padding, padding)
        
        
    def forward(self, x):
        """
        Input:
        - x: Input data of shape (N, C, H, W)
        Output:
        - out: Output data, of shape (N, F, H', W').
        """                      
        
        N, C, H, W = x.size()
        H_out = int((H + 2*self.padding[0] - (self.kernel_size[0] - 1) - 1)/self.stride[0] + 1)
        W_out = int((W + 2*self.padding[1] - (self.kernel_size[1] - 1) - 1)/self.stride[1] + 1)
        output_size = (N, self.F, H_out, W_out)

        # Convolution is equivalent with Unfold + Matrix Multiplication + Fold

        # extract sliding local blocks from batched input tensor
        # x_unf has shape [N, C*kernel_x*kernel_y, L], where L = # blocks
        x_unf = F.unfold(input=x, kernel_size=self.kernel_size, padding=self.padding, stride=self.stride)
        # number of patches
        L = x_unf.size(-1)

        # apply convolution
        # out_unf.shape = (N, F, L)
        if self.bias:
            b = np.broadcast_to(self.b, shape=(N, L, self.F))
            b = torch.tensor(b)
            # b.shape = (N, L, F)
            out_unf = (x_unf.transpose(1, 2).matmul(self.w.view(self.w.size(0), -1).t()) + b).transpose(1, 2)
        else:
            out_unf = x_unf.transpose(1, 2).matmul(self.w.view(self.w.size(0), -1).t()).transpose(1, 2)

        # recombine array of sliding local blocks (after convolution) into a large containing tensor
        out = out_unf.view(output_size)
        return out