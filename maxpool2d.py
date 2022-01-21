import torch
import torch.nn as nn
import torch.nn.functional as F

class MaxPool2d(nn.Module):
    def __init__(self, kernel_size):
        super(MaxPool2d, self).__init__()
        """
        An implementation of a max-pooling layer.

        Parameters:
        - kernel_size: the size of the window to take a max over
        """

        if isinstance(kernel_size, tuple):
           self.kernel_size = kernel_size
        elif isinstance(kernel_size, int):
            self.kernel_size = (kernel_size, kernel_size)
        
        self.stride = self.kernel_size
        # default 0 padding
        self.padding = (0, 0)


    def forward(self, x):
        """
        Input:
        - x: Input data of shape (N, C, H, W)
        Output:
        - out: Output data, of shape (N, C, H', W').
        """

        N, C, H, W = x.size()
        H_out = int((H + 2*self.padding[0] - (self.kernel_size[0] - 1) - 1)/self.stride[0] + 1)
        W_out = int((W + 2*self.padding[1] - (self.kernel_size[1] - 1) - 1)/self.stride[1] + 1)
        output_size = (N, C, H_out, W_out)

        out = torch.empty(size=output_size)
        # apply max pooling to each channel separately 
        for c in range(C):
            x_c = x[:, c, :, :]
            x_c = x_c.view(x_c.size(0), 1, x_c.size(1), x_c.size(2))
            x_c_unf = F.unfold(input=x_c, kernel_size=self.kernel_size, padding=self.padding, stride=self.stride)

            # apply max pooling
            out_c_unf = torch.amax(x_c_unf, dim=1)

            # convert to correct shape
            out_c = out_c_unf.view((N, H_out, W_out))
            out[:, c, :, :] = out_c

        return out