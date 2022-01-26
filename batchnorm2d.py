import torch
import torch.nn as nn

class BatchNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-05, momentum=0.1):
        super().__init__()
        """
        An implementation of a Batch Normalization over a mini-batch of 2D inputs.

        The mean and standard-deviation are calculated per-dimension over the
        mini-batches and gamma and beta are learnable parameter vectors of
        size num_features.

        Parameters:
        - num_features: C from an expected input of size (N, C, H, W).
        - eps: a value added to the denominator for numerical stability. Default: 1e-5
        - momentum: momentum - the value used for the running_mean and running_var
        computation. Default: 0.1
        - gamma: the learnable weights of shape (1, num_features, 1, 1).
        - beta: the learnable bias of the module of shape (1, num_features, 1, 1).
        """

        # num_features == C
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum

        # self.register_parameter is not used as it was mentioned on piazza
        # that this will be overridden
        # By default, the elements of γ are set to 1 and the elements of β are set to 0
        self.gamma = torch.ones(size=(1,num_features,1,1))  # shape (1, C, 1, 1)
        self.beta = torch.zeros(size=(1,num_features,1,1))  # shape (1, C, 1, 1)

        self.running_mean = torch.zeros(size=(1,num_features,1,1))  # shape (1, C, 1, 1)
        self.running_var = torch.ones(size=(1,num_features,1,1))  # shape (1, C, 1, 1)
        
        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    def forward(self, x):
        """
        During training this layer keeps running estimates of its computed mean and
        variance, which are then used for normalization during evaluation.
        Input:
        - x: Input data of shape (N, C, H, W)
        Output:
        - out: Output data of shape (N, C, H, W) (same shape as input)
        """

        # Training mode
        if torch.is_grad_enabled():
            # calculate mean and variance on the channel dimension (axis=1)
            # shapes of mu, var --> (1, C, 1, 1)
            mu, var = torch.var_mean(x, dim=(0,2,3), unbiased=False, keepdim=True)
            x_hat = (x - mu) / torch.sqrt(var + self.eps)

            # update running mean and variance
            self.running_mean = self.momentum*mu + (1 - self.momentum)*self.running_mean
            self.running_var = self.momentum*var + (1 - self.momentum)*self.running_var
        else:
            # Test mode
            x_hat = (x - self.running_mean) / torch.sqrt(self.running_var + self.eps)
        
        out = self.gamma*x_hat + self.beta

        return out


if __name__ == "__main__":
    pass