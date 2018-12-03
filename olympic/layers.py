from torch import nn
import torch.nn.functional as F


class Flatten(nn.Module):
    """Module that flattens N-dimensional Tensor of shape [batch_size, d1, d2, ..., dn]
    to 2-dimensional Tensor of shape [batch_size, d1*d2*...*dn].
    """
    def forward(self, input):
        """Converts N-dimensional Tensor of shape [batch_size, d1, d2, ..., dn] to 2-dimensional Tensor
        of shape [batch_size, d1*d2*...*dn].

        # Arguments
           input: Input tensor

        # Returns
            output:
        """
        return input.view(input.size(0), -1)


class GlobalMaxPool1d(nn.Module):
    """Performs global max pooling over the entire length of a batched 1D tensor

    # Arguments
        input: Input tensor
    """
    def forward(self, input):
        return F.max_pool1d(input, kernel_size=input.size()[2:]).view(-1, input.size(1))


class GlobalAvgPool1d(nn.Module):
    """Performs global average pooling over the entire length of a batched 1D tensor

        # Arguments
            input: Input tensor
        """
    def forward(self, *input):
        return F.avg_pool1d(input, kernel_size=input.size()[2:]).view(-1, input.size(1))


class GlobalMaxPool2d(nn.Module):
    """Performs global max pooling over the entire height and width of a batched 2D tensor

    # Arguments
        input: Input tensor
    """
    def forward(self, input):
        return F.max_pool2d(input, kernel_size=input.size()[2:]).view(-1, input.size(1))


class GlobalAvgPool2d(nn.Module):
    """Performs global average pooling over the entire height and width of a batched 2D tensor

    # Arguments
        input: Input tensor
    """
    def forward(self, input):
        return F.avg_pool2d(input, kernel_size=input.size()[2:]).view(-1, input.size(1))
