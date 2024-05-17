"""MultiHeadedMLPModule."""
import copy

import torch
import torch.nn as nn

from garage.torch import NonLinearity
from garage.torch.modules import MultiHeadedMLPModule
from torch.nn import functional as F


class SefarMLPModule(MultiHeadedMLPModule):
    """SefarMLPModule Model.

    A PyTorch module composed only of a multi-layer perceptron (MLP) with
    multiple parallel output layers which maps real-valued inputs to
    real-valued outputs. The length of outputs is n_heads and shape of each
    output element is depend on each output dimension

    Args:
        n_heads (int): Number of different output layers
        input_dim (int): Dimension of the network input.
        output_dims (int or list or tuple): Dimension of the network output.
        hidden_sizes (list[int]): Output dimension of dense layer(s).
            For example, (32, 32) means this MLP consists of two
            hidden layers, each with 32 hidden units.
        hidden_nonlinearity (callable or torch.nn.Module or list or tuple):
            Activation function for intermediate dense layer(s).
            It should return a torch.Tensor. Set it to None to maintain a
            linear activation.
        hidden_w_init (callable): Initializer function for the weight
            of intermediate dense layer(s). The function should return a
            torch.Tensor.
        hidden_b_init (callable): Initializer function for the bias
            of intermediate dense layer(s). The function should return a
            torch.Tensor.
        output_nonlinearities (callable or torch.nn.Module or list or tuple):
            Activation function for output dense layer. It should return a
            torch.Tensor. Set it to None to maintain a linear activation.
            Size of the parameter should be 1 or equal to n_head
        output_w_inits (callable or list or tuple): Initializer function for
            the weight of output dense layer(s). The function should return a
            torch.Tensor. Size of the parameter should be 1 or equal to n_head
        output_b_inits (callable or list or tuple): Initializer function for
            the bias of output dense layer(s). The function should return a
            torch.Tensor. Size of the parameter should be 1 or equal to n_head
        layer_normalization (bool): Bool for using layer normalization or not.

    """

    def __init__(self,
                 input_dim,
                 output_dims,
                 hidden_sizes,
                 hidden_nonlinearity=torch.relu,
                 hidden_w_init=nn.init.xavier_normal_,
                 hidden_b_init=nn.init.zeros_,
                 output_nonlinearities=None,
                 output_w_inits=nn.init.xavier_normal_,
                 output_b_inits=nn.init.zeros_,
                 layer_normalization=False,
                 sparsity=0.5,
                 update_mask=False,
                 ):
        super().__init__(2, input_dim, output_dims, hidden_sizes,
                         hidden_nonlinearity, hidden_w_init,
                         hidden_b_init,
                         output_nonlinearities, output_w_inits,
                         output_b_inits,
                         layer_normalization)

        self.update_mask = update_mask
        self.sparsity = sparsity
        self._output_dims = output_dims

        mask = torch.rand(self._mask_input_size)
        mask[mask <= self.sparsity] = 0.
        mask[mask != 0] = 1.
        self._mask = mask

    def _update_mask(self):
        self._mask.uniform_()
        self._mask[self._mask <= self.sparsity] = 0.
        self._mask[self._mask != 0] = 1.

    @property
    def output_dim(self):
        """Return output dimension of network.

        Returns:
            int: Output dimension of network.

        """
        return self._output_dims

    # pylint: disable=arguments-differ
    def forward(self, input_val):
        """Forward method.

        Args:
            input_val (torch.Tensor): Input values with (N, *, input_dim)
                shape.

        Returns:
            List[torch.Tensor]: Output values

        """
        x = input_val
        for layer in self._layers:
            x = layer(x)

        # No sefar output
        outputs = [self._output_layers[0](x)]

        # Sefar output
        if self.update_mask:
            self._update_mask()
        outputs.append(self._output_layers[1](x*self._mask))

        return outputs
