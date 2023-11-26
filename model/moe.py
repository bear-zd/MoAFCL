# Sparsely-Gated Mixture-of-adapters Layers.
# See "Outrageously Large Neural Networks"
# https://arxiv.org/abs/1701.06538
#
# Author: David Rau
#
# The code is based on the TensorFlow implementation:
# https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/utils/adapter_utils.py


import torch
import torch.nn as nn
from torch.distributions.normal import Normal
import numpy as np
import copy
from copy import deepcopy
import torch.nn.functional as F


class SparseDispatcher(object):
    """Helper for implementing a mixture of adapters.
    The purpose of this class is to create input minibatches for the
    adapters and to combine the results of the adapters to form a unified
    output tensor.
    There are two functions:
    dispatch - take an input Tensor and create input Tensors for each adapter.
    combine - take output Tensors from each adapter and form a combined output
      Tensor.  Outputs from different adapters for the same batch element are
      summed together, weighted by the provided "gates".
    The class is initialized with a "gates" Tensor, which specifies which
    batch elements go to which adapters, and the weights to use when combining
    the outputs.  Batch element b is sent to adapter e iff gates[b, e] != 0.
    The inputs and outputs are all two-dimensional [batch, depth].
    Caller is responsible for collapsing additional dimensions prior to
    calling this class and reshaping the output to the original shape.
    See common_layers.reshape_like().
    Example use:
    gates: a float32 `Tensor` with shape `[batch_size, num_adapters]`
    inputs: a float32 `Tensor` with shape `[batch_size, input_size]`
    adapters: a list of length `num_adapters` containing sub-networks.
    dispatcher = SparseDispatcher(num_adapters, gates)
    adapter_inputs = dispatcher.dispatch(inputs)
    adapter_outputs = [adapters[i](adapter_inputs[i]) for i in range(num_adapters)]
    outputs = dispatcher.combine(adapter_outputs)
    The preceding code sets the output for a particular example b to:
    output[b] = Sum_i(gates[b, i] * adapters[i](inputs[b]))
    This class takes advantage of sparsity in the gate matrix by including in the
    `Tensor`s for adapter i only the batch elements for which `gates[b, i] > 0`.
    """

    def __init__(self, num_adapters, gates):
        """Create a SparseDispatcher."""

        self._gates = gates
        self._num_adapters = num_adapters
        # sort adapters
        sorted_adapters, index_sorted_adapters = torch.nonzero(gates).sort(0)
        # drop indices
        _, self._adapter_index = sorted_adapters.split(1, dim=1)
        # get according batch index for each adapter
        self._batch_index = torch.nonzero(gates)[index_sorted_adapters[:, 1], 0]
        # calculate num samples that each adapter gets
        self._part_sizes = (gates > 0).sum(0).tolist()
        # expand gates to match with self._batch_index
        gates_exp = gates[self._batch_index.flatten()]
        self._nonzero_gates = torch.gather(gates_exp, 1, self._adapter_index)

    def dispatch(self, inp):
        """Create one input Tensor for each adapter.
        The `Tensor` for a adapter `i` contains the slices of `inp` corresponding
        to the batch elements `b` where `gates[b, i] > 0`.
        Args:
          inp: a `Tensor` of shape "[batch_size, <extra_input_dims>]`
        Returns:
          a list of `num_adapters` `Tensor`s with shapes
            `[adapter_batch_size_i, <extra_input_dims>]`.
        """

        # assigns samples to adapters whose gate is nonzero

        # expand according to batch index so we can just split by _part_sizes
        inp_exp = inp[self._batch_index].squeeze(1)
        return torch.split(inp_exp, self._part_sizes, dim=0)

    def combine(self, adapter_out, multiply_by_gates=True):
        """Sum together the adapter output, weighted by the gates.
        The slice corresponding to a particular batch element `b` is computed
        as the sum over all adapters `i` of the adapter output, weighted by the
        corresponding gate values.  If `multiply_by_gates` is set to False, the
        gate values are ignored.
        Args:
          adapter_out: a list of `num_adapters` `Tensor`s, each with shape
            `[adapter_batch_size_i, <extra_output_dims>]`.
          multiply_by_gates: a boolean
        Returns:
          a `Tensor` with shape `[batch_size, <extra_output_dims>]`.
        """
        # apply exp to adapter outputs, so we are not longer in log space
        stitched = torch.cat(adapter_out, 0).exp()

        if multiply_by_gates:
            stitched = stitched.mul(self._nonzero_gates)
        zeros = torch.zeros(self._gates.size(0), adapter_out[-1].size(1), requires_grad=True, device=stitched.device)
        # combine samples that have been processed by the same k adapters
        combined = zeros.index_add(0, self._batch_index, stitched.float())
        # add eps to all zero values in order to avoid nans when going back to log space
        combined[combined == 0] = np.finfo(float).eps
        # back to log space
        return combined.log()

    def adapter_to_gates(self):
        """Gate values corresponding to the examples in the per-adapter `Tensor`s.
        Returns:
          a list of `num_adapters` one-dimensional `Tensor`s with type `tf.float32`
              and shapes `[adapter_batch_size_i]`
        """
        # split nonzero gates for each adapter
        return torch.split(self._nonzero_gates, self._part_sizes, dim=0)




class MoE(nn.Module):

    """Call a Sparsely gated mixture of adapters layer with 1-layer Feed-Forward networks as adapters.
    Args:
    input_size: integer - size of the input
    output_size: integer - size of the input
    num_adapters: an integer - number of adapters
    hidden_size: an integer - hidden size of the adapters
    noisy_gating: a boolean
    k: an integer - how many adapters to use for each batch element
    """

    def __init__(self, extract_feature_size ,input_size,  adapter_model ,num_adapters, noisy_gating=True, k=1, device=None):
        super(MoE, self).__init__()
        self.noisy_gating = noisy_gating
        self.num_adapters = num_adapters
        self.input_size = input_size
        self.device = device
        self.k = k
        # instantiate adapters
        self.adapters = nn.ModuleList([copy.deepcopy(adapter_model) for i in range(self.num_adapters)])
        # self.gating = nn.ParameterDict({"w_gate":nn.Parameter(torch.zeros(extract_feature_size, num_adapters), requires_grad=True), 
        #                              "w_noise":nn.Parameter(torch.zeros(extract_feature_size, num_adapters), requires_grad=True)})
        # torch.nn.init.normal_(self.gating['w_gate'], mean=0, std=0.01) 
        # torch.nn.init.normal_(self.gating['w_noise'], mean=0, std=0.01) 
        self.gating = nn.Sequential(nn.Linear(extract_feature_size, 512), nn.ReLU(), nn.Dropout(0.2), nn.Linear(512, num_adapters))
        self.softplus = nn.Softplus()
        self.softmax = nn.Softmax(1)
        self.register_buffer("mean", torch.tensor([0.0]))
        self.register_buffer("std", torch.tensor([1.0]))
        assert(self.k <= self.num_adapters)

    
    

    def cv_squared(self, x):
        """The squared coefficient of variation of a sample.
        Useful as a loss to encourage a positive distribution to be more uniform.
        Epsilons added for numerical stability.
        Returns 0 for an empty Tensor.
        Args:
        x: a `Tensor`.
        Returns:
        a `Scalar`.
        """
        eps = 1e-10
        # if only num_adapters = 1

        if x.shape[0] == 1:
            return torch.tensor([0], device=x.device, dtype=x.dtype)
        return x.float().var() / (x.float().mean()**2 + eps)

    def _gates_to_load(self, gates):
        """Compute the true load per adapter, given the gates.
        The load is the number of examples for which the corresponding gate is >0.
        Args:
        gates: a `Tensor` of shape [batch_size, n]
        Returns:
        a float32 `Tensor` of shape [n]
        """
        return (gates > 0).sum(0)

    def _prob_in_top_k(self, clean_values, noisy_values, noise_stddev, noisy_top_values):
        """Helper function to NoisyTopKGating.
        Computes the probability that value is in top k, given different random noise.
        This gives us a way of backpropagating from a loss that balances the number
        of times each adapter is in the top k adapters per example.
        In the case of no noise, pass in None for noise_stddev, and the result will
        not be differentiable.
        Args:
        clean_values: a `Tensor` of shape [batch, n].
        noisy_values: a `Tensor` of shape [batch, n].  Equal to clean values plus
          normally distributed noise with standard deviation noise_stddev.
        noise_stddev: a `Tensor` of shape [batch, n], or None
        noisy_top_values: a `Tensor` of shape [batch, m].
           "values" Output of tf.top_k(noisy_top_values, m).  m >= k+1
        Returns:
        a `Tensor` of shape [batch, n].
        """
        batch = clean_values.size(0)
        m = noisy_top_values.size(1)
        top_values_flat = noisy_top_values.flatten()

        threshold_positions_if_in = torch.arange(batch, device=clean_values.device) * m + self.k
        threshold_if_in = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_in), 1)
        is_in = torch.gt(noisy_values, threshold_if_in)
        threshold_positions_if_out = threshold_positions_if_in - 1
        threshold_if_out = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_out), 1)
        # is each value currently in the top k.
        normal = Normal(self.mean, self.std)
        prob_if_in = normal.cdf((clean_values - threshold_if_in)/noise_stddev)
        prob_if_out = normal.cdf((clean_values - threshold_if_out)/noise_stddev)
        prob = torch.where(is_in, prob_if_in, prob_if_out)
        return prob

    def noisy_top_k_gating(self, x, train, noise_epsilon=1e-2, temperature=5):
        """Noisy top-k gating.
          See paper: https://arxiv.org/abs/1701.06538.
          Args:
            x: input Tensor with shape [batch_size, input_size]
            train: a boolean - we only add noise at training time.
            noise_epsilon: a float
          Returns:
            gates: a Tensor with shape [batch_size, num_adapters]
            load: a Tensor with shape [num_adapters]
        """
        # clean_logits = x @ self.gating["w_gate"]

        # if self.noisy_gating and train:
        #     raw_noise_stddev = x @ self.gating['w_noise'] 
        #     noise_stddev = ((self.softplus(raw_noise_stddev) + noise_epsilon))
        #     noisy_logits = clean_logits + (torch.randn_like(clean_logits) * noise_stddev)
        #     logits = noisy_logits
        # else:
        #     logits = clean_logits
        logits = self.gating(x)

        # calculate topk + 1 that will be needed for the noisy gates
        # top_logits, top_indices = logits.topk(min(self.k + 1, self.num_adapters), dim=1)
        top_logits, top_indices = logits.topk(min(self.k + 1, self.num_adapters), dim=1)
        top_k_logits = top_logits[:, :self.k]
        top_k_indices = top_indices[:, :self.k]
        top_k_gates = self.softmax(top_k_logits/temperature)

        zeros = torch.zeros_like(logits, requires_grad=True)
        gates = zeros.scatter(1, top_k_indices, top_k_gates)
        # self.noisy_gating = False
        # if self.noisy_gating and self.k < self.num_adapters and train:
        load = self._gates_to_load(gates)
        #     load = (self._prob_in_top_k(clean_logits, noisy_logits, noise_stddev, top_logits)).sum(0)
        # else:
        #     load = self._gates_to_load(gates)
        # print(logits[0])

        return gates, load, logits

    def forward(self, x, x2=None, loss_coef=1e-2, train_gate=False):
        """Args:
        x: tensor shape [batch_size, input_size]
        train: a boolean scalar.
        loss_coef: a scalar - multiplier on load-balancing losses

        Returns:
        y: a tensor with shape [batch_size, output_size].
        extra_training_loss: a scalar.  This should be added into the overall
        training loss of the model.  The backpropagation of this loss
        encourages all adapters to be approximately equally used across a batch.
        """
        gates, load, logits = self.noisy_top_k_gating(x, self.training)
        # print(gates, load)
        # calculate importance loss
        importance = gates.sum(0)
        #
        loss = self.cv_squared(importance) + self.cv_squared(load)
        loss *= loss_coef
        if train_gate:
            return None, loss, logits
        else:
          dispatcher = SparseDispatcher(self.num_adapters, gates)
          adapter_inputs = dispatcher.dispatch(x2)
          gates = dispatcher.adapter_to_gates()
          adapter_outputs = [self.adapters[i](adapter_inputs[i]) for i in range(self.num_adapters)]
          y = dispatcher.combine(adapter_outputs)
          return y, loss, logits
