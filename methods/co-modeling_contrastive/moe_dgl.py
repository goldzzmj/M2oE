# Note by Haotao Wang:
# Adapted form https://raw.githubusercontent.com/davidmrau/mixture-of-experts/master/moe.py

# Sparsely-Gated Mixture-of-Experts Layers.
# See "Outrageously Large Neural Networks"
# https://arxiv.org/abs/1701.06538
#
# Author: David Rau
#
# The code is based on the TensorFlow implementation:
# https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/utils/expert_utils.py


import torch
import torch.nn as nn
from torch.distributions.normal import Normal
import numpy as np
from torch_geometric.nn import GCNConv
from dgl.nn.pytorch import GraphConv
import torch.nn.functional as F

class SparseDispatcher(object):
    """Helper for implementing a mixture of experts.
    The purpose of this class is to create input minibatches for the
    experts and to combine the results of the experts to form a unified
    output tensor.
    There are two functions:
    dispatch - take an input Tensor and create input Tensors for each expert.
    combine - take output Tensors from each expert and form a combined output
      Tensor.  Outputs from different experts for the same batch element are
      summed together, weighted by the provided "gates".
    The class is initialized with a "gates" Tensor, which specifies which
    batch elements go to which experts, and the weights to use when combining
    the outputs.  Batch element b is sent to expert e iff gates[b, e] != 0.
    The inputs and outputs are all two-dimensional [batch, depth].
    Caller is responsible for collapsing additional dimensions prior to
    calling this class and reshaping the output to the original shape.
    See common_layers.reshape_like().
    Example use:
    gates: a float32 `Tensor` with shape `[batch_size, num_experts]`
    inputs: a float32 `Tensor` with shape `[batch_size, input_size]`
    experts: a list of length `num_experts` containing sub-networks.
    dispatcher = SparseDispatcher(num_experts, gates)
    expert_inputs = dispatcher.dispatch(inputs)
    expert_outputs = [experts[i](expert_inputs[i]) for i in range(num_experts)]
    outputs = dispatcher.combine(expert_outputs)
    The preceding code sets the output for a particular example b to:
    output[b] = Sum_i(gates[b, i] * experts[i](inputs[b]))
    This class takes advantage of sparsity in the gate matrix by including in the
    `Tensor`s for expert i only the batch elements for which `gates[b, i] > 0`.
    """

    def __init__(self, num_experts, gates):
        """Create a SparseDispatcher."""

        self._gates = gates
        self._num_experts = num_experts
        # sort experts
        sorted_experts, index_sorted_experts = torch.nonzero(gates).sort(0)
        # drop indices
        _, self._expert_index = sorted_experts.split(1, dim=1)
        # get according batch index for each expert
        self._batch_index = torch.nonzero(gates)[index_sorted_experts[:, 1], 0]
        # calculate num samples that each expert gets
        self._part_sizes = (gates > 0).sum(0).tolist()
        # expand gates to match with self._batch_index
        gates_exp = gates[self._batch_index.flatten()]
        self._nonzero_gates = torch.gather(gates_exp, 1, self._expert_index)


    def dispatch(self, inp, edge_index, edge_attr):
        """Create one input Tensor for each expert.
        The `Tensor` for a expert `i` contains the slices of `inp` corresponding
        to the batch elements `b` where `gates[b, i] > 0`.
        Args:
          inp: a `Tensor` of shape "[batch_size, <extra_input_dims>]`
        Returns:
          a list of `num_experts` `Tensor`s with shapes
            `[expert_batch_size_i, <extra_input_dims>]`.
        """

        # Note by Haotao:
        # self._batch_index: shape=(N_batch). The re-order indices from 0 to N_batch-1.
        # inp_exp: shape=inp.shape. The input Tensor re-ordered by self._batch_index along the batch dimension.
        # self._part_sizes: shape=(N_experts), sum=N_batch. self._part_sizes[i] is the number of samples routed towards expert[i].
        # return value: list [Tensor with shape[0]=self._part_sizes[i] for i in range(N_experts)]

        # assigns samples to experts whose gate is nonzero

        # expand according to batch index so we can just split by _part_sizes
        inp_exp = inp[self._batch_index].squeeze(1)
        edge_index_exp = edge_index[:,self._batch_index]
        edge_attr_exp = edge_attr[self._batch_index]
        return torch.split(inp_exp, self._part_sizes, dim=0), torch.split(edge_index_exp, self._part_sizes, dim=1), torch.split(edge_attr_exp, self._part_sizes, dim=0)

    def combine(self, expert_out, multiply_by_gates=True):
        """Sum together the expert output, weighted by the gates.
        The slice corresponding to a particular batch element `b` is computed
        as the sum over all experts `i` of the expert output, weighted by the
        corresponding gate values.  If `multiply_by_gates` is set to False, the
        gate values are ignored.
        Args:
          expert_out: a list of `num_experts` `Tensor`s, each with shape
            `[expert_batch_size_i, <extra_output_dims>]`.
          multiply_by_gates: a boolean
        Returns:
          a `Tensor` with shape `[batch_size, <extra_output_dims>]`.
        """
        # apply exp to expert outputs, so we are not longer in log space
        stitched = torch.cat(expert_out, 0).exp()

        if multiply_by_gates:
            stitched = stitched.mul(self._nonzero_gates)
        zeros = torch.zeros(self._gates.size(0), expert_out[-1].size(1), requires_grad=True, device=stitched.device)
        # combine samples that have been processed by the same k experts
        combined = zeros.index_add(0, self._batch_index, stitched.float())
        # add eps to all zero values in order to avoid nans when going back to log space
        combined[combined == 0] = np.finfo(float).eps
        # back to log space
        return combined.log()

    def expert_to_gates(self):
        """Gate values corresponding to the examples in the per-expert `Tensor`s.
        Returns:
          a list of `num_experts` one-dimensional `Tensor`s with type `tf.float32`
              and shapes `[expert_batch_size_i]`
        """
        # split nonzero gates for each expert
        return torch.split(self._nonzero_gates, self._part_sizes, dim=0)

class MoE(nn.Module):

    """Call a Sparsely gated mixture of experts layer with 1-layer Feed-Forward networks as experts.
    Args:
    input_size: integer - size of the input
    output_size: integer - size of the input
    num_experts: an integer - number of experts
    hidden_size: an integer - hidden size of the experts
    noisy_gating: a boolean
    k: an integer - how many experts to use for each batch element
    """

    def __init__(self, input_size, output_size, num_experts, experts_conv, experts_bn, noisy_gating=True, k=4, coef=1e-2, num_experts_1hop=None):
        super(MoE, self).__init__()
        self.noisy_gating = noisy_gating # True
        self.num_experts = num_experts # 8
        self.output_size = output_size # 256
        self.input_size = input_size # 256
        self.k = k # 4
        self.loss_coef = coef # 1
        if not num_experts_1hop:
            self.num_experts_1hop = num_experts # by default, all experts are hop-1 experts.
        else:
            assert num_experts_1hop <= num_experts
            self.num_experts_1hop = num_experts_1hop
        # instantiate experts
        # self.experts = nn.ModuleList([MLP(self.input_size, self.output_size, self.hidden_size) for i in range(self.num_experts)])
        self.experts_conv = experts_conv
        self.experts_bn = experts_bn
        self.w_gate = nn.Parameter(torch.zeros(input_size, num_experts), requires_grad=True)
        # self.w_gate = nn.Linear(input_size, num_experts)
        self.epsilon = 1e-6
        self.w_noise = nn.Parameter(torch.zeros(input_size, num_experts), requires_grad=True)

        self.softplus = nn.Softplus()
        self.softmax = nn.Softmax(1)
        self.register_buffer("mean", torch.tensor([0.0]))
        self.register_buffer("std", torch.tensor([1.0]))
        assert(self.k <= self.num_experts)

        self.head_2_dim = nn.Linear(self.input_size * num_experts, self.output_size)

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
        # if only num_experts = 1

        if x.shape[0] == 1:
            return torch.tensor([0], device=x.device, dtype=x.dtype)
        return x.float().var() / (x.float().mean()**2 + eps)

    def _gates_to_load(self, gates):
        """Compute the true load per expert, given the gates.
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
        of times each expert is in the top k experts per example.
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
        batch = clean_values.size(0) # 352
        m = noisy_top_values.size(1) # 5
        top_values_flat = noisy_top_values.flatten() # [352*5,]=[1760,]

        threshold_positions_if_in = torch.arange(batch, device=clean_values.device) * m + self.k # [352,]
        threshold_if_in = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_in), 1) # [352,1]
        is_in = torch.gt(noisy_values, threshold_if_in) # [352,8]
        threshold_positions_if_out = threshold_positions_if_in - 1
        threshold_if_out = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_out), 1) # [352,1]
        # is each value currently in the top k.
        normal = Normal(self.mean, self.std)
        prob_if_in = normal.cdf((clean_values - threshold_if_in)/noise_stddev) # [352,8]
        prob_if_out = normal.cdf((clean_values - threshold_if_out)/noise_stddev) # [352,8]
        prob = torch.where(is_in, prob_if_in, prob_if_out) # [352,8]
        return prob

    def noisy_top_k_gating(self, x, train, noise_epsilon=1e-2):
        """Noisy top-k gating.
          See paper: https://arxiv.org/abs/1701.06538.
          Args:
            x: input Tensor with shape [batch_size, input_size]
            train: a boolean - we only add noise at training time.
            noise_epsilon: a float
          Returns:
            gates: a Tensor with shape [batch_size, num_experts]
            load: a Tensor with shape [num_experts]
        """
        # self.w_gate、self.w_noise: [256,8]
        clean_logits = x @ self.w_gate # [node_num,8]
        if self.noisy_gating and train:
            raw_noise_stddev = x @ self.w_noise # [352,8]
            noise_stddev = ((self.softplus(raw_noise_stddev) + noise_epsilon))
            noisy_logits = clean_logits + (torch.randn_like(clean_logits) * noise_stddev)
            logits = noisy_logits
        else:
            logits = clean_logits

        # calculate topk + 1 that will be needed for the noisy gates
        top_logits, top_indices = logits.topk(min(self.k + 1, self.num_experts), dim=1) # [352,5]
        top_k_logits = top_logits[:, :self.k] # [352,4]
        top_k_indices = top_indices[:, :self.k]
        top_k_gates = self.softmax(top_k_logits)

        zeros = torch.zeros_like(logits, requires_grad=True) # [352,8]
        gates = zeros.scatter(1, top_k_indices, top_k_gates) # [352,8]

        if self.noisy_gating and self.k < self.num_experts and train:
            load = (self._prob_in_top_k(clean_logits, noisy_logits, noise_stddev, top_logits)).sum(0) # [8,]
        else:
            load = self._gates_to_load(gates)
        return gates, load

    def forward(self, g_dgl, x, edge_index_2hop=None, edge_attr_2hop=None):
        """Args:
        x: tensor shape [batch_size, input_size]
        train: a boolean scalar.
        loss_coef: a scalar - multiplier on load-balancing losses

        Returns:
        y: a tensor with shape [batch_size, output_size].
        extra_training_loss: a scalar.  This should be added into the overall
        training loss of the model.  The backpropagation of this loss
        encourages all experts to be approximately equally used across a batch.
        """
        # gates: [352,8], load:[8,]
        gates, load = self.noisy_top_k_gating(x, self.training)
        # calculate importance loss
        importance = gates.sum(0)
        #
        loss = self.cv_squared(importance) + self.cv_squared(load)
        loss *= self.loss_coef

        expert_outputs = []
        for i in range(self.num_experts):
            expert_i_output = self.experts_conv[i](g_dgl, x)  # [499,150]
            expert_i_output = self.experts_bn[i](expert_i_output)
            expert_outputs.append(expert_i_output)
        expert_outputs = torch.stack(expert_outputs, dim=1) # shape=[num_nodes, num_experts, d_feature]

        # gates: shape=[num_nodes, num_experts]
        y = gates.unsqueeze(dim=-1) * expert_outputs

        # concat all experts
        y = y.reshape((-1,
             y.size(1) * y.size(2)))

        y = self.head_2_dim(y)

        # y = y.mean(dim=1)

        # gate_scores = F.softmax(self.w_gate(x), dim=-1)
        #
        # top_k_scores, top_k_indices = gate_scores.topk(1, dim=-1) # [512,10,1]
        #
        # # Mask to enforce sparsity
        # mask = torch.zeros_like(gate_scores).scatter_(
        #     1, top_k_indices, 1
        # )
        #
        # # Combine gating scores with the mask
        # masked_gate_scores = gate_scores * mask
        #
        # # Denominators
        # denominators = (
        #         masked_gate_scores.sum(0, keepdim=True) + self.epsilon
        # )
        #
        # # Norm gate scores to sum to the capacity
        # gate_scores = (masked_gate_scores / denominators)
        #
        # importance = gate_scores.sum(0)
        # load = (gate_scores > 0).sum(0)
        #
        # loss = self.cv_squared(importance) + self.cv_squared(load)
        #
        # # gates: shape=[num_nodes, num_experts]
        # y = gate_scores.unsqueeze(dim=-1) * expert_outputs
        # y = y.sum(dim=1)

        return y, loss

if __name__ == '__main__':
    class MLP(nn.Module):
        def __init__(self, input_size, output_size, hidden_size):
            super(MLP, self).__init__()
            self.fc1 = nn.Linear(input_size, hidden_size)
            self.fc2 = nn.Linear(hidden_size, output_size)
            self.relu = nn.ReLU()
            self.soft = nn.Softmax(1)

        def forward(self, x):
            out = self.fc1(x)
            out = self.relu(out)
            out = self.fc2(out)
            out = self.soft(out)
            return out

    moe_model = MoE(10, 10, 3, [GraphConv(10,10), GraphConv(10,10), GraphConv(10,10)], [nn.ReLU(),nn.ReLU(),nn.ReLU()], k=1)
    x = torch.randn((7,10))
    edge_index, edge_attr = torch.randint(0,7,(2,8)), torch.randn(8,10)
    import dgl
    g_dgl = dgl.graph((edge_index[0], edge_index[1]))
    g_dgl.ndata['feat'] = x
    g_dgl = dgl.add_self_loop(g_dgl)
    print(g_dgl)
    h, loss = moe_model(x, g_dgl)
    print(h.shape)
    print(loss)