# import unittest

import numpy as np
import torch
from torch.autograd import Function,Variable

# from nearest_embed import nearest_embed


# class NearestEmbedTest(unittest.TestCase):
#     def test_something(self):
#
#         emb = Variable(torch.eye(10, 10).double())
#         a = np.array(([1,0,0,0,0,0,0,0,0,0],
#                       [0,1,0,0,0,0,0,0,0,0]), dtype=np.double)
#         input = Variable(torch.from_numpy(a))
#         z_q = nearest_embed(input, emb, dim=1)
#         self.assertEqual(True, torch.equal(z_q.data, input.data))


# class NearestEmbed2dT(unittest.TestCase):
#
#     def test_single_embedding(self):
#         # inputs
#         emb = Variable(torch.eye(5, 7).float(), requires_grad=True)
#
#         a = np.array([[[0.9, 0.],
#                        [0., 0.],
#                        [0., 2.],
#                        [0., 0.],
#                        [0., 0.]],
#
#                       [[0., 0.7],
#                        [0., 0.],
#                        [0., 0.],
#                        [0.6, 0.],
#                        [0., 0.]]], dtype=np.float32)
#
#         # expected results
#         out = np.array([[[1., 0.],
#                          [0., 0.],
#                          [0., 1.],
#                          [0., 0.],
#                          [0., 0.]],
#
#                         [[0., 1.],
#                          [0., 0.],
#                          [0., 0.],
#                          [1., 0.],
#                          [0., 0.]]], dtype=np.float32)
#
#         grad_input = np.array([[[1., 0.],
#                                 [0., 0.],
#                                 [0., 1.],
#                                 [0., 0.],
#                                 [0., 0.]],
#
#                                [[0., 1.],
#                                 [0., 0.],
#                                 [0., 0.],
#                                 [1., 0.],
#                                 [0., 0.]]], dtype=np.float32)
#
#         grad_emb = np.array([[1., 0., 0., 0., 0., 0., 0.],
#                              [0., 0., 0., 0., 0., 0., 0.],
#                              [0., 0., 1., 0., 0., 0., 0.],
#                              [0., 0., 0., 1., 0., 0., 0.],
#                              [0., 0., 0., 0., 0., 0., 0.]], dtype=np.float32)
#
#         grad_input = torch.from_numpy(grad_input).float()
#         grad_emb = torch.from_numpy(grad_emb).float()
#
#         input = Variable(torch.from_numpy(a).float(), requires_grad=True)
#         z_q, _ = nearest_embed(input, emb)
#
#         (0.5 * z_q.pow(2)).sum().backward(retain_graph=True)
#         out = torch.from_numpy(out)
#
#         self.assertEqual(True, torch.equal(z_q.data, out))
#         self.assertEqual(True, torch.equal(input.grad.data, grad_input))
#         self.assertEqual(True, torch.equal(emb.grad.data, grad_emb))
class NearestEmbedFunc(Function):
    """
    Input:
    ------
    x - (batch_size, emb_dim, *)
        Last dimensions may be arbitrary
    emb - (emb_dim, num_emb)
    """
    @staticmethod
    def forward(ctx, input, emb):
        if input.size(1) != emb.size(0):
            raise RuntimeError('invalid argument: input.size(1) ({}) must be equal to emb.size(0) ({})'.
                               format(input.size(1), emb.size(0)))

        # save sizes for backward
        ctx.batch_size = input.size(0)
        ctx.num_latents = int(np.prod(np.array(input.size()[2:])))
        ctx.emb_dim = emb.size(0)
        ctx.num_emb = emb.size(1)
        ctx.input_type = type(input)
        ctx.dims = list(range(len(input.size())))   # [0, 1]

        # expand to be broadcast-able
        x_expanded = input.unsqueeze(-1)        # [batch_size, emb_dim, 1]
        num_arbitrary_dims = len(ctx.dims) - 2
        if num_arbitrary_dims:
            emb_expanded = emb.view(emb.shape[0], *([1] * num_arbitrary_dims), emb.shape[1])
        else:
            emb_expanded = emb

        # find nearest neighbors
        dist = torch.norm(x_expanded - emb_expanded, p=2, dim=1)
        _, argmin = dist.min(-1)
        shifted_shape = [input.shape[0], *list(input.shape[2:]) ,input.shape[1]]
        result = emb.t().index_select(0, argmin.view(-1)).view(shifted_shape).permute(0, ctx.dims[-1], *ctx.dims[1:-1])

        ctx.save_for_backward(argmin)
        return result.contiguous(), argmin

    @staticmethod
    def backward(ctx, grad_output, argmin=None):
        grad_input = grad_emb = None
        if ctx.needs_input_grad[0]:
            grad_input = grad_output

        if ctx.needs_input_grad[1]:
            argmin, = ctx.saved_tensors
            latent_indices = torch.arange(ctx.num_emb).type_as(argmin)
            idx_choices = (argmin.view(-1, 1) == latent_indices.view(1, -1)).type_as(grad_output.data)
            n_idx_choice = idx_choices.sum(0)
            n_idx_choice[n_idx_choice == 0] = 1
            idx_avg_choices = idx_choices / n_idx_choice
            grad_output = grad_output.permute(0, *ctx.dims[2:], 1).contiguous()
            grad_output = grad_output.view(ctx.batch_size * ctx.num_latents, ctx.emb_dim)
            grad_emb = torch.sum(grad_output.data.view(-1, ctx.emb_dim, 1) *
                                 idx_avg_choices.view(-1, 1, ctx.num_emb), 0)
        return grad_input, grad_emb, None, None


def nearest_embed(x, emb):
    return NearestEmbedFunc().apply(x, emb)

def single_embedding():
    # inputs
    emb = Variable(torch.eye(5, 7).float(), requires_grad=True)

    a = np.array([[[0.9, 0.],
                   [0., 0.],
                   [0., 2.],
                   [0., 0.],
                   [0., 0.]],

                  [[0., 0.7],
                   [0., 0.],
                   [0., 0.],
                   [0.6, 0.],
                   [0., 0.]]], dtype=np.float32)

    # expected results
    out = np.array([[[1., 0.],
                     [0., 0.],
                     [0., 1.],
                     [0., 0.],
                     [0., 0.]],

                    [[0., 1.],
                     [0., 0.],
                     [0., 0.],
                     [1., 0.],
                     [0., 0.]]], dtype=np.float32)

    grad_input = np.array([[[1., 0.],
                            [0., 0.],
                            [0., 1.],
                            [0., 0.],
                            [0., 0.]],

                           [[0., 1.],
                            [0., 0.],
                            [0., 0.],
                            [1., 0.],
                            [0., 0.]]], dtype=np.float32)

    grad_emb = np.array([[1., 0., 0., 0., 0., 0., 0.],
                         [0., 0., 0., 0., 0., 0., 0.],
                         [0., 0., 1., 0., 0., 0., 0.],
                         [0., 0., 0., 1., 0., 0., 0.],
                         [0., 0., 0., 0., 0., 0., 0.]], dtype=np.float32)

    grad_input = torch.from_numpy(grad_input).float()
    grad_emb = torch.from_numpy(grad_emb).float()

    input = Variable(torch.from_numpy(a).float(), requires_grad=True)
    z_q, _ = nearest_embed(input, emb)
    print(z_q.size())

    (0.5 * z_q.pow(2)).sum().backward(retain_graph=True)
    out = torch.from_numpy(out)

    print(out)
    print(out.size())
if __name__ == '__main__':
    single_embedding()