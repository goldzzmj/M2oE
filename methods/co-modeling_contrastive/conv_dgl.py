import torch
import torch.nn as nn

import dgl
import torch.nn.functional as F
# from ogb.graphproppred.mol_encoder import AtomEncoder,BondEncoder
from dgl.nn.pytorch import GraphConv, GINConv, SAGEConv
from torch.nn.functional import relu
from timm.models.layers import DropPath

# from ogb.utils.features import get_atom_feature_dims, get_bond_feature_dims
# full_bond_feature_dims = get_bond_feature_dims()

from moe_dgl import MoE


class AtomEncoder(torch.nn.Module):

    def __init__(self, args, emb_dim):
        super(AtomEncoder, self).__init__()

        self.args = args
        self.emb = torch.nn.Embedding(self.args.src_vocab_size_graph + 1, emb_dim)
        torch.nn.init.xavier_uniform_(self.emb.weight.data)

    def forward(self, x):
        x_embedding = self.emb(x)

        return x_embedding

class GNN_SpMoE_node(torch.nn.Module):
    """
    Output:
        node representations
    """

    def __init__(self, args, num_layer, emb_dim, num_experts=3, drop_ratio=0.5, JK="last", residual=False, gnn_type='gcn',
                 k=1, coef=1e-2, num_experts_1hop=None):
        '''
            emb_dim (int): node embedding dimensionality
            num_layer (int): number of GNN message passing layers
            JK: Jumping knowledge refers to "Representation Learning on Graphs with Jumping Knowledge Networks"
            k: k value for top-k sparse gating.
            num_experts: total number of experts in each layer.
            num_experts_1hop: number of hop-1 experts in each layer. The first num_experts_1hop are hop-1 experts. The rest num_experts-num_experts_1hop are hop-2 experts.
        '''

        super(GNN_SpMoE_node, self).__init__()
        self.args = args
        self.num_layer = num_layer
        self.num_experts = num_experts
        self.k = k
        self.drop_ratio = drop_ratio
        self.JK = JK
        ### add residual connection or not
        self.residual = residual

        if not num_experts_1hop:
            self.num_experts_1hop = num_experts  # by default, all experts are hop-1 experts.
        else:
            assert num_experts_1hop <= num_experts
            self.num_experts_1hop = num_experts_1hop

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        self.atom_encoder = AtomEncoder(args, emb_dim)

        self.lin = nn.Linear(emb_dim,emb_dim)

        # DropPath
        self.drop_path = DropPath(drop_ratio) if drop_ratio > 0. else nn.Identity()

        ###List of GNNs
        self.ffns = torch.nn.ModuleList()

        for layer in range(num_layer):
            convs_list = torch.nn.ModuleList()
            bn_list = torch.nn.ModuleList()
            for expert_idx in range(num_experts):
                if gnn_type == 'gin':
                    if expert_idx < self.num_experts_1hop:
                        convs_list.append(GINConv(self.lin,'max', activation=relu))
                elif gnn_type == 'gcn':
                    if expert_idx < self.num_experts_1hop:
                        convs_list.append(GraphConv(emb_dim, emb_dim))
                elif gnn_type == 'sage':
                    if expert_idx < self.num_experts_1hop:
                        convs_list.append(SAGEConv(emb_dim, emb_dim, self.args.GraphSAGE_aggregator))
                else:
                    raise ValueError('Undefined GNN type called {}'.format(gnn_type))

                bn_list.append(torch.nn.BatchNorm1d(emb_dim))
                # bn_list.append(nn.ReLU())

            ffn = MoE(input_size=emb_dim, output_size=emb_dim, num_experts=num_experts, experts_conv=convs_list,
                      experts_bn=bn_list,
                      k=k, coef=coef, num_experts_1hop=self.num_experts_1hop)

            self.ffns.append(ffn)

        # self.mix_fn = lambda h_expert_list: torch.mean(torch.stack(h_expert_list, dim=0), dim=0)

    def forward(self, g, x):
        # g = dgl.add_self_loop(g)
        # x, edge_index, edge_attr, batch = batched_data.x, batched_data.edge_index, batched_data.edge_attr, batched_data.batch

        ### computing input node embedding

        h_list = [self.atom_encoder(x)] # [(9701,256)]
        self.load_balance_loss = 0  # initialize load_balance_loss to 0 at the beginning of each forward pass.
        for layer in range(self.num_layer):
            # h:(9701,256)
            h, _layer_load_balance_loss = self.ffns[layer](g, h_list[layer])
            self.load_balance_loss += _layer_load_balance_loss

            if layer == self.num_layer - 1:
                # remove relu for the last layer
                # h = F.dropout(h, self.drop_ratio, training=self.training)
                # DropPath
                h = h + self.drop_path(F.relu(h))
                # #DropOut
                # h = F.dropout(F.relu(h), self.drop_ratio, training=self.training)
            else:
                # DropPath
                h = h + self.drop_path(F.relu(h))
                # #DropOut
                # h = F.dropout(F.relu(h), self.drop_ratio, training=self.training)

            if self.residual:
                h += h_list[layer]

            h_list.append(h)

        self.load_balance_loss /= self.num_layer

        ### Different implementations of Jk-concat
        if self.JK == "last":
            node_representation = h_list[-1]
        elif self.JK == "sum":
            node_representation = 0
            for layer in range(self.num_layer + 1):
                node_representation += h_list[layer]

        return node_representation


