import math
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
import dgl
from dgl.nn.pytorch import GraphConv,GATConv,SAGEConv,GINConv
from typing import Optional
from conv_dgl import GNN_SpMoE_node
from dgl.nn.pytorch.glob import mean_nodes
from model import MultiHeadAttention,PositionalEncoding,FeedForward,CrossAttention,get_attn_pad_mask
from torch.nn.functional import relu
from utils_seq import get_seq_mask
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from loss import info_nce_loss
import time

class M2oE(nn.Module):
    def __init__(self, args, num_tokens, dim, hidden_dim, output_dim, src_len=10, max_nodes=40,
                 task_type='Regression', GraphSAGE_aggregator='lstm', num_experts=5, heads=4,
                 dim_head=64, depth=6, gnn_depth=1, capacity_factor = 1,
                 gnn_type='sage', residue=True, gnn_residue=True, mult=4, cross_residual=True, cal_infoloss=False):
        super(M2oE, self).__init__()
        self.num_tokens = num_tokens
        self.dim = dim
        self.heads = heads
        self.dim_head = dim_head
        self.num_experts = num_experts
        self.fc_layers = depth # 6
        self.src_len = src_len
        self.task_type = task_type
        self.cal_infoloss = cal_infoloss

        # Sequence Embedding
        self.Seq_embedding = nn.Embedding(num_tokens, dim)
        self.pos_emb = PositionalEncoding(dim)

        # Graph nodes Embedding
        self.graph_embedding = AtomEncoder(args, dim)
        self.pos_emb_gra = PositionalEncoding(dim)

        # GTMoE block repeat numbers: depth
        self.gtmoe = nn.ModuleList([])
        for _ in range(depth):
            self.gtmoe.append(
                GTMoE(
                    num_experts,
                    dim,
                    hidden_dim,
                    output_dim,
                    heads,
                    dim_head,
                    gnn_depth,
                    capacity_factor,
                    gnn_type,
                    GraphSAGE_aggregator,
                    residue,
                    gnn_residue,
                    mult,
                    cross_residual
                )
            )

        # Decoder
        # Sequence decoder
        self.seq_decoder = nn.ModuleList([])
        for _ in range(self.fc_layers):
            self.seq_decoder.append(
                nn.Linear(self.dim, self.dim),
            )
            self.seq_decoder.append(
                nn.ReLU(),
            )

        self.fc_seq = nn.Linear(self.dim * self.src_len, self.dim)
        self.fc_gra = nn.Linear(self.dim * 139, self.dim)
        # self.fc_seq_gra = nn.Linear(self.dim * (max_nodes + self.src_len), self.dim)

        if self.task_type == 'Classification':
            self.predictor = nn.Linear(self.dim, 2)
        elif self.task_type == 'Regression':
            self.predictor = nn.Linear(self.dim, 1)
        self.seq_decoder.append(self.predictor)

        # graph decoder
        self.gra_decoder = nn.ModuleList([])
        for _ in range(self.fc_layers):
            self.gra_decoder.append(
                nn.Linear(self.dim, self.dim),
            )
            self.gra_decoder.append(
                nn.ReLU(),
            )
        self.gra_decoder.append(self.predictor)

        # combine Graph and Sequence decoder(MLP)
        self.seq_gra_decoder = nn.ModuleList([])
        for i in range(self.fc_layers):
            self.seq_gra_decoder.append(
                nn.Linear(self.dim, self.dim),
            )
            self.seq_gra_decoder.append(
                nn.ReLU(),
            )
        self.seq_gra_decoder.append(self.predictor)

        # combine Graph and Sequence decoder(Transformer block)
        self.fusion = FusionNet(self.dim)
        # self.fusion = SelfAttentionFusion(self.dim, 4, depth, 'avg')
        # self.fusion = FusionNet(self.dim)
        # self.fusion = nn.Bilinear(self.dim, self.dim, self.dim)
        
        # self.seq_gra_decoder = nn.ModuleList([])
        # for i in range(self.fc_layers):
        #     self.seq_gra_decoder.append(
        #             TransformerBlock(dim, heads, dim_head, mult),
        #         )
        # self.seq_gra_decoder.append(self.predictor)

        # self.seq_gra_decoder = nn.Sequential(*[TransformerBlock(dim, heads, dim_head, mult) for i in range(self.fc_layers)])

        # self.seq_gra_decoder = nn.ModuleList([])
        # for i in range(self.fc_layers):
        #     self.seq_gra_decoder.append(LMSA(dim, heads, dim_head))
        #     self.seq_gra_decoder.append(nn.LayerNorm(dim))
        #     self.seq_gra_decoder.append(FeedForward(dim, dim, mult))

        # classifier Learnable Paramater
        self.alpha = nn.Parameter(torch.tensor(0.8, requires_grad=True)) 


    def forward(self, x_seq, g, seq_pad_len):
        """
        :param x_seq: (batch, src_len)
        :param g: DGLGraph use dgl.batch batch size subgraph
        :param seq_pad_len: List, (batch,)
        :return: y
        """

        batch_size = x_seq.size(0)
        x_seq = self.Seq_embedding(x_seq) # (batch, src_len, dim)
        x_seq = self.pos_emb(x_seq.transpose(0, 1)).transpose(0, 1)
        x = g.ndata['feat'] # (all_num_nodes, )
        # mask_gra = torch.reshape(x.eq(15),(batch_size,-1))
        x = torch.reshape(x,(batch_size, -1)) # (batch, num_nodes)
        x_gra = self.graph_embedding(x) # (batch, num_nodes, dim)
        # x_gra = x_gra*mask_gra.unsqueeze(-1)
        # x_gra = self.pos_emb_gra(x_gra.transpose(0, 1)).transpose(0, 1)
        g.ndata['h'] = torch.reshape(x_gra, (-1, self.dim)) # (all_num_nodes, )


        loss = 0
        for layer in self.gtmoe:
            # x_seq:(batch, n, dim), x_gra:(batch, num_nodes, dim)
            if self.cal_infoloss:
                x_seq, x_gra, loss_seq_moe, loss_gra_moe = layer(x_seq, g, seq_pad_len)
                loss += layer.get_infoloss()
            else:
                x_seq, x_gra, loss_seq_moe, loss_gra_moe = layer(x_seq, g, seq_pad_len)
            loss += loss_seq_moe + loss_gra_moe
            g.ndata['h'] = torch.reshape(x_gra,(-1,self.dim)) # (batch*num_node, dim)

        x_seq = torch.reshape(x_seq, (x_seq.shape[0], -1)) # (batch, dim*num_node)
        x_seq = self.fc_seq(x_seq) # (batch, dim)
        x_gra = torch.reshape(x_gra, (batch_size, -1)) # (batch, dim*num_node)
        # x_gra = dgl.mean_nodes(g,'h')
        x_gra = dgl.sum_nodes(g,'h')
        # x_gra = self.fc_gra(x_gra) # (batch, dim)

        # loss_infonce = torch.nn.CrossEntropyLoss().to('cuda')
        # hid_pairs = torch.concat((x_seq, x_gra),dim=0)
        # logits, cont_labels = info_nce_loss(None, hid_pairs)
        # infoloss = 1e-4 * loss_infonce(logits, cont_labels)
        # loss += infoloss


        # fusion layer
        x_seq_gra = self.fusion(x_seq, x_gra)
        
        # # decoder block(Transformer)
        # for i,layer in enumerate(self.seq_gra_decoder):
        #     if i % 3 == 0:
        #         res_x = x_seq_gra
        #         x_seq_gra = layer(x_seq_gra,None) + res_x
        #     elif i % 3 == 1:
        #         res_x = x_seq_gra
        #         x_seq_gra = layer(x_seq_gra)
        #     elif i % 2 == 2:
        #         x_seq_gra = layer(x_seq_gra) + res_x

        # # x_seq_gra = self.seq_gra_decoder(x_seq_gra, None)
        # x_seq_gra = torch.reshape(x_seq_gra, (x_seq_gra.shape[0], -1)) # (batch, dim*num_node)
        # x_seq_gra = self.fc_seq_gra(x_seq_gra) # (batch, dim)
        # y = self.predictor(x_seq_gra)
        
        # for depth in range(len(self.seq_gra_decoder)):
        #     if depth == len(self.seq_gra_decoder) - 1:
        #         x_seq_gra = torch.reshape(x_seq_gra, (x_seq_gra.shape[0], -1)) # (batch, dim*num_node)
        #         x_seq_gra = self.fc_seq_gra(x_seq_gra) # (batch, dim)
        #         y = self.seq_gra_decoder[depth](x_seq_gra)
        #     x_seq_gra = self.seq_gra_decoder[depth](x_seq_gra, None)

        
        # decoder block(MLP)
        for depth in range(len(self.seq_gra_decoder)):
            if depth == len(self.seq_gra_decoder) - 1:
                y = self.seq_gra_decoder[depth](x_seq_gra)
            x_seq_gra = self.seq_gra_decoder[depth](x_seq_gra)


        if self.task_type == 'Classification':
            return F.log_softmax(y, dim=1), loss
        elif self.task_type == 'Regression':
            return y, loss

        # # parallel decoder
        # for depth in range(len(self.seq_decoder)):
        #     if depth == len(self.seq_decoder) - 1:
        #         y_seq = self.seq_decoder[depth](x_seq)
        #     x_seq = self.seq_decoder[depth](x_seq)

        # for depth in range(len(self.gra_decoder)):
        #     if depth == len(self.gra_decoder) - 1:
        #         y_gra = self.gra_decoder[depth](x_gra)
        #     x_gra = self.gra_decoder[depth](x_gra)

        # if self.task_type == 'Classification':
        #     return F.log_softmax(self.alpha * y_seq + (1 - self.alpha) * y_gra, dim=1), loss
        # elif self.task_type == 'Regression':
        #     return self.alpha * y_seq + (1 - self.alpha) * y_gra, loss






class GTMoE(nn.Module):
    def __init__(self, num_experts, dim, hidden_dim, output_dim, heads, dim_head, gnn_depth=1, capacity_factor=1, gnn_type='sage', GraphSAGE_aggregator='lstm', residue=True, gnn_residue=True, mult=4, cross_residual=True):
        super(GTMoE, self).__init__()
        # Layer Norm + Multi-Head Self Attention
        self.att_block = LMSA(dim, heads, dim_head)
        # self.att_block = nn.TransforsmerEncoderLayer(dim, 4)

        # GNN block
        self.gnn_block = Gnn_block(gnn_type, dim, gnn_depth, GraphSAGE_aggregator, gnn_residue)

        # cross Attention Block
        self.gate = GaT(dim, hidden_dim, num_experts, residue, mult, cross_residual)

        # Sequence and Graph, mixture of experts
        self.seq_moe = SwitchMoE(dim, hidden_dim, output_dim, num_experts, capacity_factor, mult)
        self.gra_moe = SwitchMoE(dim, hidden_dim, output_dim, num_experts, capacity_factor, mult)

        # self.ffn_seq = [FeedForward(dim, dim, mult).to('cuda') for i in range(num_experts)]
        # self.ffn_gra = [FeedForward(dim, dim, mult).to('cuda') for i in range(num_experts)]

        # self.fc_seq = nn.Linear(dim * num_experts, dim)
        # self.fc_gra = nn.Linear(dim * num_experts, dim)


    def forward(self, x_seq, g, seq_pad_len):
        # Sequence Model
        pad_mask = get_seq_mask(x_seq, seq_pad_len) # （512， 10）
        batch_size = pad_mask.size(0)
        len_ = pad_mask.size(1) # 10

        # pad_mask = pad_mask.transpose(0,1)
        # print(pad_mask)
        # x_seq = self.att_block(x_seq, src_key_padding_mask=pad_mask.to('cuda')) # (512,10,64)
        # print(x_seq)
        # x_seq[torch.isnan(x_seq)] = 0
        
        
        pad_mask = pad_mask.unsqueeze(1).expand(batch_size, len_, len_)
        # Trans_cos_time = time.time()
        x_seq = self.att_block(x_seq, pad_mask.to('cuda')) # (512,10,64)
        # print(f'Trans_cos_time:{time.time() - Trans_cos_time} s')

        # Graph Model
        # gnn_cos_time = time.time()
        _ = self.gnn_block(g) # (20480,64)
        # print(f'gnn_cos_time:{time.time() - gnn_cos_time} s')

        # Cross Attention
        # CA_cos_time = time.time()
        x_seq, x_gra = self.gate(x_seq, g, seq_pad_len)
        # print(f'CA_cos_time:{time.time() - CA_cos_time} s')

        self.x_seq, self.x_gra = torch.sum(x_seq,dim=1),  torch.sum(x_gra,dim=1)

        # # MoE
        # x_seq_moe, loss_seq_moe = self.seq_moe(x_seq)
        # x_gra_moe, loss_gra_moe = self.gra_moe(x_gra)

        loss_gra_moe = 0
        loss_seq_moe = 0

        # x_seq_moe = []
        # x_gra_moe = []
        # for expert in self.ffn_seq:
        #     x_seq_moe.append(expert(x_seq))

        # for expert in self.ffn_gra:
        #     x_gra_moe.append(expert(x_gra))

        # x_seq_moe = torch.stack(x_seq_moe,-1)
        # x_gra_moe = torch.stack(x_gra_moe,-1)

        
        # x_seq = self.fc_seq(x_seq_moe.reshape(batch_size, len_, -1))
        # x_gra = self.fc_gra(x_gra_moe.reshape(batch_size, 40, -1))

        return x_seq, x_gra, loss_seq_moe, loss_gra_moe


    def get_infoloss(self):
        loss_infonce = torch.nn.CrossEntropyLoss().to('cuda')
        hid_pairs = torch.concat((self.x_seq, self.x_gra),dim=0)
        logits, cont_labels = info_nce_loss(None, hid_pairs)
        loss = 1e-4 * loss_infonce(logits, cont_labels)

        return loss




# ----------------Cross Model----------------------
# Transformer fusion
class SelfAttentionFusion(nn.Module):
    def __init__(self, d_model=64, nhead=3, num_layers=3, pool='avg'):
        super(SelfAttentionFusion, self).__init__()
        encoder_layers = TransformerEncoderLayer(d_model, nhead)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers)
        self.pool = pool
    
    def forward(self, features_text, features_graph):
        features_text = features_text.unsqueeze(1) # (batch, 1, d)
        features_graph = features_graph.unsqueeze(1) # (batch, 1, d)
        
        sequence = torch.cat([features_text, features_graph], dim=1) # (batch, 2, d)
        #print(f"sequence.shape : {sequence.shape}")

        output = self.transformer_encoder(sequence) # (batch, 2, d)

        if self.pool == 'avg':
            features = F.avg_pool1d(output, output.shape[1]) # (batch, 2, d//2)
        else:
            features = F.max_pool1d(output, output.shape[1])

        features = torch.reshape(features,(features.size(0), -1))
        
        # if self.pool == 'avg':
        #     features = torch.mean(output, dim=1) # (batch, d)
        # else:
        #     features = torch.sum(output, dim=1)

        return features

# weighted sum
class FusionNet(nn.Module):
    def __init__(self, input_size):
        super(FusionNet, self).__init__()
        self.fc = nn.Linear(input_size * 2, 1)
        self.weights = 0.5
    def forward(self, features_text, features_graph):
        combined = torch.cat([features_text, features_graph], dim=1)  # (batch, 2d)
        self.weights = torch.sigmoid(self.fc(combined))  # (batch, 1)
        fused_embedding = self.weights * features_text + (1 - self.weights) * features_graph # (batch, n + m, d)
        return fused_embedding

# concat
class Fusion_Embed(nn.Module):
    def __init__(self, embed_dim, bias=False):
        super(Fusion_Embed, self).__init__()

        self.fusion_proj = nn.Linear(embed_dim * 2, embed_dim, bias=bias)

    def forward(self, x_A, x_B):
        x = torch.concat([x_A, x_B], dim=1)  # (batch, 2d)
        x = self.fusion_proj(x)  # (batch, d)
        return x

class SwitchGate(nn.Module):
    """
    SwitchGate module for MoE (Mixture of Experts) model.

    Args:
        dim (int): Input dimension.
        num_experts (int): Number of experts.
        capacity_factor (float, optional): Capacity factor for sparsity. Defaults to 1.0.
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.
    """

    def __init__(
            self,
            dim,
            num_experts: int,
            capacity_factor: float = 1.0,
            epsilon: float = 1e-6,
            *args,
            **kwargs,
    ):
        super().__init__()
        self.dim = dim
        self.num_experts = num_experts
        self.capacity_factor = capacity_factor
        self.epsilon = epsilon
        self.w_gate = nn.Linear(dim, num_experts)

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

    def forward(self, x: Tensor):
        """
        Forward pass of the SwitchGate module.

        Args:
            x (Tensor): Input tensor. [batch,seq,dim]=[512,10,64]

        Returns:
            Tensor: Gate scores.
        """
        # Compute gate scores
        batch = x.size(0) # 512
        N = x.size(1) # 10 or 40
        # x = x.reshape((batch*N,-1)) # (5120,64)
        gate_scores = F.softmax(self.w_gate(x), dim=-1) # [batch,seq,num_experts]=[512,10,5]

        # Determine the top-1 expert for each token
        capacity = int(self.capacity_factor * x.size(0))

        top_k_scores, top_k_indices = gate_scores.topk(1, dim=-1) # [512,10,1]

        # Mask to enforce sparsity
        mask = torch.zeros_like(gate_scores).scatter_(
            1, top_k_indices, 1
        )

        # Combine gating scores with the mask
        masked_gate_scores = gate_scores * mask

        # Denominators
        denominators = (
                masked_gate_scores.sum(0, keepdim=True) + self.epsilon
        )

        # Norm gate scores to sum to the capacity
        gate_scores = (masked_gate_scores / denominators) * capacity

        importance = gate_scores.sum(0)
        load = (gate_scores > 0).sum(0)

        loss = self.cv_squared(importance) + self.cv_squared(load)

        # gate_scores = gate_scores.reshape((batch, N, -1))

        return gate_scores, loss


class SwitchMoE(nn.Module):
    """
    A module that implements the Switched Mixture of Experts (MoE) architecture.

    Args:
        dim (int): The input dimension.
        hidden_dim (int): The hidden dimension of the feedforward network.
        output_dim (int): The output dimension.
        num_experts (int): The number of experts in the MoE.
        capacity_factor (float, optional): The capacity factor that controls the capacity of the MoE. Defaults to 1.0.
        mult (int, optional): The multiplier for the hidden dimension of the feedforward network. Defaults to 4.
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Attributes:
        dim (int): The input dimension.
        hidden_dim (int): The hidden dimension of the feedforward network.
        output_dim (int): The output dimension.
        num_experts (int): The number of experts in the MoE.
        capacity_factor (float): The capacity factor that controls the capacity of the MoE.
        mult (int): The multiplier for the hidden dimension of the feedforward network.
        experts (nn.ModuleList): The list of feedforward networks representing the experts.
        gate (SwitchGate): The switch gate module.

    """

    def __init__(
            self,
            dim: int,
            hidden_dim: int,
            output_dim: int,
            num_experts: int,
            capacity_factor: float = 1.0,
            mult: int = 4,
            *args,
            **kwargs,
    ):
        super().__init__()
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_experts = num_experts
        self.capacity_factor = capacity_factor
        self.mult = mult

        self.experts = nn.ModuleList(
            [
                FeedForward(dim, dim, mult)
                for _ in range(num_experts)
            ]
        )

        self.gate = SwitchGate(
            dim,
            num_experts,
            capacity_factor,
        )

    def forward(self, x: Tensor):
        """
        Forward pass of the SwitchMoE module.

        Args:
            x (Tensor): The input tensor.

        Returns:
            Tensor: The output tensor of the MoE.

        """
        # gate_scores:(batch_size, seq_len, num_experts)
        gate_scores, loss = self.gate(x)

        # Dispatch to experts
        expert_outputs = [expert(x) for expert in self.experts]

        # Check if any gate scores are nan and handle
        if torch.isnan(gate_scores).any():
            print("NaN in gate scores")
            gate_scores[torch.isnan(gate_scores)] = 0

        # Stack and weight outputs
        stacked_expert_outputs = torch.stack(
            expert_outputs, dim=-1
        )  # (batch_size, seq_len, output_dim, num_experts)

        if torch.isnan(stacked_expert_outputs).any():
            stacked_expert_outputs[
                torch.isnan(stacked_expert_outputs)
            ] = 0

        # Combine expert outputs and gating scores
        moe_output = torch.sum(
            gate_scores.unsqueeze(-2) * stacked_expert_outputs, dim=-1
        )

        # moe_output = torch.sum(
        #     stacked_expert_outputs, dim=-1
        # )

        return moe_output, loss

class GaT(nn.Module):
    def __init__(self, dim, hidden_dim, num_experts, residue=True, mult=4, cross_residual=True):
        super(GaT, self).__init__()
        self.residue = residue
        self.dim = dim

        self.Seq_experts = nn.ModuleList([
            FeedForward(dim, dim, mult)
            for _ in range(num_experts)
        ])

        self.Graph_experts = nn.ModuleList([
            FeedForward(dim, dim, mult)
            for _ in range(num_experts)
        ])

        # self weight for experts
        self.seq2exp_seq = nn.Linear(dim, num_experts, bias=False)
        self.gra2exp_gra = nn.Linear(dim, num_experts, bias=False)
        # cross weight for experts
        self.seq2exp_gra = nn.Linear(dim, num_experts, bias=False)
        self.gra2exp_seq = nn.Linear(dim, num_experts, bias=False)

        self.ffn_seq = [FeedForward(dim, dim, mult).to('cuda') for i in range(num_experts)]
        self.ffn_gra = [FeedForward(dim, dim, mult).to('cuda') for i in range(num_experts)]

        self.fc_seq = nn.Linear(dim * num_experts, dim)
        self.fc_gra = nn.Linear(dim * num_experts, dim)

        # cross model
        self.cross_attn = CrossAttention(dim, dim, hidden_dim, cross_residual)

    def forward(self, x_seq, g, seq_pad_len):
        """
        GaT module define switched experts that Graph tokens and Transformer tokens is through Cross-Attention
        :param x_seq: (batch, n, dim)
        :param g: Graph
        :return: new x_seq and x_gra feature
        """

        # # get all node id
        # subgraph_ids = g.ndata['subgraph_id']
        # seen = set()
        # # classifier node belong to batch
        # result = [x.item() for x in subgraph_ids if not (x.item() in seen or seen.add(x.item()))]
        # print(result)
        #
        # # obtain each batch's nodes feature
        # for batch_id, subgraph_id in enumerate(result):
        #     subgraph_nodes = (g.ndata['subgraph_id'] == subgraph_id).nonzero().squeeze(1)
        #     batch_x_gra = g.ndata['h'][subgraph_nodes]
        #
        #     btach_x_seq = x_seq[batch_id]

        batch_size = x_seq.size(0)
        len_ = x_seq.size(1)
        batch_nodes_feature = []
        batch_nodes_mask = []
        # get all node id
        subgraph_ids = g.ndata['subgraph_id'] # (20480,)
        unique_ids = torch.unique(subgraph_ids) # (512,)
        for batch_id, subgraph_id in enumerate(unique_ids):
            mask = (subgraph_ids == subgraph_id)
            batch_x_gra = g.ndata['h'][mask] # (40,64)
            batch_nodes_feature.append(batch_x_gra)
            # get non-padding node mask
            pad_nodes = g.ndata['node_label'][mask].eq(15) # (40,)
            batch_nodes_mask.append(pad_nodes)

        x_gra = torch.stack(batch_nodes_feature) # (batch,node_num,dim) # (512,40,64)

        mask_seq = get_seq_mask(x_seq, seq_pad_len) # (batch, n)
        mask_gra = torch.stack(batch_nodes_mask) # (batch, node_num)

        # print(f'mask_gra :{mask_gra}')

        # mask_seq = mask_seq.unsqueeze(2).eq(0) # (batch, n, 1)
        # mask_gra = mask_gra.unsqueeze(2).eq(0) # (batch, node_num, 1)
        
        x_seq, x_gra = self.cross_attn(x_seq, x_gra, mask_seq.to('cuda'), mask_gra)

        # seq_expert_noutputs = [expert(x_seq) for expert in self.Seq_experts]
        # gra_expert_outputs = [expert(x_gra) for expert in self.Graph_experts]


        # update nodes feature
        if self.residue:
            g.ndata['h'] = g.ndata['h'] + torch.reshape(x_gra,(-1, self.dim))
        else:
            g.ndata['h'] = torch.reshape(x_gra, (-1, self.dim))

        x_seq_moe = []
        x_gra_moe = []
        for expert in self.ffn_seq:
            x_seq_moe.append(expert(x_seq))
            # x_seq_moe.append(expert(x_seq) * mask_seq.unsqueeze(2).to('cuda'))

        for expert in self.ffn_gra:
            x_gra_moe.append(expert(x_gra))
            # x_gra_moe.append(expert(x_gra) * mask_gra.unsqueeze(2).to('cuda'))

        x_seq_gate = F.softmax(self.seq2exp_seq(x_seq), dim=-1) # (batch, n, num_experts)
        x_gra_gate = F.softmax(self.gra2exp_gra(x_gra), dim=-1) # (batch, node_num, num_experts)

        x_seq_moe = torch.stack(x_seq_moe,-1) # (batch, n, dim, num_experts)
        x_gra_moe = torch.stack(x_gra_moe,-1) # (batch, node_num, dim, num_experts)

        # x_seq = self.fc_seq(x_seq_moe.reshape(batch_size, len_, -1))
        # x_gra = self.fc_gra(x_gra_moe.reshape(batch_size, 40, -1))

        x_seq = torch.sum(x_seq_gate.unsqueeze(-2) * x_seq_moe, dim=-1)
        x_gra = torch.sum(x_gra_gate.unsqueeze(-2) * x_gra_moe, dim=-1)
        

        return x_seq, x_gra





# -----------------Seqence Model-----------------
class TransformerBlock(nn.Module):
    def __init__(self, dim, heads, dim_head, mult):
        super(TransformerBlock, self).__init__()
        self.lmsa = LMSA(dim, heads, dim_head)
        self.norm = nn.LayerNorm(dim)
        self.ffn = FeedForward(dim, dim, mult)

    def forward(self, x, pad_mask):
        x = self.lmsa(x, pad_mask)
        x = x + self.ffn(self.norm(x))

        return x

class SeqMoE(nn.Module):
    def __init__(self):
        super(SeqMoE, self).__init__()
        self.experts = nn.ModuleList([
            FeedForward(dim, dim, mult)
            for _ in range(num_experts)
        ])

    def forward(self, g):

        pass



class LMSA(nn.Module):
    def __init__(self, dim, heads, dim_head):
        super(LMSA, self).__init__()

        self.attn = MultiHeadAttention(None, dim, heads, dim_head)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, enc_self_attn_mask):
        """
        :param x:  (batch, src_len, dim)
        :param enc_self_attn_mask: (batch, src_len, src_len)
        :return:
        """
        # x = self.norm(x + self.attn(x,x,x,enc_self_attn_mask)[0])

        res = x
        x, _ = self.attn(x,x,x,enc_self_attn_mask)
        x = x + res
        x = self.norm(x)
        

        return x


class SwitchTransformer(nn.Module):
    """
    reference:https://github.com/kyegomez/SwitchTransformers/tree/main
    SwitchTransformer is a PyTorch module that implements a transformer model with switchable experts.

    Args:
        num_tokens (int): The number of tokens in the input vocabulary.
        dim (int): The dimensionality of the token embeddings and hidden states.
        heads (int): The number of attention heads.
        dim_head (int, optional): The dimensionality of each attention head. Defaults to 64.
        mult (int, optional): The multiplier for the hidden dimension in the feed-forward network. Defaults to 4.
        dropout (float, optional): The dropout rate. Defaults to 0.1.
        num_experts (int, optional): The number of experts in the switchable experts mechanism. Defaults to 3.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.
    """

    def __init__(
            self,
            num_tokens: int,
            dim: int,
            heads: int,
            dim_head: int = 64,
            mult: int = 4,
            dropout: float = 0.1,
            num_experts: int = 5,
            depth: int = 4,
            task_type: str= 'Regression',
            *args,
            **kwargs,
    ):
        super().__init__()
        self.num_tokens = num_tokens
        self.dim = dim
        self.heads = heads
        self.dim_head = dim_head
        self.mult = mult
        self.dropout = dropout
        self.num_experts = num_experts
        self.depth = depth
        self.fc_layers = depth + 2 # 4 + 2
        self.src_len = 10 # peptides length
        self.task_type = task_type

        self.embedding = nn.Embedding(num_tokens, dim)
        self.pos_emb = PositionalEncoding(dim)
        self.layers = nn.ModuleList([])

        for _ in range(depth):
            self.layers.append(
                SwitchTransformerBlock(
                    dim,
                    heads,
                    dim_head,
                    mult,
                    dropout,
                    num_experts,
                    *args,
                    **kwargs,
                )
            )

        self.to_out = nn.Sequential(
            nn.Softmax(dim=-1),
            nn.LayerNorm(dim),
            nn.Linear(dim, num_tokens),
        )

        self.switch = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, 1),
            nn.Softmax(dim=-1),
        )

        self.mlp = nn.Sequential(
            nn.Linear(num_tokens,num_tokens//2),
            nn.LeakyReLU(),
            nn.Linear(num_tokens//2,1),
        )

        self.fc = nn.Linear(self.src_len*self.dim, self.dim).to(device)
        self.mlp_layers = nn.ModuleList([nn.Linear(self.dim, self.dim).to(device) for _ in range(self.fc_layers)])
        if self.task_type == 'Classification':
            self.predictor = nn.Linear(self.dim, 2).to(device)
        elif self.task_type == 'Regression':
            self.predictor = nn.Linear(self.dim, 1).to(device)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the SwitchTransformer.

        Args:
            x (Tensor): The input tensor of shape (batch_size, sequence_length).

        Returns:
            Tensor: The output tensor of shape (batch_size, sequence_length, num_tokens).
        """
        # get token mask
        enc_self_attn_mask = get_attn_pad_mask(x, x)
        # Embed tokens through embedding layer
        x = self.embedding(x)
        x = self.pos_emb(x.transpose(0,1)).transpose(0,1)
        # Pass through the transformer block with MoE, it's in modulelist
        aux_loss = 0
        for layer in self.layers:
            x, loss = layer(x, enc_self_attn_mask)
            aux_loss += loss

        # poduct weight to sum x's sequence
        # w:[512,10,1], x:[512,10,64]
        w = self.switch(x)

        x = w * x # x:[512,10,64]

        x = torch.reshape(x,(x.shape[0],-1)) # (512,640)

        hidden = self.fc(x) # (512,64)
        h = hidden.clone()
        # Predictor
        for layer in self.mlp_layers:
            h = layer(F.leaky_relu(h))
        y = self.predictor(F.leaky_relu(h))

        if self.task_type == 'Classification':
            return F.log_softmax(y, dim=1),hidden, aux_loss
        elif self.task_type == 'Regression':
            return y,hidden, aux_loss





# ----------------Graph Model-------------------
class Gnn_block(nn.Module):
    def __init__(self, gnn_type, dim, gnn_depth=1, GraphSAGE_aggregator='lstm', graph_head=3, gnn_residue=True):
        super(Gnn_block, self).__init__()
        self.gnn_residue = gnn_residue
        self.dim = dim
        if gnn_type == 'gin':
            self.gnn = nn.ModuleList([GINConv(self.lin,'max', activation=relu) for _ in range(gnn_depth)])
        elif gnn_type == 'gcn':
            self.gnn = nn.ModuleList([GraphConv(dim, dim) for _ in range(gnn_depth)])
        elif gnn_type == 'gat':
            self.gnn = nn.ModuleList([GATConv(dim, dim, num_heads=graph_head) for _ in range(gnn_depth)])
        elif gnn_type == 'sage':
            self.gnn = nn.ModuleList([SAGEConv(dim, dim, GraphSAGE_aggregator) for _ in range(gnn_depth)])
        else:
            raise ValueError('Undefined GNN type called {}'.format(gnn_type))


    def forward(self, g):
        for layer in self.gnn:
            x = g.ndata['h'] # (20480,64)
            h_node = layer(g, x) # (20480,64)
            if self.gnn_residue:
                g.ndata['h'] = F.relu(h_node) + x
            else:
                g.ndata['h'] = F.relu(h_node)
            # h_node = dgl.mean_nodes(g, 'h')
        
        # with g.local_scope():
        #     for layer in self.gnn:
        #         x = g.ndata['h'] # (20480,64)
        #         h_node = layer(g, x) # (20480,64)
        #         if self.gnn_residue:
        #             g.ndata['h'] = F.relu(h_node) + x
        #         else:
        #             g.ndata['h'] = F.relu(h_node)
        #         # h_node = dgl.mean_nodes(g, 'h')

        return h_node


class AtomEncoder(torch.nn.Module):

    def __init__(self, args, emb_dim):
        super(AtomEncoder, self).__init__()

        self.args = args
        self.emb = torch.nn.Embedding(self.args.src_vocab_size_graph + 1, emb_dim)
        torch.nn.init.xavier_uniform_(self.emb.weight.data)

    def forward(self, x):
        x_embedding = self.emb(x)

        return x_embedding

class GMoE(torch.nn.Module):

    def __init__(self, args, num_tasks, num_layer = 5, emb_dim = 300,
                    gnn_type = 'gcn', virtual_node = True, moe=True, residual = False, drop_ratio = 0.5, JK = "last", graph_pooling = "mean",
                    num_experts=3, k=1, coef=1e-2, hop=1, num_experts_1hop=None):
        '''
            num_tasks (int): number of labels to be predicted
            virtual_node (bool): whether to add virtual node or not
        '''

        super(GMoE, self).__init__()

        self.args = args

        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK
        self.emb_dim = emb_dim
        self.num_tasks = num_tasks
        self.graph_pooling = graph_pooling

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        ### GNN to generate node embeddings
        self.gnn_node = GNN_SpMoE_node(args, num_layer, emb_dim, JK = JK, drop_ratio = drop_ratio, residual = residual, gnn_type = gnn_type, num_experts=num_experts, k=k, coef=coef, num_experts_1hop=num_experts_1hop)

        if graph_pooling == "set2set":
            self.graph_pred_linear = torch.nn.Linear(2*self.emb_dim, self.num_tasks)
        else:
            self.graph_pred_linear = torch.nn.Linear(self.emb_dim, self.num_tasks)

        hidden_layer = [args.hidden,args.hidden,args.hidden,args.hidden]
        self.e1 = nn.Linear(args.d_graph, hidden_layer[0])
        self.e2 = nn.Linear(hidden_layer[0], hidden_layer[1])
        self.e3 = nn.Linear(hidden_layer[1], hidden_layer[2])
        self.e4 = nn.Linear(hidden_layer[2], hidden_layer[3])
        self.sigmoid = torch.nn.Sigmoid()
        self.predictor = nn.Linear(args.hidden, args.output_layer)

    def forward(self, g):
        with g.local_scope():
            x = g.ndata['feat'] # (9325,)

            h_node = self.gnn_node(g, x) # (num_nodes,256)

            g.ndata['h'] = h_node

            # graph mean all nodes
            h_node = dgl.sum_nodes(g, 'h')  # (batchsize,256)

            # graph sum all nodes
            # h_node = dgl.sum_nodes(g,'h') # (batchsize,256)

        hidden = F.relu(self.e1(h_node))  # (512,64)
        # Predictor
        h_2 = F.relu(self.e2(hidden))  # (512,64)
        h_3 = F.relu(self.e3(h_2))  # (512,64)
        h_4 = F.relu(self.e4(h_3))  # (512,64)
        y = self.predictor(h_4)  # (512,1)

        if self.args.task_type == 'Classification':
            return F.log_softmax(y, dim=1), hidden
        elif self.args.task_type == 'Regression':
            return y, hidden



if __name__ == '__main__':
    print()
