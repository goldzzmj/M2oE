import math
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
import dgl
from dgl.nn.pytorch import GraphConv,GATConv,SAGEConv
from typing import Optional

device = 'cuda' if torch.cuda.is_available() else 'cpu'

############# (a) Sequential Encoder & Predictor #############

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))  # 10000^{2i/d_model}
        pe[:, 0::2] = torch.sin(position * div_term)  
        pe[:, 1::2] = torch.cos(position * div_term) 
        pe = pe.unsqueeze(0).transpose(0, 1) 
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :] 
        return self.dropout(x)

def get_attn_pad_mask(seq_q, seq_k):
    batch_size, len_q = seq_q.size()
    batch_size, len_k = seq_k.size()
    pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)
    return pad_attn_mask.expand(batch_size, len_q, len_k)


def get_attn_subsequence_mask(seq):
    attn_shape = [seq.size(0), seq.size(1), seq.size(1)]
    subsequence_mask = np.triu(np.ones(attn_shape), k=1)
    subsequence_mask = torch.from_numpy(subsequence_mask).byte()
    return subsequence_mask

class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V, attn_mask,d_k):
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k) # (512,64,10,10)
        if attn_mask != None:
            scores.masked_fill_(attn_mask, -1e9) 
        
        attn = nn.Softmax(dim=-1)(scores) # (512,64,10,10)
        context = torch.matmul(attn, V) # (512,64,10,8)
        return context, attn

class MultiHeadAttention(nn.Module):
    def __init__(self,parse, *args, **kwargs,):
        super(MultiHeadAttention, self).__init__()
        self.parse = parse
        if self.parse == None:
            self.d_model, self.d_k,self.n_heads = args
            self.d_v = self.d_k

        else:
            self.d_model, self.d_k, self.d_v, self.n_heads =\
                self.parse.d_model,self.parse.d_k, self.parse.d_v, self.parse.n_heads
        self.W_Q = nn.Linear(self.d_model, self.d_k * self.n_heads, bias=False)
        self.W_K = nn.Linear(self.d_model, self.d_k * self.n_heads, bias=False)
        self.W_V = nn.Linear(self.d_model, self.d_v * self.n_heads, bias=False)
        self.fc = nn.Linear(self.n_heads * self.d_v, self.d_model, bias=False)

    def forward(self,input_Q, input_K, input_V, attn_mask):
        residual, batch_size = input_Q, input_Q.size(0)

        Q = self.W_Q(input_Q).view(batch_size,-1, self.n_heads, self.d_k).transpose(1,2) # Q:[batch_size, n_heads, len_q, d_k]
        K = self.W_K(input_K).view(batch_size,-1, self.n_heads, self.d_k).transpose(1,2) # K:[batch_size, n_heads, len_k, d_k]
        V = self.W_V(input_V).view(batch_size,-1, self.n_heads, self.d_v).transpose(1,2) # V:[batch_size, n_heads, len_v, d_v]
        if attn_mask != None:
            attn_mask = attn_mask.unsqueeze(1).repeat(1, self.n_heads, 1, 1) # (512,64,10,10)
        context, attn = ScaledDotProductAttention()(Q,K,V, attn_mask,self.d_k)
        context = context.transpose(1,2).reshape(batch_size, -1, self.n_heads * self.d_v) # (512,10,512)
        output = self.fc(context) # (512,10,64)

        return nn.LayerNorm(self.d_model).to(device)(output+residual),attn

class PoswiseFeedForwardNet(nn.Module):
    def __init__(self,args):
        super(PoswiseFeedForwardNet, self).__init__()
        self.args = args
        self.fc = nn.Sequential(
            nn.Linear(args.d_model,args.d_ff,bias=False),
            nn.ReLU(),
            nn.Linear(args.d_ff, args.d_model, bias=False)
        )

    def forward(self, inputs):
        '''
        :param inputs: [batch_size, seq_len, d_model]
        :return:
        '''
        residual = inputs
        output = self.fc(inputs)
        return nn.LayerNorm(self.args.d_model).to(device)(output+residual)


class CrossAttention(nn.Module):
    def __init__(self, input_dim_a, input_dim_b, hidden_dim, residual=True):
        super(CrossAttention, self).__init__()
        self.residual = residual
        # self.residual = residuals
        self.input_dim_a = input_dim_a
        self.input_dim_b = input_dim_b

        self.linear_a = nn.Linear(input_dim_a, hidden_dim) 
        self.linear_b = nn.Linear(input_dim_b, hidden_dim)

    def forward(self, input_a, input_b, mask_a, mask_b):
        # Linear
        mapped_a = self.linear_a(input_a)  # (batch_size, seq_len_a, hidden_dim)
        mapped_b = self.linear_b(input_b)  # (batch_size, seq_len_b, hidden_dim)

        mask_a = mask_a.unsqueeze(2)
        mask_b = mask_b.unsqueeze(2)

        # get mask matrix in score matrix
        final_mask = torch.bmm(mask_a.float(), mask_b.transpose(1,2).float()).eq(0)

        # compute attention score
        scores = torch.matmul(mapped_a, mapped_b.transpose(1, 2))  # (batch_size, seq_len_a, seq_len_b)
        # scale
        scores = scores / math.sqrt(self.input_dim_a)
        # mask padding
        scores.masked_fill_(final_mask, -1e9)
        attentions_a = torch.softmax(scores, dim=-1)  # (batch_size, seq_len_a, seq_len_b)
        attentions_b = torch.softmax(scores.transpose(1, 2),
                                     dim=-1)  # (batch_size, seq_len_b, seq_len_a)

        # Use attention weights to adjust the input representation
        output_a = torch.matmul(attentions_b.transpose(1, 2), input_b)  # (batch_size, seq_len_a, input_dim_b)
        output_b = torch.matmul(attentions_a.transpose(1, 2), input_a)  # (batch_size, seq_len_b, input_dim_a)
        

        if self.residual:
            return output_a + input_a, output_b + input_b

        return output_a, output_b

        # return nn.LayerNorm(self.input_dim_a).to(device)(output_a+input_a), nn.LayerNorm(self.input_dim_b).to(device)(output_b+input_b)

class EncoderLayer(nn.Module):
    def __init__(self,args):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention(args)
        self.pos_ffn = PoswiseFeedForwardNet(args)

    def forward(self,enc_inputs, enc_self_attn_mask):
        enc_outputs, attn = self.enc_self_attn(enc_inputs,enc_inputs,enc_inputs,enc_self_attn_mask)
        enc_outputs = self.pos_ffn(enc_outputs) 
        return enc_outputs, attn


class Encoder(nn.Module):
    def __init__(self,args):
        super(Encoder, self).__init__()
        self.src_emb = nn.Embedding(args.src_vocab_size_seq, args.d_model)
        self.pos_emb = PositionalEncoding(args.d_model)
        self.layers = nn.ModuleList([EncoderLayer(args) for _ in range(args.n_layers)])

    def forward(self, enc_inputs):
        enc_outputs = self.src_emb(enc_inputs)
        enc_outputs = self.pos_emb(enc_outputs.transpose(0,1)).transpose(0,1) 
        enc_self_attn_mask = get_attn_pad_mask(enc_inputs,enc_inputs) 
        enc_self_attns = []
        for layer in self.layers:
            enc_outputs, enc_self_attn = layer(enc_outputs, enc_self_attn_mask)
            enc_self_attns.append(enc_self_attn)
        return enc_outputs, enc_self_attns

class Transformer(nn.Module):
    def __init__(self,args):
        super(Transformer,self).__init__()
        self.args = args
        self.encoder = Encoder(args).to(device)
        self.fc = nn.Linear(args.src_len*args.d_model, args.hidden).to(device)
        self.layers = nn.ModuleList([nn.Linear(args.hidden, args.hidden).to(device) for _ in range(args.fc_layers)])
        self.predictor = nn.Linear(args.hidden, args.output_layer).to(device)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self,enc_inputs):
        '''
        :param enc_inputs: [batch_size, src_len]
        :param dec_inputs: [batch_size, tgt_len]
        :return:
        '''
        # Sequential Encoder
        enc_outputs,_ = self.encoder(enc_inputs) # enc_outputs:(512,10,64) , enc_attns:6*[(512,8,10,10)]
        # Reshape the token-level repr. to sequential repr. h_seq
        dec_inputs = torch.reshape(enc_outputs,(enc_outputs.shape[0],-1)) # (512,640)
        hidden = self.fc(dec_inputs) # (512,64)
        h = hidden.clone()
        # Predictor
        for layer in self.layers:
            h = layer(F.leaky_relu(h))
        y = self.predictor(F.leaky_relu(h))

        if self.args.task_type == 'Classification':
            return F.log_softmax(y, dim=1), hidden
        elif self.args.task_type == 'Regression':
            return y, hidden


class vTransformer(nn.Module):
    def __init__(self,args):
        super(vTransformer,self).__init__()
        self.args = args
        self.encoder = Encoder(args).to(device)
        self.fc = nn.Linear(args.src_len*args.d_model, args.hidden).to(device)
        self.layers = nn.ModuleList([nn.Linear(args.hidden, args.hidden).to(device) for _ in range(args.fc_layers)])
        self.predictor = nn.Linear(args.hidden, args.output_layer).to(device)
        self.sigmoid = torch.nn.Sigmoid()

        self.mean = nn.Linear(args.hidden, args.hidden).to(device)
        self.log_std = nn.Linear(args.hidden, args.hidden).to(device)

    def forward(self,enc_inputs):
        '''
        :param enc_inputs: [batch_size, src_len]
        :param dec_inputs: [batch_size, tgt_len]
        :return:
        '''
        # Sequential Encoder
        enc_outputs,_ = self.encoder(enc_inputs) # enc_outputs:(512,10,64) , enc_attns:6*[(512,8,10,10)]
        # Reshape the token-level repr. to sequential repr. h_seq
        dec_inputs = torch.reshape(enc_outputs,(enc_outputs.shape[0],-1)) # (512,640)
        hidden = self.fc(dec_inputs) # (512,64)
        h = hidden.clone()
        # VAE
        mean = self.mean(h)
        log_std = self.log_std(h)
        # reparamer
        gaussian_noise = torch.randn(hidden.size(0), self.args.hidden).to(
            device
        )
        h = mean + gaussian_noise * torch.exp(log_std).to(
            device
        )
        sample_hidden = h.clone()
        # Predictor
        for layer in self.layers:
            h = layer(F.leaky_relu(h))
        y = self.predictor(F.leaky_relu(h))

        if self.args.task_type == 'Classification':
            return F.log_softmax(y, dim=1), sample_hidden, mean, log_std
        elif self.args.task_type == 'Regression':
            return y, sample_hidden, mean, log_std


# =================Switch Transformer==========================
# https://github.com/kyegomez/SwitchTransformers/blob/main/switch_transformers/model.py

class FeedForward(nn.Module):
    def __init__(
        self,
        dim: Optional[int] = None,
        dim_out: Optional[int] = None,
        mult: Optional[int] = 4,
        dropout: Optional[float] = 0.0,
    ):
        """
        https://github.com/kyegomez/zeta/blob/master/zeta/nn/modules/feedforward.py
        FeedForward module that applies a series of linear transformations and activations.

        Args:
            dim (int): Input dimension.
            dim_out (int, optional): Output dimension. Defaults to None.
            mult (int, optional): Multiplier for the inner dimension. Defaults to 4.
            glu (bool, optional): Whether to use Gated Linear Units (GLU). Defaults to False.
            glu_mult_bias (bool, optional): Whether to use bias in the GLU operation. Defaults to False.
            swish (bool, optional): Whether to use Swish activation. Defaults to False.
            relu_squared (bool, optional): Whether to use squared ReLU activation. Defaults to False.
            post_act_ln (bool, optional): Whether to apply Layer Normalization after the activation. Defaults to False.
            dropout (float, optional): Dropout probability. Defaults to 0.0.
            no_bias (bool, optional): Whether to use bias in the linear transformations. Defaults to False.
            zero_init_output (bool, optional): Whether to initialize the last linear layer to 0. Defaults to False.
            custom_act (nn.Module, optional): Custom activation module. Defaults to None.
            swiglu (bool, optional): Whether to use SwiGLU activation. Defaults to False.
        """
        super().__init__()
        self.dim = dim
        self.dim_out = dim_out
        self.mult = mult
        self.dropout = dropout


        inner_dim = int(dim * mult)
        dim_out = dim_out

        activation = nn.ReLU()

        project_in = nn.Sequential(
            nn.Linear(dim, inner_dim, bias=False), activation
        )

        self.ff = nn.Sequential(
            project_in,
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim_out, bias=False),
        )

    def forward(self, x):
        """
        Forward pass of the feedforward network

        Args:
            x (torch.Tensor): Input tensor

        Returns:
            torch.Tensor: Output tensor
        """
        return self.ff(x)

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

    def forward(self, x: Tensor, use_aux_loss=False):
        """
        Forward pass of the SwitchGate module.

        Args:
            x (Tensor): Input tensor. [batch,seq,dim]=[512,10,64]

        Returns:
            Tensor: Gate scores.
        """
        # Compute gate scores
        batch = x.size(0) # 512
        N = x.size(1) #100
        x = x.reshape((batch*N,-1)) # (51200,64)
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

        if use_aux_loss:
            # load = gate_scores.sum(0)  # Sum over all examples
            # importance = gate_scores.sum(1)  # Sum over all experts
            #
            # # Aux loss is mean suqared difference between load and importance
            # loss = ((load - importance) ** 2).mean()

            importance = gate_scores.sum(0)
            load = (gate_scores > 0).sum(0)

            loss = self.cv_squared(importance) + self.cv_squared(load)

            gate_scores = gate_scores.reshape((batch, N, -1))

            return gate_scores, loss

        gate_scores = gate_scores.reshape((batch, N, -1))

        return gate_scores, None


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
            use_aux_loss: bool = False,
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
        self.use_aux_loss = use_aux_loss

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
        # (batch_size, seq_len, num_experts)
        gate_scores, loss = self.gate(
            x, use_aux_loss=self.use_aux_loss
        )

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


class SwitchTransformerBlock(nn.Module):
    """
    SwitchTransformerBlock is a module that represents a single block of the Switch Transformer model.

    Args:
        dim (int): The input dimension of the block.
        heads (int): The number of attention heads.
        dim_head (int): The dimension of each attention head.
        mult (int, optional): The multiplier for the hidden dimension in the feed-forward network. Defaults to 4.
        dropout (float, optional): The dropout rate. Defaults to 0.1.
        depth (int, optional): The number of layers in the block. Defaults to 12.
        num_experts (int, optional): The number of experts in the SwitchMoE layer. Defaults to 6.
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Attributes:
        dim (int): The input dimension of the block.
        heads (int): The number of attention heads.
        dim_head (int): The dimension of each attention head.
        mult (int): The multiplier for the hidden dimension in the feed-forward network.
        dropout (float): The dropout rate.
        attn_layers (nn.ModuleList): List of MultiQueryAttention layers.
        ffn_layers (nn.ModuleList): List of SwitchMoE layers.

    Examples:
        # >>> block = SwitchTransformerBlock(dim=512, heads=8, dim_head=64)
        # >>> x = torch.randn(1, 10, 512)
        # >>> out = block(x)
        # >>> out.shape

    """

    def __init__(
            self,
            dim: int,
            heads: int,
            dim_head: int,
            mult: int = 4,
            dropout: float = 0.1,
            num_experts: int = 3,
            *args,
            **kwargs,
    ):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.dim_head = dim_head
        self.mult = mult
        self.dropout = dropout

        # self.attn = MultiQueryAttention(
        #     dim, heads, qk_ln=True * args, **kwargs
        # )

        self.attn = MultiHeadAttention(
            None, dim, heads, dim_head,
        )

        self.moe = SwitchMoE(
            dim, dim * mult, dim, num_experts, use_aux_loss = True, *args, **kwargs
        )

        self.ffn = FeedForward(dim, dim, mult)

        self.add_norm = nn.LayerNorm(dim)

    def forward(self, x: Tensor, enc_self_attn_mask):
        """
        Forward pass of the SwitchTransformerBlock.

        Args:
            x (Tensor): The input tensor.

        Returns:
            Tensor: The output tensor.

        """
        resi = x
        # enc_self_attn_mask = get_attn_pad_mask(x, x)
        # enc_self_attn_mask = torch.ones((x.size()[0],x.size()[1],x.size()[1]),device='cuda',dtype=torch.int)
        x, _ = self.attn(x,x,x,enc_self_attn_mask)
        x = x + resi
        x = self.add_norm(x)
        add_normed = x

        #### FFN ######
        # x = self.ffn(x)
        # x = x + add_normed
        # x = self.add_norm(x)

        # ##### MoE #####
        x, loss = self.moe(x)
        x = x + add_normed
        x = self.add_norm(x)
        return x, loss


class SwitchTransformer(nn.Module):
    """
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
        self.src_len = 100 # peptides length
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

############# (a) Sequential Encoder & Predictor #############

class GNNs(nn.Module):
    def __init__(self, args):
        super(GNNs, self).__init__()

        self.args = args
        # GraphSAGE
        self.convlayers = nn.ModuleList([SAGEConv(args.d_graph, args.d_graph,args.GraphSAGE_aggregator)  for _ in range(args.conv_layers)])

        # # GCN
        # self.convlayers = nn.ModuleList(
        #     [GraphConv(args.d_graph, args.d_graph) for _ in range(args.conv_layers)])

        # # GAT
        # self.convlayers = nn.ModuleList(
        #     [GATConv(args.d_graph, args.d_graph, num_heads=3) for _ in range(args.conv_layers)])

        self.predictor = nn.Linear(args.hidden, args.output_layer)

        hidden_layer = [args.hidden,args.hidden,args.hidden,args.hidden]
        self.e1 = nn.Linear(args.d_graph, hidden_layer[0])
        self.e2 = nn.Linear(hidden_layer[0], hidden_layer[1])
        self.e3 = nn.Linear(hidden_layer[1], hidden_layer[2])
        self.e4 = nn.Linear(hidden_layer[2], hidden_layer[3])

        self.dropout = nn.Dropout(p=0.3)
        self.src_emb = nn.Embedding(args.src_vocab_size_graph, args.d_graph)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, g):

        # Graphical Encoder
        with g.local_scope():
            h = self.src_emb(g.ndata['feat']) # (9525,256)
            for layer in self.convlayers:
                h = F.relu(layer(g, h))
            
            g.ndata['h'] = h    
        # Readout with mean pooling
            hg = dgl.mean_nodes(g,'h') # (512,256)

        hidden = F.relu(self.e1(hg)) # (512,64)
        # Predictor
        h_2 = F.relu(self.e2(hidden)) # (512,64)
        h_3 = F.relu(self.e3(h_2)) # (512,64)
        h_4 = F.relu(self.e4(h_3)) # (512,64)
        y = self.predictor(h_4) # (512,1)

        # # GAT
        # y = torch.mean(torch.mean(y, dim=1), dim=1)
        
        if self.args.task_type == 'Classification':
            return F.log_softmax(y, dim=1),hidden
        elif self.args.task_type == 'Regression':
            return y,hidden


#===============GNN-MoE===================

class GNNs_MoE(nn.Module):
    def __init__(self, args, mult = 4, num_experts = 5, fc_layers=4):
        super(GNNs_MoE, self).__init__()

        self.args = args
        self.fc_layers = fc_layers
        self.convlayers = nn.ModuleList(
            [SAGEConv(args.d_graph, args.d_graph, args.GraphSAGE_aggregator) for _ in range(args.conv_layers)])
        self.predictor = nn.Linear(args.hidden, args.output_layer)

        hidden_layer = [args.hidden, args.hidden, args.hidden, args.hidden]
        self.dim = hidden_layer[0]
        self.e1 = nn.Linear(args.d_graph, hidden_layer[0])
        self.e2 = nn.Linear(hidden_layer[0], hidden_layer[1])
        self.e3 = nn.Linear(hidden_layer[1], hidden_layer[2])
        self.e4 = nn.Linear(hidden_layer[2], hidden_layer[3])

        self.dropout = nn.Dropout(p=0.3)
        self.src_emb = nn.Embedding(args.src_vocab_size_graph, args.d_graph)
        self.sigmoid = torch.nn.Sigmoid()

        self.experts = nn.ModuleList(
            [
                FeedForward(hidden_layer[0], hidden_layer[0], mult)
                for _ in range(num_experts)
            ]
        )

        self.head_2_dim = nn.Linear(self.dim * num_experts ,self.dim)

        self.mlp_layers = nn.ModuleList([nn.Linear(self.dim, self.dim).to(device) for _ in range(self.fc_layers)])


    def forward(self, g):

        # Graphical Encoder
        with g.local_scope():
            h = self.src_emb(g.ndata['feat'])  # (9525,256)
            for layer in self.convlayers:
                h = F.relu(layer(g, h))

            g.ndata['h'] = h
            # Readout with mean pooling
            hg = dgl.mean_nodes(g, 'h')  # (512,256)

        hidden = F.relu(self.e1(hg))  # (512,64)

        expert_outputs = [expert(hidden) for expert in self.experts]
        stacked_expert_outputs = torch.stack(   # (512, 64, num_experts)
            expert_outputs, dim=-1
        )  # (batch_size, seq_len, output_dim, num_experts)

        # Combine expert outputs and gating scores
        # moe_output = torch.sum(
        #     gate_scores.unsqueeze(-2) * stacked_expert_outputs, dim=-1
        # )

        # # sum all experts
        # h = torch.sum(
        #     stacked_expert_outputs, dim=-1
        # )

        # concat all experts
        h = stacked_expert_outputs.reshape((-1,
             stacked_expert_outputs.size(1) * stacked_expert_outputs.size(2)))

        h = self.head_2_dim(h)

        # Predictor
        for layer in self.mlp_layers: # (512,64)
            h = layer(F.leaky_relu(h))

        y = self.predictor(F.leaky_relu(h))  # (512,1)

        if self.args.task_type == 'Classification':
            return F.log_softmax(y, dim=1), hidden
        elif self.args.task_type == 'Regression':
            return y, hidden

class VGAEModel(nn.Module):
    def __init__(self, args):
        super(VGAEModel, self).__init__()

        self.args = args
        self.in_dim = args.src_vocab_size_graph
        self.hidden1_dim = args.d_graph
        self.hidden2_dim = args.d_graph

        self.convlayers = nn.ModuleList([SAGEConv(args.d_graph, args.d_graph,args.GraphSAGE_aggregator)  for _ in range(args.conv_layers)])

        layers = [
            GraphConv(
                self.in_dim,
                self.hidden1_dim,
                activation=F.relu,
                allow_zero_in_degree=True,
            ),
            GraphConv(
                self.hidden1_dim,
                self.hidden2_dim,
                activation=lambda x: x,
                allow_zero_in_degree=True,
            ),
            GraphConv(
                self.hidden1_dim,
                self.hidden2_dim,
                activation=lambda x: x,
                allow_zero_in_degree=True,
            ),
        ]
        self.layers = nn.ModuleList(layers)

        self.src_emb = nn.Embedding(args.src_vocab_size_graph, args.d_graph)
        self.predictor = nn.Linear(args.hidden, args.output_layer)

        hidden_layer = [args.d_graph,args.d_graph,args.hidden,args.hidden,args.hidden]
        # encoder
        self.e1 = nn.Linear(args.d_graph, hidden_layer[0])
        self.e2 = nn.Linear(hidden_layer[0], hidden_layer[1])
        self.mu = nn.Linear(hidden_layer[1], hidden_layer[2])
        self.std = nn.Linear(hidden_layer[1], hidden_layer[2])
        # decoder
        self.e3 = nn.Linear(hidden_layer[2], hidden_layer[3])
        self.e4 = nn.Linear(hidden_layer[3], hidden_layer[4])

    def encoder(self, g):
        # Graphical Encoder
        with g.local_scope():
            features = self.src_emb(g.ndata['feat']) # (9525,256)
            for layer in self.convlayers:
                h = F.relu(layer(g, features))

            g.ndata['h'] = h
            # Readout with mean pooling
            hg = dgl.mean_nodes(g, 'h') # (512,256)
            # h = self.layers[0](g, features)
        # self.mean = self.layers[1](g, h)
        # self.log_std = self.layers[2](g, h)
        hidden = F.relu(self.e1(hg)) # (512,64)
        # Predictor
        h_2 = F.relu(self.e2(hidden))
        self.mean = F.relu(self.mu(h_2)) # (512,64)
        self.log_std = F.relu(self.std(h_2)) # (512,64)

        # (512,64)
        gaussian_noise = torch.randn(hidden.size(0), self.args.hidden).to(
            device
        )
        sampled_z = self.mean + gaussian_noise * torch.exp(0.5*self.log_std).to(
            device
        )
        return sampled_z

    def decoder(self, z):
        adj_rec = torch.sigmoid(torch.matmul(z, z.t()))
        return adj_rec

    def mlp_decoder(self,z):
        # Predictor
        h_3 = F.relu(self.e3(z)) # (512,64)
        h_4 = F.relu(self.e4(h_3)) # (512,64)
        y = self.predictor(h_4)

        if self.args.task_type == 'Classification':
            return F.log_softmax(y, dim=1),z
        elif self.args.task_type == 'Regression':
            return y,z


    def forward(self, g):
        z = self.encoder(g)
        # adj_rec = self.decoder(z)
        y,hidden = self.mlp_decoder(z)

        if self.args.task_type == 'Classification':
            return F.log_softmax(y, dim=1), hidden, self.mean,self.log_std
        elif self.args.task_type == 'Regression':
            return y, hidden, self.mean,self.log_std

class Decoder(nn.Module):
    """ The decoder layer converting state to observation.
    Because the observation is MNIST image whose elements are values
    between 0 and 1, the output of this layer are probabilities of
    elements being 1.
    """

    def __init__(self, z_size, hidden_size, x_size):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(z_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, x_size)

    def forward(self, z):
        t = torch.tanh(self.fc1(z))
        t = torch.tanh(self.fc2(t))
        p = torch.sigmoid(self.fc3(t))
        return p

class DBlock(nn.Module):
    """ A basie building block for parametralize a normal distribution.
    It is corresponding to the D operation in the reference Appendix.
    """

    def __init__(self, input_size, hidden_size, output_size):
        super(DBlock, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(input_size, hidden_size)
        self.fc_mu = nn.Linear(hidden_size, output_size)
        self.fc_logsigma = nn.Linear(hidden_size, output_size)

    def forward(self, input):
        t = torch.tanh(self.fc1(input))
        t = t * torch.sigmoid(self.fc2(input))
        mu = self.fc_mu(t)
        logsigma = self.fc_logsigma(t)
        return mu, logsigma

class PreProcess(nn.Module):
    """ The pre-process layer for MNIST image

    """
    def __init__(self, args, input_size, processed_x_size):
        super(PreProcess, self).__init__()
        self.embedding = nn.Embedding(100 ,processed_x_size)
        self.input_size = input_size
        self.processed_x_size = processed_x_size
        self.fc1 = nn.Linear(processed_x_size, processed_x_size)
        self.fc2 = nn.Linear(processed_x_size, processed_x_size)

        self.encoder = Encoder(args).to(device)

    def forward(self, input):
        # e = self.embedding(input) # (512,10,64)
        # e = e.view(-1,self.processed_x_size)
        # t = torch.relu(self.fc1(e))
        # t = torch.relu(self.fc2(t))
        # t = t.view(-1, self.input_size, self.processed_x_size)

        t, _ = self.encoder(input) # (512,10,64)
        return t

class TD_VAE(nn.Module):
    def __init__(self,args, x_size, processed_x_size, b_size, z_size):
        super(TD_VAE, self).__init__()
        self.args = args
        self.x_size = x_size
        self.processed_x_size = processed_x_size
        self.b_size = b_size
        self.z_size = z_size

        ## input pre-process layer
        self.process_x = PreProcess(args, self.x_size, self.processed_x_size)

        ## one layer LSTM for aggregating belief states
        ## One layer LSTM is used here and I am not sure how many layers
        ## are used in the original paper from the paper.
        self.lstm = nn.LSTM(input_size=self.processed_x_size,
                            hidden_size=self.b_size,
                            batch_first=True)

        ## Two layer state model is used. Sampling is done by sampling
        ## higher layer first.
        ## belief to state (b to z)
        ## (this is corresponding to P_B distribution in the reference;
        ## weights are shared across time but not across layers.)
        self.l2_b_to_z = DBlock(b_size, 50, z_size)  # layer 2
        self.l1_b_to_z = DBlock(b_size + z_size, 50, z_size)  # layer 1

        ## Given belief and state at time t2, infer the state at time t1
        ## infer state
        self.l2_infer_z = DBlock(b_size + 2 * z_size, 50, z_size)  # layer 2
        self.l1_infer_z = DBlock(b_size + 2 * z_size + z_size, 50, z_size)  # layer 1

        ## Given the state at time t1, model state at time t2 through state transition
        ## state transition
        self.l2_transition_z = DBlock(2 * z_size, 50, z_size)
        self.l1_transition_z = DBlock(2 * z_size + z_size, 50, z_size)

        ## state to observation
        self.z_to_x = Decoder(2 * z_size, 200, processed_x_size)

        # predict
        self.processed_x_to_y2 = nn.Linear(processed_x_size, processed_x_size * 2)
        self.processed_x_to_y1 = nn.Linear(processed_x_size * 2, 1)

        self.mu_to_y2 = nn.Linear(processed_x_size,processed_x_size * 2)
        self.mu_to_y1 = nn.Linear(processed_x_size * 2, 1)


    def forward(self, peptide, y, t2):
        self.batch_size = peptide.size()[0]
        self.x = peptide # (512,10)
        self.y = y # (512,1)
        ## pre-precess peptide x
        self.processed_x = self.process_x(self.x) # (512,10,64)

        ## aggregate the belief b
        self.b, (h_n, c_n) = self.lstm(self.processed_x)

        self.y2 = F.relu(self.processed_x_to_y2(self.b))

        self.y1 = F.relu(self.processed_x_to_y1(self.y2))

        # t2_l2_z_mu, t2_l2_z_logsigma = self.l2_b_to_z(self.b[:, t2, :])
        #
        # t2_l1_z_mu, t2_l1_z_logsigma = self.l1_b_to_z(
        #     torch.cat((self.b[:, t2, :], t2_l2_z_mu), dim=-1))
        #
        # self.mu_y2 = self.mu_to_y2(t2_l1_z_mu)
        # self.mu_y1 = self.mu_to_y1(self.mu_y2)
        #
        # return (torch.mean(self.y1,dim=1) + self.mu_y1) / 2

        return torch.mean(self.y1, dim=1)

    def predict(self):
        y2 = F.relu(self.processed_x_to_y2(self.processed_x))
        y1 = F.relu(self.processed_x_to_y1(y2))

        if self.args.task_type == 'Classification':
            return F.log_softmax(torch.mean(y1,dim=1), dim=1)
        elif self.args.task_type == 'Regression':
            return torch.mean(y1,dim=1)



    def calculate_loss(self, t1, t2):
        """ Calculate the jumpy VD-VAE loss, which is corresponding to
        the equation (6) and equation (8) in the reference.

        """

        ## Because the loss is based on variational inference, we need to
        ## draw samples from the variational distribution in order to estimate
        ## the loss function.

        ## sample a state at time t2 (see the reparametralization trick is used)
        ## z in layer 2
        t2_l2_z_mu, t2_l2_z_logsigma = self.l2_b_to_z(self.b[:, t2, :])
        t2_l2_z_epsilon = torch.randn_like(t2_l2_z_mu)
        t2_l2_z = t2_l2_z_mu + torch.exp(t2_l2_z_logsigma) * t2_l2_z_epsilon

        ## z in layer 1
        t2_l1_z_mu, t2_l1_z_logsigma = self.l1_b_to_z(
            torch.cat((self.b[:, t2, :], t2_l2_z), dim=-1))
        t2_l1_z_epsilon = torch.randn_like(t2_l1_z_mu)
        t2_l1_z = t2_l1_z_mu + torch.exp(t2_l1_z_logsigma) * t2_l1_z_epsilon

        ## concatenate z from layer 1 and layer 2
        t2_z = torch.cat((t2_l1_z, t2_l2_z), dim=-1)

        ## sample a state at time t1
        ## infer state at time t1 based on states at time t2
        t1_l2_qs_z_mu, t1_l2_qs_z_logsigma = self.l2_infer_z(
            torch.cat((self.b[:, t1, :], t2_z), dim=-1))
        t1_l2_qs_z_epsilon = torch.randn_like(t1_l2_qs_z_mu)
        t1_l2_qs_z = t1_l2_qs_z_mu + torch.exp(t1_l2_qs_z_logsigma) * t1_l2_qs_z_epsilon

        t1_l1_qs_z_mu, t1_l1_qs_z_logsigma = self.l1_infer_z(
            torch.cat((self.b[:, t1, :], t2_z, t1_l2_qs_z), dim=-1))
        t1_l1_qs_z_epsilon = torch.randn_like(t1_l1_qs_z_mu)
        t1_l1_qs_z = t1_l1_qs_z_mu + torch.exp(t1_l1_qs_z_logsigma) * t1_l1_qs_z_epsilon

        t1_qs_z = torch.cat((t1_l1_qs_z, t1_l2_qs_z), dim=-1)

        #### After sampling states z from the variational distribution, we can calculate
        #### the loss.

        ## state distribution at time t1 based on belief at time 1
        t1_l2_pb_z_mu, t1_l2_pb_z_logsigma = self.l2_b_to_z(self.b[:, t1, :])
        t1_l1_pb_z_mu, t1_l1_pb_z_logsigma = self.l1_b_to_z(
            torch.cat((self.b[:, t1, :], t1_l2_qs_z), dim=-1))

        ## state distribution at time t2 based on states at time t1 and state transition
        t2_l2_t_z_mu, t2_l2_t_z_logsigma = self.l2_transition_z(t1_qs_z)
        t2_l1_t_z_mu, t2_l1_t_z_logsigma = self.l1_transition_z(
            torch.cat((t1_qs_z, t2_l2_z), dim=-1))

        ## observation distribution at time t2 based on state at time t2
        t2_x_prob = self.z_to_x(t2_z)

        #

        #### start calculating the loss

        #### KL divergence between z distribution at time t1 based on variational distribution
        #### (inference model) and z distribution at time t1 based on belief.
        #### This divergence is between two normal distributions and it can be calculated analytically

        ## KL divergence between t1_l2_pb_z, and t1_l2_qs_z
        loss = 0.5 * torch.sum(((t1_l2_pb_z_mu - t1_l2_qs_z) / torch.exp(t1_l2_pb_z_logsigma)) ** 2, -1) + \
               torch.sum(t1_l2_pb_z_logsigma, -1) - torch.sum(t1_l2_qs_z_logsigma, -1)

        ## KL divergence between t1_l1_pb_z and t1_l1_qs_z
        loss += 0.5 * torch.sum(((t1_l1_pb_z_mu - t1_l1_qs_z) / torch.exp(t1_l1_pb_z_logsigma)) ** 2, -1) + \
                torch.sum(t1_l1_pb_z_logsigma, -1) - torch.sum(t1_l1_qs_z_logsigma, -1)

        #### The following four terms estimate the KL divergence between the z distribution at time t2
        #### based on variational distribution (inference model) and z distribution at time t2 based on transition.
        #### In contrast with the above KL divergence for z distribution at time t1, this KL divergence
        #### can not be calculated analytically because the transition distribution depends on z_t1, which is sampled
        #### after z_t2. Therefore, the KL divergence is estimated using samples

        ## state log probabilty at time t2 based on belief
        loss += torch.sum(-0.5 * t2_l2_z_epsilon ** 2 - 0.5 * t2_l2_z_epsilon.new_tensor(2 * np.pi) - t2_l2_z_logsigma,
                          dim=-1)
        loss += torch.sum(-0.5 * t2_l1_z_epsilon ** 2 - 0.5 * t2_l1_z_epsilon.new_tensor(2 * np.pi) - t2_l1_z_logsigma,
                          dim=-1)

        ## state log probabilty at time t2 based on transition
        loss += torch.sum(
            0.5 * ((t2_l2_z - t2_l2_t_z_mu) / torch.exp(t2_l2_t_z_logsigma)) ** 2 + 0.5 * t2_l2_z.new_tensor(
                2 * np.pi) + t2_l2_t_z_logsigma, -1)
        loss += torch.sum(
            0.5 * ((t2_l1_z - t2_l1_t_z_mu) / torch.exp(t2_l1_t_z_logsigma)) ** 2 + 0.5 * t2_l1_z.new_tensor(
                2 * np.pi) + t2_l1_t_z_logsigma, -1)

        # ## observation prob at time t2
        # loss += -torch.sum(self.processed_x[:, t2, :] * torch.log(t2_x_prob) + (1 - self.processed_x[:, t2, :]) * torch.log(1 - t2_x_prob),
        #                    -1)


        loss = torch.mean(loss)

        return loss

    def rollout(self, peptide, t1, t2):
        self.forward(peptide)

        ## at time t1-1, we sample a state z based on belief at time t1-1
        l2_z_mu, l2_z_logsigma = self.l2_b_to_z(self.b[:, t1 - 1, :])
        l2_z_epsilon = torch.randn_like(l2_z_mu)
        l2_z = l2_z_mu + torch.exp(l2_z_logsigma) * l2_z_epsilon

        l1_z_mu, l1_z_logsigma = self.l1_b_to_z(
            torch.cat((self.b[:, t1 - 1, :], l2_z), dim=-1))
        l1_z_epsilon = torch.randn_like(l1_z_mu)
        l1_z = l1_z_mu + torch.exp(l1_z_logsigma) * l1_z_epsilon
        current_z = torch.cat((l1_z, l2_z), dim=-1)

        rollout_x = []

        for k in range(t2 - t1 + 1):
            ## predicting states after time t1 using state transition
            next_l2_z_mu, next_l2_z_logsigma = self.l2_transition_z(current_z)
            next_l2_z_epsilon = torch.randn_like(next_l2_z_mu)
            next_l2_z = next_l2_z_mu + torch.exp(next_l2_z_logsigma) * next_l2_z_epsilon

            next_l1_z_mu, next_l1_z_logsigma = self.l1_transition_z(
                torch.cat((current_z, next_l2_z), dim=-1))
            next_l1_z_epsilon = torch.randn_like(next_l1_z_mu)
            next_l1_z = next_l1_z_mu + torch.exp(next_l1_z_logsigma) * next_l1_z_epsilon

            next_z = torch.cat((next_l1_z, next_l2_z), dim=-1)

            ## generate an observation x_t1 at time t1 based on sampled state z_t1
            next_x = self.z_to_x(next_z)
            rollout_x.append(next_x)

            current_z = next_z

        rollout_x = torch.stack(rollout_x, dim=1)

        return rollout_x





