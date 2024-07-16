from __future__ import print_function
import abc

import numpy as np
import logging
import torch
import torch.utils.data
from torch import nn
from torch.nn import functional as F
import math

from .nearest_embed import NearestEmbed, NearestEmbedEMA

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class AbstractAutoEncoder(nn.Module):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def encode(self, x):
        return

    @abc.abstractmethod
    def decode(self, z):
        return

    @abc.abstractmethod
    def forward(self, x):
        """model return (reconstructed_x, *)"""
        return

    @abc.abstractmethod
    def sample(self, size):
        """sample new images from model"""
        return

    @abc.abstractmethod
    def loss_function(self, **kwargs):
        """accepts (original images, *) where * is the same as returned from forward()"""
        return

    @abc.abstractmethod
    def latest_losses(self):
        """returns the latest losses in a dictionary. Useful for logging."""
        return


class VAE(nn.Module):
    """Variational AutoEncoder for MNIST
       Taken from pytorch/examples: https://github.com/pytorch/examples/tree/master/vae"""
    def __init__(self, kl_coef=1, emb_size=784, enc_hidden_size=400, dec_hidden_size=400, latent_size=20, **kwargs):
        super(VAE, self).__init__()

        self.emb_size = emb_size
        self.latent_size = latent_size
        self.enc_hidden_size = enc_hidden_size
        self.dec_hidden_size = dec_hidden_size
        
        self.fc1 = nn.Linear(self.emb_size, self.enc_hidden_size)
        self.fc21 = nn.Linear(self.enc_hidden_size, self.latent_size)
        self.fc22 = nn.Linear(self.enc_hidden_size, self.latent_size)
        self.fc3 = nn.Linear(self.latent_size, self.dec_hidden_size)
        self.fc4 = nn.Linear(self.dec_hidden_size, self.emb_size)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.kl_coef = kl_coef
        self.bce = 0
        self.kl = 0

    def encode(self, x):
        h1 = self.relu(self.fc1(x))
        mu = self.fc21(h1)
        logvar = self.fc22(h1)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = std.new(std.size()).normal_()     # N(0, I)
            return eps.mul(std).add_(mu)            # z = mu + eps * var
        else:
            return mu

    def decode(self, z):
        h3 = self.relu(self.fc3(z))
        h4 = self.fc4(h3)
        return h4

    def forward(self, x):
        # mu, logvar = self.encode(x.view(-1, self.emb_size))
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        print(z.size())
        output = self.decode(z)
        return output, mu, logvar

    def sample(self, size):
        sample = torch.randn(size, self.latent_size)
        if self.cuda():
            sample = sample.cuda()
        sample = self.decode(sample)
        return sample

# ===========Transformer==================================

def get_attn_pad_mask(seq_q, seq_k):
    batch_size, len_q = seq_q.size()
    batch_size, len_k = seq_k.size()
    pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)
    return pad_attn_mask.expand(batch_size, len_q, len_k)

class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V, attn_mask, args):
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(args.d_k)
        scores.masked_fill_(attn_mask, -1e9)

        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)
        return context, attn


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


class MultiHeadAttention(nn.Module):
    def __init__(self,args):
        super(MultiHeadAttention, self).__init__()
        self.args = args
        self.W_Q = nn.Linear(args.d_model, args.d_k * args.n_heads, bias=False)
        self.W_K = nn.Linear(args.d_model, args.d_k * args.n_heads, bias=False)
        self.W_V = nn.Linear(args.d_model, args.d_v * args.n_heads, bias=False)
        self.fc = nn.Linear(args.n_heads * args.d_v, args.d_model, bias=False)

    def forward(self,input_Q, input_K, input_V, attn_mask):
        residual, batch_size = input_Q, input_Q.size(0)

        Q = self.W_Q(input_Q).view(batch_size,-1, self.args.n_heads, self.args.d_k).transpose(1,2) # Q:[batch_size, n_heads, len_q, d_k]
        K = self.W_K(input_K).view(batch_size,-1, self.args.n_heads, self.args.d_k).transpose(1,2) # K:[batch_size, n_heads, len_k, d_k]
        V = self.W_V(input_V).view(batch_size,-1, self.args.n_heads, self.args.d_v).transpose(1,2) # V:[batch_size, n_heads, len_v, d_v]

        attn_mask = attn_mask.unsqueeze(1).repeat(1, self.args.n_heads, 1, 1)
        context, attn = ScaledDotProductAttention()(Q,K,V, attn_mask,self.args)
        context = context.transpose(1,2).reshape(batch_size, -1, self.args.n_heads * self.args.d_v)
        output = self.fc(context) # (512,10,64)

        return nn.LayerNorm(self.args.d_model).to(device)(output+residual),attn

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
        enc_outputs = self.pos_emb(enc_outputs.transpose(0,1)).transpose(0,1) # (512,10,64)
        enc_self_attn_mask = get_attn_pad_mask(enc_inputs,enc_inputs) # (512,10,10)
        enc_self_attns = []
        for layer in self.layers:
            enc_outputs, enc_self_attn = layer(enc_outputs, enc_self_attn_mask)
            enc_self_attns.append(enc_self_attn)
        return enc_outputs, enc_self_attns

class VQ_VAE(nn.Module):
    """Vector Quantized AutoEncoder for mnist"""
    def __init__(self, args, emb_size=640, enc_hidden_size=400, dec_hidden_size=400, latent_size=200, k=10, vq_coef=0.2, comit_coef=0.4, **kwargs):
        super(VQ_VAE, self).__init__()

        self.encoder = Encoder(args)

        self.ss_size = k
        self.emb_size = emb_size
        self.enc_hidden_size = enc_hidden_size
        self.dec_hidden_size = dec_hidden_size
        self.fc1 = nn.Linear(emb_size, enc_hidden_size)
        self.fc2 = nn.Linear(enc_hidden_size, latent_size)
        self.fc3 = nn.Linear(latent_size, dec_hidden_size)
        self.fc4 = nn.Linear(dec_hidden_size, emb_size)
        self.fc5 = nn.Linear(emb_size, emb_size // 2)
        self.fc6 = nn.Linear(emb_size // 2, 1)

        self.emb = NearestEmbed(k, self.ss_size)

        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.vq_coef = vq_coef
        self.comit_coef = comit_coef
        self.latent_size = latent_size
        self.ce_loss = 0
        self.vq_loss = 0
        self.commit_loss = 0

    def encode(self, x):
        x, _ = self.encoder(x) # x:(512,10,64)

        x = x.view(x.size(0), -1) # (512,640)

        h1 = self.relu(self.fc1(x)) # (512,400)
        h2 = self.fc2(h1) # (512,200)
        # return:(512, 10, 20)
        return h2.view(-1, self.ss_size, int(self.latent_size / self.ss_size))

    def decode(self, z):
        h3 = self.relu(self.fc3(z))
        h4 = self.relu(self.fc4(h3))
        h5 = self.relu(self.fc5(h4))
        h6 = self.fc6(h5)
        return h6,h4

    def forward(self, x):
        z_e = self.encode(x) # (512, 10, 20)
        # z_e = self.encode(x.view(-1, self.emb_size))
        # z_q, _ = self.emb(z_e, weight_sg=True).view(-1, self.latent_size)
        z_q, _ = self.emb(z_e, weight_sg=True) # z_q:(512, 10, 20); _:(512,20)
        z_q = z_q.view(-1, self.latent_size) # z_q:(512, 200)
        # emb, _ = self.emb(z_e.detach()).view(-1, self.latent_size)
        emb, _ = self.emb(z_e.detach()) # emb:(512, 10, 20)
        # emb = emb.view(-1, self.latent_size)

        y, hidden = self.decode(z_q) # y:(512,1); hidden:(512,640)

        return y, hidden, z_e, emb

    def sample(self, size):
        sample = torch.randn(size, self.ss_size, int(self.latent_size / self.ss_size))
        if self.cuda():
            sample = sample.cuda()
        emb, _ = self.emb(sample)
        _, sample = self.decode(emb(sample).view(-1, self.latent_size)).cpu()
        return sample

    def loss_function(self, x, y, z_e, emb):
        self.predict_loss = F.mse_loss(x,y) # 重建损失
        # self.ce_loss = F.binary_cross_entropy(recon_x, x.view(-1, self.emb_size))
        self.vq_loss = F.mse_loss(emb, z_e.detach()) # 第二项
        self.commit_loss = F.mse_loss(z_e, emb.detach()) # 第三项

        # return self.predict_loss + self.ce_loss + self.vq_coef*self.vq_loss + self.comit_coef*self.commit_loss

        return self.predict_loss + self.vq_coef * self.vq_loss + self.comit_coef * self.commit_loss

    def latest_losses(self):
        return {'cross_entropy': self.ce_loss, 'vq': self.vq_loss, 'commitment': self.commit_loss}


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, bn=False):
        super(ResBlock, self).__init__()

        if mid_channels is None:
            mid_channels = out_channels

        layers = [
            nn.ReLU(),
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=1, stride=1, padding=0)]
        if bn:
            layers.insert(2, nn.BatchNorm2d(out_channels))
        self.convs = nn.Sequential(*layers)

    def forward(self, x):
        return x + self.convs(x)


class CVAE(AbstractAutoEncoder):
    def __init__(self, d, kl_coef=0.1, **kwargs):
        super(CVAE, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(3, d // 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(d // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(d // 2, d, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(d),
            nn.ReLU(inplace=True),
            ResBlock(d, d, bn=True),
            nn.BatchNorm2d(d),
            ResBlock(d, d, bn=True),
        )
        self.decoder = nn.Sequential(
            ResBlock(d, d, bn=True),
            nn.BatchNorm2d(d),
            ResBlock(d, d, bn=True),
            nn.BatchNorm2d(d),

            nn.ConvTranspose2d(d, d // 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(d//2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(d // 2, 3, kernel_size=4, stride=2, padding=1, bias=False),
        )
        self.f = 8
        self.d = d
        self.fc11 = nn.Linear(d * self.f ** 2, d * self.f ** 2)
        self.fc12 = nn.Linear(d * self.f ** 2, d * self.f ** 2)
        self.kl_coef = kl_coef
        self.kl_loss = 0
        self.mse = 0

    def encode(self, x):
        h1 = self.encoder(x)
        h1 = h1.view(-1, self.d * self.f ** 2)
        return self.fc11(h1), self.fc12(h1)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = std.new(std.size()).normal_()
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, z):
        z = z.view(-1, self.d, self.f, self.f)
        h3 = self.decoder(z)
        return torch.tanh(h3)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    def sample(self, size):
        sample = torch.randn(size, self.d * self.f ** 2, requires_grad=False)
        if self.cuda():
            sample = sample.cuda()
        return self.decode(sample).cpu()

    def loss_function(self, x, recon_x, mu, logvar):
        self.mse = F.mse_loss(recon_x, x)
        batch_size = x.size(0)

        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        self.kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        # Normalise by same number of elements as in reconstruction
        self.kl_loss /= batch_size * 3 * 1024

        # return mse
        return self.mse + self.kl_coef * self.kl_loss

    def latest_losses(self):
        return {'mse': self.mse, 'kl': self.kl_loss}


class VQ_CVAE(nn.Module):
    def __init__(self, d, k=10, bn=True, vq_coef=1, commit_coef=0.5, num_channels=3, **kwargs):
        super(VQ_CVAE, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(num_channels, d, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(d),
            nn.ReLU(inplace=True),
            nn.Conv2d(d, d, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(d),
            nn.ReLU(inplace=True),
            ResBlock(d, d, bn=bn),
            nn.BatchNorm2d(d),
            ResBlock(d, d, bn=bn),
            nn.BatchNorm2d(d),
        )
        self.decoder = nn.Sequential(
            ResBlock(d, d),
            nn.BatchNorm2d(d),
            ResBlock(d, d),
            nn.ConvTranspose2d(d, d, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(d),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(d, num_channels, kernel_size=4, stride=2, padding=1),
        )
        self.d = d
        self.emb = NearestEmbed(k, d)
        self.vq_coef = vq_coef
        self.commit_coef = commit_coef
        self.mse = 0
        self.vq_loss = torch.zeros(1)
        self.commit_loss = 0

        for l in self.modules():
            if isinstance(l, nn.Linear) or isinstance(l, nn.Conv2d):
                l.weight.detach().normal_(0, 0.02)
                torch.fmod(l.weight, 0.04)
                nn.init.constant_(l.bias, 0)

        self.encoder[-1].weight.detach().fill_(1 / 40)

        self.emb.weight.detach().normal_(0, 0.02)
        torch.fmod(self.emb.weight, 0.04)

    def encode(self, x):
        return self.encoder(x)

    def decode(self, x):
        return torch.tanh(self.decoder(x))

    def forward(self, x):
        z_e = self.encode(x)
        self.f = z_e.shape[-1]
        z_q, argmin = self.emb(z_e, weight_sg=True)
        emb, _ = self.emb(z_e.detach())
        return self.decode(z_q), z_e, emb, argmin

    def sample(self, size):
        sample = torch.randn(size, self.d, self.f, self.f, requires_grad=False),
        if self.cuda():
            sample = sample.cuda()
        emb, _ = self.emb(sample)
        return self.decode(emb.view(size, self.d, self.f, self.f)).cpu()

    def loss_function(self, x, recon_x, z_e, emb, argmin):
        self.mse = F.mse_loss(recon_x, x)

        self.vq_loss = torch.mean(torch.norm((emb - z_e.detach())**2, 2, 1))
        self.commit_loss = torch.mean(torch.norm((emb.detach() - z_e)**2, 2, 1))

        return self.mse + self.vq_coef*self.vq_loss + self.commit_coef*self.commit_loss

    def latest_losses(self):
        return {'mse': self.mse, 'vq': self.vq_loss, 'commitment': self.commit_loss}

    def print_atom_hist(self, argmin):

        argmin = argmin.detach().cpu().numpy()
        unique, counts = np.unique(argmin, return_counts=True)
        logging.info(counts)
        logging.info(unique)
