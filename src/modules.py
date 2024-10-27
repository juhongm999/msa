import torch
import torch.nn as nn
import torch.nn.functional as F

import math
import numpy as np


class Retina(nn.Module):
    def __init__(self, g, k, s):
        super().__init__()

        self.g = g
        self.k = k
        self.s = s

    def foveate(self, x, l):
        """Extract `k` square patches of size `g`, centered
        at location `l`. The initial patch is a square of
        size `g`, and each subsequent patch is a square
        whose side is `s` times the size of the previous
        patch.
        """
        phi = []
        size = self.g

        # extract k patches of increasing size
        for i in range(self.k):
            phi.append(self.extract_patch(x, l, size))
            size = int(self.s * size)

        # resize the patches to squares of size g
        for i in range(1, len(phi)):
            k = phi[i].shape[-1] // self.g
            phi[i] = F.avg_pool2d(phi[i], k)

        # concatenate into a single tensor and flatten
        phi = torch.cat(phi, 1)
        phi = phi.view(phi.shape[0], -1)

        return phi

    def extract_patch(self, x, l, size):
        """Extract a single patch for each image in `x`.
        """
        B, C, H, W = x.shape

        start = self.denormalize(H, l)
        end = start + size

        # pad with zeros
        x = F.pad(x, (size // 2, size // 2, size // 2, size // 2))

        # loop through mini-batch and extract patches
        patch = []
        for i in range(B):
            patch.append(
                x[i, :, start[i, 1]: end[i, 1], start[i, 0]: end[i, 0]])
        return torch.stack(patch)

    def denormalize(self, H, coords):
        """Convert coordinates in the range [-1, 1] to
        coordinates in the range [0, H] where `H` is
        the size of the image.
        """
        return (0.5 * ((coords + 1.0) * H)).long()


class GlimpseNetwork(nn.Module):
    def __init__(self, g, k, s, c, h_g, h_l, n_hiddens):
        super().__init__()

        self.retina = Retina(g, k, s)

        # glimpse layer
        # TODO: replace with conv layer
        D_in = k * g * g * c
        self.fc1 = nn.Linear(D_in, h_g)

        # location layer
        D_in = 2
        self.fc2 = nn.Linear(D_in, h_l)

        self.fc3 = nn.Linear(h_g, n_hiddens)
        self.fc4 = nn.Linear(h_l, n_hiddens)

    def forward(self, x, l_t_prev):
        # generate glimpse phi from image x
        phi = self.retina.foveate(x, l_t_prev)

        # flatten location vector
        l_t_prev = l_t_prev.view(l_t_prev.size(0), -1)

        # feed phi and l to respective fc layers
        phi_out = F.relu(self.fc1(phi))
        l_out = F.relu(self.fc2(l_t_prev))

        what = self.fc3(phi_out)
        where = self.fc4(l_out)

        # feed to fc layer
        g_t = F.relu(what + where)

        return g_t


class TestCoreNetwork(nn.Module):
    """Modified implementation of nn.TransformerEncoderLayer
    """

    def __init__(self, use_memory, d_model, n_heads=1, n_glimpses=10,
                 dim_feedforward=256, dropout=0.1,
                 batch_first=True, layer_norm_eps=1e-5):
        super().__init__()

        self.use_memory = use_memory
        self.d_model = d_model  # same as n_hiddens

        if use_memory:

            self.mem = []

            # Vanilla Transformer: the key and value will always be of equal size.
            self.self_attn = nn.MultiheadAttention(
                d_model, n_heads, dropout=dropout, batch_first=batch_first)

            self.pe = None

            # sa_block
            self.dropout1 = nn.Dropout(dropout)
            self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
            self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)

            # ff_block
            self.linear1 = nn.Linear(d_model, dim_feedforward)
            self.dropout2 = nn.Dropout(dropout)
            self.linear2 = nn.Linear(dim_feedforward, d_model)
            self.dropout3 = nn.Dropout(dropout)

            self.n_glimpses = n_glimpses

            self.out = nn.Linear(d_model * n_glimpses, d_model)

        else:
            self.g2h = nn.Linear(d_model, d_model)
            self.h2h = nn.Linear(d_model, d_model)

    def forward(self, *args):
        g_t = args[0]
        if self.use_memory:
            self.mem.append(g_t)
            h_t, attention = self._multi_head_attn()
            # TODO: maybe remove this relu?
            return F.relu(h_t), attention
        else:
            h_t_prev = args[1]
            h_t = self.g2h(g_t) + self.h2h(h_t_prev)
            return F.relu(h_t)

    def _multi_head_attn(self):
        if self.pe is None:  # hack to put on same device as inputs
            self.pe = self._get_positional_encoding()

        x = torch.stack(self.mem, dim=1)  # b x g' x d
        len_mem = x.shape[1]
        x = x + self.pe[:, :len_mem]
        x = torch.cat([x, torch.zeros(  # b x g x d
            (x.shape[0], self.n_glimpses-len_mem, x.shape[2]),
            device=x.device)], dim=1)

        # TODO: test with diagonal mask?
        # attn_mask = self._generate_square_subsequent_mask(self.n_glimpses)

        attn_mask = torch.zeros(
            (self.n_glimpses, self.n_glimpses), device=x.device)
        if len_mem < self.n_glimpses:
            attn_mask[:, len_mem:] = -float('inf')
        # attn_mask = None

        # key_padding_mask = torch.tensor([1,1,1,0,0,0]).repeat(10,1).bool()
        key_padding_mask = None

        attn_output, attention = self._sa_block(
            x, attn_mask, key_padding_mask)
        z = self.norm1(x + attn_output)  # b x g x d
        z = self.norm2(z + self._ff_block(z))  # b x g x d

        # zero-out padding embeddings (with attn_mask)
        z = z * ~attn_mask[0].unsqueeze(0).unsqueeze(-1).bool()
        h_t = self.out(z.view(z.shape[0], -1))  # b x g x d -> b x d

        # ignore masking/padding and just take mean
        # h_t = z.mean(dim=1)  # b x g x d -> b x d

        return h_t, attention

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float(
            '-inf')).masked_fill(mask == 1, float(0.0))
        return mask.to(self.linear1.weight.device)

    def _get_positional_encoding(self, n=10000.0):
        pe = torch.zeros(self.n_glimpses, self.d_model,
                         device=self.linear1.weight.device)
        position = torch.arange(
            0, self.n_glimpses, dtype=torch.float).unsqueeze(1)
        denom = torch.exp(torch.arange(0, self.d_model, 2).float()
                          * (-math.log(n) / self.d_model))
        pe[:, 0::2] = torch.sin(position * denom)
        pe[:, 1::2] = torch.cos(position * denom)
        pe = pe.unsqueeze(0)  # 1, L, E
        return pe

    def _sa_block(self, x, attn_mask, key_padding_mask):
        attn_output, attn_output_weights = self.self_attn(x, x, x,
                                                          attn_mask=attn_mask,
                                                          key_padding_mask=key_padding_mask,
                                                          need_weights=True,
                                                          average_attn_weights=False)
        return self.dropout1(attn_output), attn_output_weights

    def _ff_block(self, x):
        x = self.linear2(self.dropout2(F.relu(self.linear1(x))))
        return self.dropout3(x)


# class CoreNetwork(nn.Module):
#     def __init__(self, n_hiddens, use_memory, n_glimpses, h_m):
#         super().__init__()

#         self.use_memory = use_memory
#         self.n_hiddens = n_hiddens

#         if use_memory:

#             self.n_glimpses = n_glimpses

#             self.wk = nn.Linear(n_hiddens, h_m)
#             self.wq = nn.Linear(n_hiddens, h_m)
#             self.wv = nn.Linear(n_hiddens, n_hiddens)

#             self.norm = nn.LayerNorm(
#                 n_hiddens, elementwise_affine=False)

#             # same feed-forward network is independently applied to each position
#             self.fc = nn.Linear(n_hiddens, h_m)

#             self.out = nn.Linear(h_m * n_glimpses, n_hiddens)

#             self.scale = 1 / np.sqrt(h_m)
#             self.pe = None

#             self.mem = []

#         else:
#             self.g2h = nn.Linear(n_hiddens, n_hiddens)
#             self.h2h = nn.Linear(n_hiddens, n_hiddens)

#     def _get_positional_encoding(self, g, n_hiddens, n=10000):
#         pe = torch.zeros(g, n_hiddens, device=self.wk.weight.device)
#         for pos in range(g):
#             for i in range(0, n_hiddens, 2):
#                 demon = n ** ((2 * i) / n_hiddens)
#                 pe[pos, i] = np.sin(pos / demon)
#                 pe[pos, i + 1] = np.cos(pos / demon)
#         return pe

#     def _scaled_dot_product_attention(self):
#         if self.pe is None:  # hack to put on same device as inputs
#             self.pe = self._get_positional_encoding(
#                 self.n_glimpses, self.n_hiddens, n=12)
#         m_t = torch.stack(self.mem, dim=1)  # b x g' x d
#         m_t = torch.cat([m_t, torch.zeros(  # b x g x d
#             (m_t.shape[0], self.n_glimpses-m_t.shape[1], m_t.shape[2]),
#             device=m_t.device)], dim=1)
#         m_t = m_t + self.pe  # positional encoding, b x g x d

#         K, Q = self.wk(m_t), self.wq(m_t)  # b x g x m
#         V = self.wv(m_t)  # b x g x d

#         # (QK^T + M) / sqrt(d)
#         energy = torch.bmm(K, Q.transpose(2, 1)) * self.scale  # b x g x g
#         if len(self.mem) < self.n_glimpses:
#             energy[:, :, len(self.mem)] = -float('inf')

#         attention = F.softmax(energy, dim=2)

#         sa = torch.bmm(attention, V)  # b x g x d

#         # residule connection
#         sa = self.norm(sa + m_t)

#         # feed-forward network
#         sa = F.relu(self.fc(sa))  # b x g x m

#         h_t = self.out(sa.view(sa.shape[0], -1))

#         return h_t, attention

#     def forward(self, *args):
#         g_t = args[0]
#         if self.use_memory:
#             self.mem.append(g_t)
#             h_t, attention = self._scaled_dot_product_attention()
#             return F.relu(h_t), attention
#         else:
#             h_t_prev = args[1]
#             h_t = self.g2h(g_t) + self.h2h(h_t_prev)
#             return F.relu(h_t)


class LocationNetwork(nn.Module):
    def __init__(self, n_hiddens, loc_std):
        super().__init__()

        self.loc_std = loc_std
        hid_size = n_hiddens // 2
        self.fc = nn.Linear(n_hiddens, hid_size)
        self.fc_lt = nn.Linear(hid_size, 2)

    def forward(self, h_t):
        # compute mean
        feat = F.relu(self.fc(h_t.detach()))
        mu = torch.tanh(self.fc_lt(feat))

        # reparametrization trick
        l_t = torch.distributions.Normal(mu, self.loc_std).rsample()
        l_t = l_t.detach()
        log_pi = torch.distributions.Normal(mu, self.loc_std).log_prob(l_t)

        # we assume both dimensions are independent
        # 1. pdf of the joint is the product of the pdfs
        # 2. log of the product is the sum of the logs
        log_pi = torch.sum(log_pi, dim=1)

        # TODO: replace with tanh, l_t = torch.tanh(l_t)
        l_t = torch.clamp(l_t, -1, 1)

        return log_pi, l_t


class ActionNetwork(nn.Module):
    def __init__(self, n_hiddens, n_outputs):
        super().__init__()
        self.out = nn.Linear(n_hiddens, n_outputs)

    def forward(self, h_t):
        a_t = self.out(h_t)  # e.g., b x n_outputs
        return a_t


class BaselineNetwork(nn.Module):
    def __init__(self, n_hiddens):
        super().__init__()
        self.fc = nn.Linear(n_hiddens, 1)

    def forward(self, h_t):
        b_t = self.fc(h_t.detach())
        return b_t


# class MemoryNetwork(nn.Module):
#     def __init__(self, n_hiddens, n_glimpses, h_m):
#         super().__init__()
#         self.mem = []
#         self.n_hiddens = n_hiddens
#         self.n_glimpses = n_glimpses
#         self.h_m = h_m

#         self.wk = nn.Linear(n_hiddens, h_m)
#         self.wq = nn.Linear(n_hiddens, h_m)

#         self.scale = 1 / np.sqrt(h_m)
#         self.pe = None

#     def _get_positional_encoding(self, g, n_hiddens, n=10000):
#         pe = torch.zeros(g, n_hiddens, device=self.wk.weight.device)
#         for pos in range(g):
#             for i in range(0, n_hiddens, 2):
#                 demon = n ** ((2 * i) / n_hiddens)
#                 pe[pos, i] = np.sin(pos / demon)
#                 pe[pos, i + 1] = np.cos(pos / demon)
#         return pe

#     def scaled_dot_product_attention(self):
#         if self.pe is None:  # hack to put on same device as inputs
#             self.pe = self._get_positional_encoding(
#                 self.n_glimpses, self.n_hiddens, n=12)
#         m_t = torch.stack(self.mem, dim=1)  # b x g x m
#         m_t = m_t + self.pe  # positional encoding, b x g x m
#         K, Q = self.wk(m_t), self.wq(m_t)  # b x g x kq
#         energy = torch.bmm(K, Q.transpose(2, 1)) * self.scale  # b x g x g
#         attention = F.softmax(energy, dim=2)
#         return attention

#     def forward(self, g_t):
#         self.mem.append(g_t.detach())


if __name__ == '__main__':

    # Transformer
    use_memory = True

    d_model = 256
    n_heads = 1
    n_glimpses = 6
    dim_feedforward = 256
    dropout = 0.1

    test = TestCoreNetwork(use_memory, d_model, n_heads,
                           n_glimpses, dim_feedforward, dropout)

    B = 10
    g_t = torch.rand((B, d_model))

    h_t, attention = test(g_t)
    print(h_t.shape, attention.shape)

    # RNN
    use_memory = False

    d_model = 256

    test = TestCoreNetwork(use_memory, d_model)

    B = 10
    g_t = torch.rand((B, d_model))
    h_t_prev = g_t

    h_t = test(g_t, h_t_prev)
    print(h_t.shape)
