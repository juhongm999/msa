import torch
import torch.nn as nn

import modules


class RecurrentAttention(nn.Module):
    """
    Params:
        g: size of the square patches in the glimpses extracted by the retina.
        k: number of patches to extract per glimpse.
        s: scaling factor that controls the size of successive patches.
        c: number of channels in each image.
        h_g: hidden layer size of the fc layer for `phi`.
        h_l: hidden layer size of the fc layer for `l`.
        n_hiddens: hidden size of the rnn.
        loc_std: standard deviation of the Gaussian policy.
        n_outputs: number of outputs in the dataset.
        n_glimpses: number of glimpses to take.
        h_m: dimension of memory vector.
    """

    def __init__(self, g, k, s, c, h_g, h_l, n_hiddens,
                 loc_std, n_outputs, n_glimpses,
                 use_memory, n_heads, dim_feedforward, dropout):
        super().__init__()

        self.g = g
        self.k = k
        self.s = s
        self.c = c
        self.h_g = h_g
        self.h_l = h_l

        self.n_hiddens = n_hiddens

        self.loc_std = loc_std
        self.n_outputs = n_outputs

        # specific to memory network
        self.use_memory = use_memory
        self.n_heads = n_heads
        self.n_glimpses = n_glimpses
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout

        self.glimpse = modules.GlimpseNetwork(g, k, s, c,
                                              h_g, h_l, n_hiddens)
        # self.core = modules.CoreNetwork(n_hiddens, use_memory,
        #                                 n_glimpses, h_m)
        self.core = modules.TestCoreNetwork(use_memory, n_hiddens, n_heads,
                                            n_glimpses, dim_feedforward, dropout)
        self.locator = modules.LocationNetwork(n_hiddens, loc_std)
        self.classifier = modules.ActionNetwork(n_hiddens, n_outputs)
        self.baseliner = modules.BaselineNetwork(n_hiddens)

    def reset(self, batch_size, device, loc='random'):
        """
        batch_size: size of the batch
        device: cpu or gpu
        loc: [center, top-left, top-middle, bottom-right, bottom-middle, random]
        """
        h_t = torch.zeros(
            batch_size,
            self.n_hiddens,
            dtype=torch.float,
            device=device,
            requires_grad=True,
        )
        if loc == 'center':
            l_t = torch.zeros(batch_size, 2)
        elif loc == 'top-middle':
            l_t = torch.zeros(batch_size, 2)
            l_t[:, 1] = -1.
        elif loc == 'top-left':
            l_t = torch.ones(batch_size, 2)
            l_t *= -1.
        elif loc == 'bottom-right':
            l_t = torch.ones(batch_size, 2)
        elif loc == 'bottom-middle':
            l_t = torch.zeros(batch_size, 2)
            l_t[:, 0] = 1.
        else:  # random
            l_t = torch.FloatTensor(batch_size, 2).uniform_(-1, 1)

        l_t = l_t.to(device)
        l_t.requires_grad = True

        if self.use_memory:
            self.core.mem = []

        return h_t, l_t

    def forward(self, x, l_t_prev, h_t_prev, last=False):
        g_t = self.glimpse(x, l_t_prev)

        if self.use_memory:
            h_t, attention = self.core(g_t)
        else:
            h_t = self.core(g_t, h_t_prev)
            attention = None

        log_pi, l_t = self.locator(h_t)
        b_t = self.baseliner(h_t).squeeze()

        if last:
            a_t = self.classifier(h_t)
            return h_t, l_t, b_t, a_t, attention, log_pi

        return h_t, l_t, b_t, log_pi


if __name__ == '__main__':

    config = {
        'g': 8,
        'k': 1,
        's': 1,
        'c': 1,
        'h_g': 256,
        'h_l': 64,
        'n_hiddens': 256,
        'loc_std': 0.1,
        'n_outputs': 10,
        'n_glimpses': 6,
        'use_memory': True,
        'n_heads': 4,
        'dim_feedforward': 256,
        'dropout': 0.1,
    }

    model = model = RecurrentAttention(
        config['g'],
        config['k'],
        config['s'],
        config['c'],
        config['h_g'],
        config['h_l'],
        config['n_hiddens'],
        config['loc_std'],
        config['n_outputs'],
        config['n_glimpses'],
        config['use_memory'],
        config['n_heads'],
        config['dim_feedforward'],
        config['dropout'],
    )
    print(model)
    print(
        f'Trainable Params: {sum(p.numel() for p in model.parameters() if p.requires_grad)}')
