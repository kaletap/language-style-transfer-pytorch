import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        pass

    def forward(self, x):
        pass


class Generator(nn.Module):
    """Decoder"""
    def __init__(self):
        super().__init__()
        pass

    def forward(self, x):
        pass


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        pass

    def forward(self, x):
        pass


class StyleTransferModel:
    def __init__(self, args, vocab):
        super().__init__()
        dim_y = args.dim_y
        dim_z = args.dim_z
        dim_h = dim_y + dim_z
        dim_emb = args.dim_emb
        n_layers = args.n_layers
        max_len = args.max_seq_length
        filter_sizes = [int(x) for x in args.filter_sizes.split(',')]
        n_filters = args.n_filters
        beta1, beta2 = 0.5, 0.999
        grad_clip = 30.0

        self.dropout = nn.Dropout(p=0.5)
        self.learning_rate = 0.0001
        self.rho = None  # TODO: check what it is
        self.gamma = 0.001

    def forward(self, x):
        pass
