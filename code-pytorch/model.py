import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, input_size, latent_dim, num_layers=1):
        super().__init__()
        self.encode = nn.GRU(input_size, latent_dim, num_layers=num_layers, batch_first=True)

    def forward(self, x):
        h, _ = self.encode(x)
        z = h[:, -1, :]  # taking last state of a hidden representations
        return z


class Generator(nn.Module):
    """Decoder"""
    def __init__(self, input_size, latent_dim, num_layers=1):
        super().__init__()
        self.decode = nn.GRU(input_size, latent_dim, num_layers=num_layers, batch_first=True)

    def forward(self, x, z):
        """
        :param x: either a real sequence or soft output distribution with temperature gamma
        :param z: latent content representation
        :return: transferred hidden states sequence
        """
        hs, _ = self.decode(x, z)
        return hs


class Discriminator(nn.Module):
    def __init__(self, max_seq_len, kernel_size, hidden_dim):
        super().__init__()
        self.max_seq_len = max_seq_len
        assert kernel_size % 2 == 0
        padding = (kernel_size - 1) / 2
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels=1, out_channels=128, kernel_size=(kernel_size, hidden_dim), stride=(1, 0),
                padding=(padding, 0)
            ),
            nn.ReLU()  # TODO: think about max pooling and how to do it
        )
        self.linear = nn.Linear(128*max_seq_len, 1)

    def pad(self, x):
        pass

    def forward(self, x):
        x = self.pad(x)
        x = self.conv(x)
        x = self.linear(x)
        return x


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
