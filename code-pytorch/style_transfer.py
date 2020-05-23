import torch
import os

from options import load_arguments
from file_io import load_sent
from vocab import build_vocab, Vocabulary
from model import Encoder, Generator, Discriminator


if __name__ == '__main__':
    args = load_arguments()

    #  data preparation
    if args.train:
        train0 = load_sent(args.train + '.0', args.max_train_size)
        train1 = load_sent(args.train + '.1', args.max_train_size)
        print(('#sents of training file 0:', len(train0)))
        print(('#sents of training file 1:', len(train1)))

        if not os.path.isfile(args.vocab):
            build_vocab(train0 + train1, args.vocab)

    vocab = Vocabulary(args.vocab, args.embedding, args.dim_emb)
    print(('vocabulary size:', vocab.size))

    if args.dev:
        dev0 = load_sent(args.dev + '.0')
        dev1 = load_sent(args.dev + '.1')

    if args.test:
        test0 = load_sent(args.test + '.0')
        test1 = load_sent(args.test + '.1')

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print("Using device", device)

    encoder = Encoder(args.dim_emb, args.dim_z)
    generator = Generator(args.dim_emb, args.dim_z)
    discriminator = Discriminator(hidden_dim=args.dim_y)
