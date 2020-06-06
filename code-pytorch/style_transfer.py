import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import random
import time

from options import load_arguments
from file_io import load_sent
from vocab import build_vocab, Vocabulary
from model import Encoder, Generator, Discriminator
from utils import get_batches


def gumbel_softmax(logits, gamma, eps=1e-20):
    u = torch.rand_like(logits)
    g = -torch.log(-torch.log(u + eps) + eps)
    return F.softmax((logits + g) / gamma)


def softsample_word(proj_w, proj_b, embedding, gamma=0.1):
    def loop_func(output):
        # TODO: add dropout
        logits = output @ proj_w + proj_b
        prob = gumbel_softmax(logits, gamma)
        inp = prob @ embedding.weight
        return inp, logits
    return loop_func


def softmax_word(proj_w, proj_b, embedding, gamma=0.1):
    """Useless here"""
    def loop_func(output):
        # TODO: add dropout
        logits = output @ proj_w + proj_b
        prob = F.softmax(logits / gamma)
        inp = prob @ embedding.weight  # multiplying by embedding matrix (soft sampling)
        return inp, logits
    return loop_func


def argmax_word(proj_w, proj_b, embedding):
    def loop_func(output):
        # TODO: add dropout
        logits = output @ proj_w + proj_b
        word = torch.argmax(logits, 1)
        inp = embedding(word).float()
        return inp, logits
    return loop_func


def rnn_decode(h, inp, length, generator, loop_func):
    """
    :param h: initial hidden state
    :param inp: initial input  # is it number or a vector?
    :param length: length of a sequence
    :param generator:
    :param loop_func: function of a hidden state producing output to be fed later on
    :return: unrolled hidden states sequence (from first to second last state) with input given by loop_func
    """
    h_seq, logits_seq = [], []
    # TODO: refactor a little bit, this code looks like shit, let's think how to improve it
    output = h  # hack to make dimensions work (we want final h_seq to be tensor of dims batch_size x length+1 x dim_z)
    for t in range(length + 1):
        h_seq.append(output.view(-1, 1, dim_z))
        output, h = generator(inp, h)  # first element is batch_size x 1 (number of timesteps) x dim_z,
        # second is 1 (num_layers * num_directions) x batch_size x dim_z
        # (warning: in tf it would be just batch_size x dim_z)
        # Note: both elements in tuple have exactly the same elements, but have different shapes
        inp, logits = loop_func(output)
        logits_seq.append(logits)
    return torch.cat(h_seq, 1), torch.cat(logits_seq, 1)


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
    embedding = nn.Embedding.from_pretrained(torch.tensor(vocab.embedding).float(), freeze=False)

    if args.dev:
        dev0 = load_sent(args.dev + '.0')
        dev1 = load_sent(args.dev + '.1')

    if args.test:
        test0 = load_sent(args.test + '.0')
        test1 = load_sent(args.test + '.1')

    lr = args.learning_rate
    gamma = args.gamma_init
    dim_z = args.dim_z
    dim_emb = args.dim_emb
    max_seq_length = args.max_seq_length

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print("Using device", device)

    encoder1 = Encoder(args.dim_emb, args.dim_z)
    generator1 = Generator(args.dim_emb, args.dim_z)  # generates samples from space1 to space2
    discriminator1 = Discriminator(hidden_dim=args.dim_z)

    encoder2 = Encoder(args.dim_emb, args.dim_z)
    generator2 = Generator(args.dim_emb, args.dim_z)  # generates samples from space2 to space1
    discriminator2 = Discriminator(hidden_dim=args.dim_z)

    # Training loop
    if args.train:
        print("Creating batches from sentences...")
        batch_size = args.batch_size
        batches, _, _ = get_batches(train0, train1, vocab.word2id,
                                    args.batch_size, noisy=True)
        random.shuffle(batches)
        print("Done")

        start_time = time.time()
        step = 0
        best_dev = float("inf")

        zeros = torch.zeros(batch_size).float()
        ones = torch.ones(batch_size).float()

        for epoch in range(1, 1 + args.max_epochs):
            print("--------------------epoch {}--------------------".format(epoch))
            print("learning_rate:", lr, "  gamma:", gamma)

            for batch in batches:
                print(batch.keys())
                enc_inputs = embedding(torch.tensor(batch["enc_inputs"])).float()
                dec_inputs = embedding(torch.tensor(batch["dec_inputs"])).float()  # have <eos> at the end
                target = torch.tensor(batch["targets"])
                half = enc_inputs.shape[0] // 2

                # Auto encoder
                z1 = encoder1(enc_inputs[:half])  # batch_size x dim_z
                z2 = encoder2(enc_inputs[half:])  # batch_size x dim_z

                def expand(z):
                    return z.view(1, half, dim_z)

                # batch_size x seq_len x dim_z
                # TODO: make sure index og g1 and g2 is correct
                g1_outputs, _ = generator1(dec_inputs[:batch_size], expand(z1))
                g2_outputs, _ = generator2(dec_inputs[batch_size:], expand(z2))

                # Proper hidden states (used for professor forcing), attached h_0 in the front
                teach_h2 = torch.cat([z2.view(half, 1, args.dim_z), g1_outputs], 1)
                teach_h1 = torch.cat([z1.view(half, 1, args.dim_z), g2_outputs], 1)

                # TODO: add dropout to g_outputs

                # projection matrix shared between two generators (TODO: is it useful?)
                proj_w = torch.rand([dim_z, vocab.size], dtype=torch.float).requires_grad_()
                proj_b = torch.rand([vocab.size], dtype=torch.float).requires_grad_()

                g1_outputs = g1_outputs.reshape(-1, dim_z)
                g2_outputs = g1_outputs.reshape(-1, dim_z)
                g1_logits = g1_outputs @ proj_w + proj_b
                g2_logits = g2_outputs @ proj_w + proj_b

                criterion_rec = nn.NLLLoss()
                loss_rec1 = criterion_rec(g1_logits, target[:half].view(-1))
                loss_rec2 = criterion_rec(g2_logits, target[half:].view(-1))
                loss_rec = (loss_rec1 + loss_rec2) / batch_size

                # Feeding previous decoding
                go = dec_inputs[:, 0, :].view(-1, 1, dim_emb)  # <go> token, making it a sequence of 1 to pass into
                # generator
                go1, go2 = go[:half], go[half:]  # it is the same anyway (it's the same embedding). (go1 == go2).all()
                soft_func = softsample_word(proj_w, proj_b, embedding, gamma)

                max_len = dec_inputs.shape[1]
                # h_ori is z in my case (z1 and z2). Byty przestają być mnożone. Różnice w generatorach
                soft_h2, soft_logits2 = rnn_decode(expand(z1), go1, max_len, generator1, soft_func)
                soft_h1, soft_logits1 = rnn_decode(expand(z2), go2, max_len, generator2, soft_func)

                # Discriminator
                print(soft_h2.shape, teach_h2.shape, half, max_len)

                break
