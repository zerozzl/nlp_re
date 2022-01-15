import torch
from torch import nn
from torch.nn.utils import rnn
import torch.nn.functional as F


class AttentionLayer(nn.Module):
    def __init__(self, input_size):
        super(AttentionLayer, self).__init__()
        self.w = nn.Linear(input_size, 1)

    def forward(self, H):
        M = torch.tanh(H)
        attn = F.softmax(self.w(M), dim=1)
        r = attn.mul(H).sum(1)
        return r


class LSTM(nn.Module):
    def __init__(self, num_tags, vocab_size, embed_size, hidden_size, inp_dropout_rate, hid_dropout_rate,
                 embed_fix=False, use_bigram=False, bigram_vocab_size=0, bigram_embed_size=0):
        super(LSTM, self).__init__()
        self.use_bigram = use_bigram

        inp_size = embed_size
        self.embedding = nn.Embedding(vocab_size, embed_size)

        if use_bigram:
            inp_size += bigram_embed_size * 2
            self.bigram_embedding = nn.Embedding(bigram_vocab_size, bigram_embed_size)

        if embed_fix:
            for param in self.embedding.parameters():
                param.requires_grad = False

            if use_bigram:
                for param in self.bigram_embedding.parameters():
                    param.requires_grad = False

        self.encoder = nn.LSTM(input_size=inp_size,
                               hidden_size=hidden_size,
                               num_layers=1,
                               dropout=hid_dropout_rate,
                               batch_first=True,
                               bidirectional=True)

        self.attention = AttentionLayer(hidden_size * 2)
        self.linear = nn.Linear(hidden_size * 2, num_tags)
        self.in_dropout = nn.Dropout(inp_dropout_rate)
        self.hid_dropout = nn.Dropout(hid_dropout_rate)
        self.ce_loss = nn.CrossEntropyLoss()

    def init_embedding(self, pretrained_embeddings):
        self.embedding.weight.data.copy_(torch.from_numpy(pretrained_embeddings))

    def init_bigram_embedding(self, pretrained_embeddings):
        self.bigram_embedding.weight.data.copy_(torch.from_numpy(pretrained_embeddings))

    def forward(self, context, bigram=None, decode=True, tags=None):
        context_len = [len(ctx) for ctx in context]
        context = rnn.pad_sequence(context, batch_first=True)
        bigram = rnn.pad_sequence(bigram, batch_first=True)

        embed = self.embedding(context)
        if self.use_bigram:
            embed_bi = torch.cat(
                [self.bigram_embedding(bigram[:, :, i]) for i in range(bigram.size()[2])], dim=2)
            embed = torch.cat((embed, embed_bi), dim=2)
        embed = self.in_dropout(embed)

        rep = rnn.pack_padded_sequence(embed, context_len, batch_first=True)
        rep, _ = self.encoder(rep)
        rep, _ = rnn.pad_packed_sequence(rep, batch_first=True)

        rep = self.attention(rep)
        rep = torch.tanh(rep)
        rep = self.hid_dropout(rep)

        scores = self.linear(rep)

        if decode:
            out = F.softmax(scores, dim=1)
            out = torch.argmax(out, dim=1)
            pred = out.cpu().numpy()
            return pred
        else:
            loss = self.ce_loss(scores, tags)
            return loss
