import torch
from torch import nn
from torch.nn.utils import rnn
import torch.nn.functional as F
from torchcrf import CRF


class NovelTag(nn.Module):
    def __init__(self, num_tags, vocab_size, embed_size, hidden_size, inp_dropout_rate, hid_dropout_rate, loss_bias,
                 embed_fix=False, use_bigram=False, bigram_vocab_size=0, bigram_embed_size=0, use_crf=False):
        super(NovelTag, self).__init__()
        self.use_bigram = use_bigram
        self.use_crf = use_crf

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
        self.decoder = nn.LSTM(input_size=hidden_size * 2,
                               hidden_size=hidden_size * 2,
                               num_layers=1,
                               dropout=hid_dropout_rate,
                               batch_first=True,
                               bidirectional=False)
        self.linear = nn.Linear(hidden_size * 2, num_tags)
        self.in_dropout = nn.Dropout(inp_dropout_rate)

        if use_crf:
            self.crf = CRF(num_tags, batch_first=True)
        else:
            loss_weights = torch.zeros(num_tags) + loss_bias
            loss_weights[0] = 1
            self.ce_loss = nn.CrossEntropyLoss(weight=loss_weights)

    def init_embedding(self, pretrained_embeddings):
        self.embedding.weight.data.copy_(torch.from_numpy(pretrained_embeddings))

    def init_bigram_embedding(self, pretrained_embeddings):
        self.bigram_embedding.weight.data.copy_(torch.from_numpy(pretrained_embeddings))

    def forward(self, context, bigram=None, decode=True, tags=None):
        context_len = [len(ctx) for ctx in context]
        context = rnn.pad_sequence(context, batch_first=True)
        bigram = rnn.pad_sequence(bigram, batch_first=True)
        masks = context > 0

        if tags is not None:
            tags = rnn.pad_sequence(tags, batch_first=True)

        embed = self.embedding(context)
        if self.use_bigram:
            embed_bi = torch.cat(
                [self.bigram_embedding(bigram[:, :, i]) for i in range(bigram.size()[2])], dim=2)
            embed = torch.cat((embed, embed_bi), dim=2)
        embed = self.in_dropout(embed)

        rep = rnn.pack_padded_sequence(embed, context_len, batch_first=True)
        rep, _ = self.encoder(rep)
        rep, _ = self.decoder(rep)
        rep, _ = rnn.pad_packed_sequence(rep, batch_first=True)

        scores = self.linear(rep)

        if decode:
            if self.use_crf:
                pred = self.crf.decode(scores, masks)
            else:
                out = F.softmax(scores, dim=2)
                out = torch.argmax(out, dim=2)
                pred = out.cpu().numpy()
            return pred
        else:
            if self.use_crf:
                loss = -self.crf(scores, tags, masks)
            else:
                out_shape = scores.size()
                loss = self.ce_loss(scores.reshape(out_shape[0] * out_shape[1], out_shape[2]),
                                    tags.reshape(out_shape[0] * out_shape[1]))
            return loss
