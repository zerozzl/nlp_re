import torch
from torch import nn
from torch.nn.utils import rnn
import torch.nn.functional as F
from torchcrf import CRF
import numpy as np


class MultiHeadSelection(nn.Module):
    def __init__(self, num_rela, num_ner_tag, vocab_size, embed_size, ner_embed_size, hidden_size,
                 inp_dropout_rate, hid_dropout_rate, rela_threshold=0.5, embed_fix=False,
                 use_bigram=False, bigram_vocab_size=0, bigram_embed_size=0, use_crf=False):
        super(MultiHeadSelection, self).__init__()
        self.num_rela = num_rela
        self.use_bigram = use_bigram
        self.use_crf = use_crf
        self.rela_threshold = rela_threshold

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

        self.ner_embedding = nn.Embedding(num_ner_tag, ner_embed_size)

        self.encoder = nn.LSTM(input_size=inp_size,
                               hidden_size=hidden_size,
                               num_layers=1,
                               dropout=hid_dropout_rate,
                               batch_first=True,
                               bidirectional=True)

        self.in_dropout = nn.Dropout(inp_dropout_rate)

        # ner
        self.ner_hid = nn.Linear(hidden_size * 2, hidden_size)
        self.ner_out = nn.Linear(hidden_size, num_ner_tag)

        if use_crf:
            self.ner_crf = CRF(num_ner_tag, batch_first=True)
        else:
            self.ner_loss = nn.CrossEntropyLoss()

        # multi head select
        self.select_u = nn.Linear(hidden_size * 2 + ner_embed_size, hidden_size)
        self.select_w = nn.Linear(hidden_size * 2 + ner_embed_size, hidden_size)
        self.select_v = nn.Linear(hidden_size * 2, num_rela)

    def init_embedding(self, pretrained_embeddings):
        self.embedding.weight.data.copy_(torch.from_numpy(pretrained_embeddings))

    def init_bigram_embedding(self, pretrained_embeddings):
        self.bigram_embedding.weight.data.copy_(torch.from_numpy(pretrained_embeddings))

    def forward(self, context, bigram=None, decode=True, ner_tags=None, rela_links=None):
        context_len = [len(ctx) for ctx in context]
        context = rnn.pad_sequence(context, batch_first=True)
        bigram = rnn.pad_sequence(bigram, batch_first=True)
        masks = context > 0
        select_mask = (masks.unsqueeze(2) * masks.unsqueeze(1)).unsqueeze(3).expand(-1, -1, -1, self.num_rela)

        if ner_tags is not None:
            ner_tags = rnn.pad_sequence(ner_tags, batch_first=True)
        if rela_links is not None:
            for link_idx in range(1, len(rela_links)):
                link = rela_links[link_idx]
                pad = torch.zeros((link.shape[0], context_len[0] - link.shape[1], link.shape[2]), device=link.device)
                link = torch.cat([link, pad], dim=1)
                pad = torch.zeros((context_len[0] - link.shape[0], link.shape[1], link.shape[2]), device=link.device)
                link = torch.cat([link, pad], dim=0)
                rela_links[link_idx] = link
            rela_links = rnn.pad_sequence(rela_links, batch_first=True)
            rela_links = rela_links.float()

        embed = self.embedding(context)
        if self.use_bigram:
            embed_bi = torch.cat(
                [self.bigram_embedding(bigram[:, :, i]) for i in range(bigram.size()[2])], dim=2)
            embed = torch.cat((embed, embed_bi), dim=2)
        embed = self.in_dropout(embed)

        rep = rnn.pack_padded_sequence(embed, context_len, batch_first=True)
        rep, _ = self.encoder(rep)
        rep, _ = rnn.pad_packed_sequence(rep, batch_first=True)

        ner_score = self.ner_hid(rep)
        ner_score = torch.tanh(ner_score)
        ner_score = self.ner_out(ner_score)

        if decode:
            if self.use_crf:
                ner_pred = self.ner_crf.decode(ner_score, masks)
                ner_pred = [torch.LongTensor(np.array(p)) for p in ner_pred]
                ner_pred = [item.to(ner_score.device) for item in ner_pred]
                ner_pred = rnn.pad_sequence(ner_pred, batch_first=True)
            else:
                ner_pred = F.softmax(ner_score, dim=2)
                ner_pred = torch.argmax(ner_pred, dim=2)

            ner_embed = self.ner_embedding(ner_pred)
            multi_head_inp = torch.cat([rep, ner_embed], dim=2)
            B, L, H = multi_head_inp.size()
            multi_head_u = torch.tanh(self.select_u(multi_head_inp)).unsqueeze(1).expand(B, L, L, -1)
            multi_head_w = torch.tanh(self.select_w(multi_head_inp)).unsqueeze(2).expand(B, L, L, -1)
            rela_score = self.select_v(torch.cat((multi_head_u, multi_head_w), dim=-1))
            rela_pred = (torch.sigmoid(rela_score) * select_mask.float()) > self.rela_threshold
            rela_pred = [rela_pred[idx][:context_len[idx], :context_len[idx], :].int() for idx in range(len(rela_pred))]
            return ner_pred, rela_pred
        else:
            if self.use_crf:
                ner_loss = -self.ner_crf(ner_score, ner_tags, masks)
            else:
                out_shape = ner_score.size()
                ner_loss = self.ner_loss(ner_score.reshape(out_shape[0] * out_shape[1], out_shape[2]),
                                         ner_tags.reshape(out_shape[0] * out_shape[1]))

            ner_embed = self.ner_embedding(ner_tags)
            multi_head_inp = torch.cat([rep, ner_embed], dim=2)
            B, L, H = multi_head_inp.size()
            multi_head_u = torch.tanh(self.select_u(multi_head_inp)).unsqueeze(1).expand(B, L, L, -1)
            multi_head_w = torch.tanh(self.select_w(multi_head_inp)).unsqueeze(2).expand(B, L, L, -1)
            rela_score = self.select_v(torch.cat((multi_head_u, multi_head_w), dim=-1))

            rela_loss = F.binary_cross_entropy_with_logits(rela_score, rela_links, reduction='none')
            rela_loss = rela_loss.masked_select(select_mask).sum()
            rela_loss /= masks.sum()

            total_loss = ner_loss + rela_loss
            return total_loss
