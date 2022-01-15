import torch
from torch import nn
from torch.nn.utils import rnn
import torch.nn.functional as F


class CNN(nn.Module):
    def __init__(self, num_tags, vocab_size, embed_size, max_inp_len, pf_embed_size, conv_kernel_num, conv_kernel_size,
                 linear_hidden_size, inp_dropout_rate, hid_dropout_rate,
                 embed_fix=False, use_bigram=False, bigram_vocab_size=0, bigram_embed_size=0):
        super(CNN, self).__init__()
        self.use_bigram = use_bigram
        self.max_inp_len = max_inp_len

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

        # self.token_cnn = nn.Conv1d(in_channels=inp_size, out_channels=inp_size, kernel_size=3, padding=1)

        self.pos_embedding = nn.Embedding(max_inp_len * 2, pf_embed_size)

        self.sent_cnn = nn.Conv1d(in_channels=inp_size * 3 + pf_embed_size * 2,
                                  out_channels=conv_kernel_num, kernel_size=conv_kernel_size,
                                  padding=conv_kernel_size // 2)
        self.sent_linear = nn.Linear(conv_kernel_num, linear_hidden_size)

        self.com_linear = nn.Linear(inp_size * 3 * 2 + linear_hidden_size, num_tags)

        self.in_dropout = nn.Dropout(inp_dropout_rate)
        self.hid_dropout = nn.Dropout(hid_dropout_rate)
        self.ce_loss = nn.CrossEntropyLoss()

    def init_embedding(self, pretrained_embeddings):
        self.embedding.weight.data.copy_(torch.from_numpy(pretrained_embeddings))

    def init_bigram_embedding(self, pretrained_embeddings):
        self.bigram_embedding.weight.data.copy_(torch.from_numpy(pretrained_embeddings))

    def forward(self, context, subject_idx, object_idx, subject_dis, object_dis, bigram=None, decode=True, tags=None):
        context = rnn.pad_sequence(context, batch_first=True)
        bigram = rnn.pad_sequence(bigram, batch_first=True)

        subject_dis = [item + self.max_inp_len for item in subject_dis]
        object_dis = [item + self.max_inp_len for item in object_dis]
        subject_dis = rnn.pad_sequence(subject_dis, batch_first=True)
        object_dis = rnn.pad_sequence(object_dis, batch_first=True)

        embed = self.embedding(context)
        if self.use_bigram:
            embed_bi = torch.cat(
                [self.bigram_embedding(bigram[:, :, i]) for i in range(bigram.size()[2])], dim=2)
            embed = torch.cat((embed, embed_bi), dim=2)
        embed = self.in_dropout(embed)

        embed_pad = torch.zeros(embed.size(2), requires_grad=False)
        embed_pad = embed_pad.to(embed.device)

        subject_features = []
        object_features = []
        for idx in range(len(subject_idx)):
            subject_start = subject_idx[idx][0]
            subject_end = subject_idx[idx][1]
            subject_entity = embed[idx, subject_start:subject_end + 1, :]
            # subject_entity = subject_entity.permute(1, 0).unsqueeze(0)
            # subject_entity = self.token_cnn(subject_entity)
            # subject_entity = F.max_pool1d(subject_entity, subject_entity.size(2)).squeeze(0).squeeze(1)
            subject_entity = torch.mean(subject_entity, dim=0)
            subject_left = embed[idx, subject_start - 1, :] if subject_start >= 0 else embed_pad
            subject_right = embed[idx, subject_end + 1, :] if (subject_end + 1) < embed.size(1) else embed_pad
            subject_entity = torch.cat([subject_entity, subject_left, subject_right])
            subject_features.append(subject_entity)

            object_start = object_idx[idx][0]
            object_end = object_idx[idx][1]
            object_entity = embed[idx, object_start:object_end + 1, :]
            # object_entity = object_entity.permute(1, 0).unsqueeze(0)
            # object_entity = self.token_cnn(object_entity)
            # object_entity = F.max_pool1d(object_entity, object_entity.size(2)).squeeze(0).squeeze(1)
            object_entity = torch.mean(object_entity, dim=0)
            object_left = embed[idx, object_start - 1, :] if object_start >= 0 else embed_pad
            object_right = embed[idx, object_end + 1, :] if (object_end + 1) < embed.size(1) else embed_pad
            object_entity = torch.cat([object_entity, object_left, object_right])
            object_features.append(object_entity)

        subject_features = torch.stack(subject_features, dim=0)
        object_features = torch.stack(object_features, dim=0)

        embed_pad = torch.zeros((embed.size(0), 1, embed.size(2)), requires_grad=False)
        embed_pad = embed_pad.to(embed.device)
        embed = torch.cat((embed_pad, embed, embed_pad), dim=1)
        word_features = [torch.cat((embed[:, i - 1, :], embed[:, i, :], embed[:, i + 1, :]), dim=1)
                         for i in range(1, embed.size(1) - 1)]
        word_features = torch.stack(word_features, dim=1)

        pos_features = torch.cat([self.pos_embedding(subject_dis), self.pos_embedding(object_dis)], dim=2)
        sent_features = torch.cat([word_features, pos_features], dim=2)
        sent_features = self.hid_dropout(sent_features)

        sent_features = sent_features.permute(0, 2, 1)
        sent_features = self.sent_cnn(sent_features)
        sent_features = F.max_pool1d(sent_features, sent_features.size(2)).squeeze(2)
        sent_features = torch.tanh(self.sent_linear(sent_features))

        com_features = torch.cat([subject_features, object_features, sent_features], dim=1)
        scores = self.com_linear(com_features)

        if decode:
            out = F.softmax(scores, dim=1)
            out = torch.argmax(out, dim=1)
            pred = out.cpu().numpy()
            return pred
        else:
            loss = self.ce_loss(scores, tags)
            return loss
