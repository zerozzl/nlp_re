import torch
from torch import nn
from torch.nn.utils import rnn
import torch.nn.functional as F


class MulAttCNN(nn.Module):
    def __init__(self, num_rela, vocab_size, embed_size, max_inp_len, pf_embed_size,
                 conv_kernel_num, conv_kernel_size, inp_dropout_rate, hid_dropout_rate,
                 embed_fix=False, use_bigram=False, bigram_vocab_size=0, bigram_embed_size=0):
        super(MulAttCNN, self).__init__()
        self.num_rela = num_rela
        self.use_bigram = use_bigram
        self.max_inp_len = max_inp_len
        self.loss_margin = 1

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

        self.pos_embedding = nn.Embedding(max_inp_len * 2, pf_embed_size)
        self.rela_embedding = nn.Parameter(torch.randn(num_rela, conv_kernel_num))

        self.sent_cnn = nn.Conv1d(in_channels=(inp_size + pf_embed_size * 2) * 3,
                                  out_channels=conv_kernel_num, kernel_size=conv_kernel_size,
                                  padding=conv_kernel_size // 2)
        self.rela_attn_linear = nn.Linear(conv_kernel_num, num_rela, bias=False)

        self.in_dropout = nn.Dropout(inp_dropout_rate)
        self.hid_dropout = nn.Dropout(hid_dropout_rate)

    def init_embedding(self, pretrained_embeddings):
        self.embedding.weight.data.copy_(torch.from_numpy(pretrained_embeddings))

    def init_bigram_embedding(self, pretrained_embeddings):
        self.bigram_embedding.weight.data.copy_(torch.from_numpy(pretrained_embeddings))

    def input_attention(self, word_features, subject_idx, object_idx):
        subject_features = []
        object_features = []
        for idx in range(len(subject_idx)):
            subject_start = subject_idx[idx][0]
            subject_end = subject_idx[idx][1]
            subject_entity = word_features[idx, subject_start:subject_end + 1, :]
            subject_entity = torch.mean(subject_entity, dim=0)
            subject_features.append(subject_entity)

            object_start = object_idx[idx][0]
            object_end = object_idx[idx][1]
            object_entity = word_features[idx, object_start:object_end + 1, :]
            object_entity = torch.mean(object_entity, dim=0)
            object_features.append(object_entity)

        subject_features = torch.stack(subject_features, dim=0).unsqueeze(2)
        object_features = torch.stack(object_features, dim=0).unsqueeze(2)

        rs = torch.bmm(word_features, subject_features).squeeze(2)
        ro = torch.bmm(word_features, object_features).squeeze(2)
        rs = F.softmax(rs, dim=1)
        ro = F.softmax(ro, dim=1)

        rso = (rs + ro) / 2
        word_features = word_features * rso.unsqueeze(2)
        return word_features

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

        pos_features = torch.cat([self.pos_embedding(subject_dis), self.pos_embedding(object_dis)], dim=2)
        word_features = torch.cat([embed, pos_features], dim=2)

        features_pad = torch.zeros((word_features.size(0), 1, word_features.size(2)), requires_grad=False)
        features_pad = features_pad.to(word_features.device)
        word_features = torch.cat((features_pad, word_features, features_pad), dim=1)
        word_features = [
            torch.cat((word_features[:, i - 1, :], word_features[:, i, :], word_features[:, i + 1, :]), dim=1)
            for i in range(1, word_features.size(1) - 1)]
        word_features = torch.stack(word_features, dim=1)

        sent_features = self.input_attention(word_features, subject_idx, object_idx)
        sent_features = sent_features.permute(0, 2, 1)
        sent_features = self.sent_cnn(sent_features)
        sent_features = torch.tanh(sent_features)
        sent_features = sent_features.permute(0, 2, 1)
        sent_features = self.hid_dropout(sent_features)

        rela_attn = self.rela_attn_linear(sent_features)
        rela_attn = torch.matmul(rela_attn, self.rela_embedding)
        rela_attn = F.softmax(rela_attn, dim=1)

        wo = torch.mul(sent_features, rela_attn)
        wo = wo.permute(0, 2, 1)
        wo = F.max_pool1d(wo, kernel_size=wo.shape[2]).squeeze(2)
        wo = F.normalize(wo)

        wo_rep = wo.unsqueeze(1).repeat(1, self.num_rela, 1)
        all_rela = torch.eye(self.num_rela)
        all_rela = all_rela.to(self.rela_embedding.device)
        all_rela_emb = torch.mm(all_rela, self.rela_embedding)
        all_dist = torch.norm(wo_rep - all_rela_emb, 2, 2)

        if decode:
            predict = torch.min(all_dist, 1)[1]
            return predict
        else:
            in_rela = torch.eye(self.num_rela)[tags, :]
            in_rela = in_rela.to(self.rela_embedding.device)
            tgt_rela_emb = torch.mm(in_rela, self.rela_embedding)

            tgt_dist = torch.norm(wo - tgt_rela_emb, 2, 1)

            masking = torch.mul(in_rela, 10000)
            neg_dist = torch.min(torch.add(all_dist, masking), 1)[0]
            loss = torch.mean(self.loss_margin + tgt_dist - neg_dist)
            return loss
