import torch
from torch import nn
from torch.nn.utils import rnn
import torch.nn.functional as F


class BAG_ATT(nn.Module):
    def __init__(self, num_tags, vocab_size, embed_size, max_inp_len, pf_embed_size, conv_kernel_num, conv_kernel_size,
                 inp_dropout_rate, hid_dropout_rate, embed_fix=False,
                 use_bigram=False, bigram_vocab_size=0, bigram_embed_size=0, use_cpu=False):
        super(BAG_ATT, self).__init__()
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

        self.pos_embedding = nn.Embedding(max_inp_len * 2, pf_embed_size)

        self.mask_embedding = nn.Embedding(4, 3)
        self.mask_embedding.weight.data.copy_(torch.LongTensor(([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])))
        self.mask_embedding.weight.requires_grad = False

        self.rela_embedding = nn.Linear(conv_kernel_num * 3, num_tags)
        self.rela_diag = torch.ones(num_tags).diag().unsqueeze(0)
        self.rela_diag = self.rela_diag.cpu() if use_cpu else self.rela_diag.cuda()

        self.cnn = nn.Conv1d(in_channels=inp_size + pf_embed_size * 2,
                             out_channels=conv_kernel_num, kernel_size=conv_kernel_size,
                             padding=conv_kernel_size // 2)

        self.in_dropout = nn.Dropout(inp_dropout_rate)
        self.hid_dropout = nn.Dropout(hid_dropout_rate)
        self.ce_loss = nn.CrossEntropyLoss()

    def init_embedding(self, pretrained_embeddings):
        self.embedding.weight.data.copy_(torch.from_numpy(pretrained_embeddings))

    def init_bigram_embedding(self, pretrained_embeddings):
        self.bigram_embedding.weight.data.copy_(torch.from_numpy(pretrained_embeddings))

    def get_ins_feature(self, context, subject_dis, object_dis, pool_mask, bigram=None):
        context = rnn.pad_sequence(context, batch_first=True)
        pool_mask = rnn.pad_sequence(pool_mask, batch_first=True)
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

        pos_embed = torch.cat([self.pos_embedding(subject_dis), self.pos_embedding(object_dis)], dim=2)
        embed = torch.cat([embed, pos_embed], dim=2)
        embed = self.in_dropout(embed)

        features = embed.permute(0, 2, 1)
        features = self.cnn(features)
        features = features.unsqueeze(-1).permute(0, 2, 1, 3)
        mask_embed = self.mask_embedding(pool_mask).unsqueeze(-2) * 100
        features = features + mask_embed
        features = torch.max(features, 1)[0] - 100
        features = features.view(-1, features.size(1) * features.size(2))
        features = torch.tanh(features)
        features = self.hid_dropout(features)

        return features

    def get_bag_feature(self, features):
        alpha = self.rela_embedding(features)
        alpha = F.softmax(alpha, 0)
        bag_features = torch.matmul(alpha.t(), features)
        bag_features = self.hid_dropout(bag_features)
        return bag_features

    def forward(self, group_data, decode=True, tags=None):
        group_features = []
        for group_idx in range(len(group_data)):
            bags = group_data[group_idx]
            bags_features = []
            for bag_idx in range(len(bags[0])):
                context = bags[0][bag_idx]
                bigram = bags[1][bag_idx]
                subject_dis = bags[2][bag_idx]
                object_dis = bags[3][bag_idx]
                pool_masks = bags[4][bag_idx]
                bags_features.append(self.get_ins_feature(context, subject_dis, object_dis, pool_masks, bigram))
            group_features.append(bags_features)

        if decode:
            preds = []
            for idx in range(len(group_features)):
                bags_pred = []
                bags_features = group_features[idx]
                for bag_idx in range(len(bags_features)):
                    scores = self.rela_embedding(bags_features[bag_idx])
                    pred = F.softmax(scores, dim=1)
                    pred = torch.argmax(pred, dim=1)
                    bags_pred.append(pred)
                preds.append(bags_pred)
            return preds
        else:
            losses = []
            for idx in range(len(group_features)):
                label = tags[idx]
                bags_features = group_features[idx]
                for bag_idx in range(len(bags_features)):
                    features = self.get_bag_feature(bags_features[bag_idx])
                    bags_features[bag_idx] = features
                bags_features = torch.stack(bags_features, dim=0)

                if label == 0:
                    scores = torch.sum(self.rela_embedding(bags_features) * self.rela_diag, 2)
                else:
                    bags_features = bags_features.transpose(0, 1)
                    bags_features = bags_features / torch.norm(bags_features, 2, 2, keepdim=True)
                    crossatt = torch.matmul(bags_features, bags_features.transpose(1, 2))
                    crossatt = torch.sum(crossatt, 2)
                    crossatt = F.softmax(crossatt, 1)
                    bags_features = torch.matmul(crossatt.unsqueeze(1), bags_features).squeeze(1)
                    scores = self.rela_embedding(bags_features).diag().unsqueeze(0)

                label = label.repeat(len(scores))
                loss = self.ce_loss(scores, label)
                losses.append(loss)

            losses = torch.stack(losses).mean()
            return losses
