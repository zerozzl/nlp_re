import torch
from torch import nn
from torch.nn.utils import rnn
import torch.nn.functional as F
from transformers import BertConfig, BertModel
import numpy as np


class Casrel(nn.Module):
    def __init__(self, config_path, model_path, num_rela, bert_freeze, threshold=0.5):
        super(Casrel, self).__init__()
        self.threshold = threshold

        config = BertConfig.from_json_file(config_path)
        self.embedding = BertModel.from_pretrained(model_path, config=config)
        self.subject_head_linear = nn.Linear(config.hidden_size, 1)
        self.subject_tail_linear = nn.Linear(config.hidden_size, 1)
        self.object_head_linear = nn.Linear(config.hidden_size, num_rela)
        self.object_tail_linear = nn.Linear(config.hidden_size, num_rela)

        if bert_freeze:
            for param in self.embedding.parameters():
                param.requires_grad = False

    def get_objects(self, context, subject_head, subject_tail):
        subject_head = subject_head.unsqueeze(-2)
        subject_tail = subject_tail.unsqueeze(-2)

        subject_head = torch.matmul(subject_head, context)
        subject_tail = torch.matmul(subject_tail, context)
        subject = (subject_head + subject_tail) / 2
        context_subject = context + subject

        pred_object_heads = self.object_head_linear(context_subject)
        pred_object_tails = self.object_tail_linear(context_subject)

        return pred_object_heads, pred_object_tails

    def get_loss(self, gold, pred, mask):
        loss = F.binary_cross_entropy_with_logits(pred, gold, reduction='none')
        if loss.shape != mask.shape:
            mask = mask.unsqueeze(-1)
        loss = torch.sum(loss * mask) / torch.sum(mask)
        return loss

    def forward(self, tokens, segments, masks, decode=True, all_subject_heads=None, all_subject_tails=None,
                subject_head=None, subject_tail=None, object_heads=None, object_tails=None):
        tokens = rnn.pad_sequence(tokens, batch_first=True)
        segments = rnn.pad_sequence(segments, batch_first=True)
        masks = rnn.pad_sequence(masks, batch_first=True)

        if not decode:
            all_subject_heads = rnn.pad_sequence(all_subject_heads, batch_first=True).float()
            all_subject_tails = rnn.pad_sequence(all_subject_tails, batch_first=True).float()
            subject_head = rnn.pad_sequence(subject_head, batch_first=True).float()
            subject_tail = rnn.pad_sequence(subject_tail, batch_first=True).float()
            object_heads = rnn.pad_sequence(object_heads, batch_first=True).float()
            object_tails = rnn.pad_sequence(object_tails, batch_first=True).float()

        context = self.embedding(input_ids=tokens, token_type_ids=segments, attention_mask=masks)
        context = context.last_hidden_state

        pred_subject_heads = self.subject_head_linear(context).squeeze(2)
        pred_subject_tails = self.subject_tail_linear(context).squeeze(2)

        if decode:
            pred_subject_heads = torch.sigmoid(pred_subject_heads) * masks
            pred_subject_tails = torch.sigmoid(pred_subject_tails) * masks

            triples = []
            for ex_idx in range(len(pred_subject_heads)):
                single_context = context[ex_idx]
                subject_heads = pred_subject_heads[ex_idx]
                subject_tails = pred_subject_tails[ex_idx]

                subject_heads = np.where(subject_heads.cpu() > self.threshold)[0]
                subject_tails = np.where(subject_tails.cpu() > self.threshold)[0]

                s2o = {}
                if (len(subject_heads) == 0) or (len(subject_tails) == 0):
                    triples.append(s2o)
                    continue

                for subject_head in subject_heads:
                    subject_tail = subject_tails[subject_tails >= subject_head]

                    if len(subject_tail) <= 0:
                        continue

                    subject_tail = subject_tail[0]
                    subject = (subject_head, subject_tail)

                    subject_head_inp = torch.zeros(single_context.shape[0]).to(single_context.device)
                    subject_tail_inp = torch.zeros(single_context.shape[0]).to(single_context.device)
                    subject_head_inp[subject_head] = 1
                    subject_tail_inp[subject_tail] = 1

                    object_heads, object_tails = self.get_objects(single_context, subject_head_inp, subject_tail_inp)
                    object_heads = torch.sigmoid(object_heads)
                    object_tails = torch.sigmoid(object_tails)

                    object_heads = np.where(object_heads.cpu() > self.threshold)
                    object_tails = np.where(object_tails.cpu() > self.threshold)

                    for object_head, rela_head in zip(*object_heads):
                        for object_tail, rela_tail in zip(*object_tails):
                            if rela_head == rela_tail and object_head <= object_tail:
                                if subject not in s2o:
                                    s2o[subject] = []
                                s2o[subject].append((object_head, object_tail, rela_head))
                                break

                triples.append(s2o)
            return triples
        else:
            pred_object_heads, pred_object_tails = self.get_objects(context, subject_head, subject_tail)

            subject_heads_loss = self.get_loss(all_subject_heads, pred_subject_heads, masks)
            subject_tails_loss = self.get_loss(all_subject_tails, pred_subject_tails, masks)
            object_heads_loss = self.get_loss(object_heads, pred_object_heads, masks)
            object_tails_loss = self.get_loss(object_tails, pred_object_tails, masks)

            loss = subject_heads_loss + subject_tails_loss + object_heads_loss + object_tails_loss
            return loss
