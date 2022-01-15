import os
import logging
from argparse import ArgumentParser
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch import optim
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel
from transformers import BertTokenizer

from joint_casrel.model import Casrel
from utils.dataloader import CasrelDataset, load_relation_schema
from utils.logger import Logger
from utils import model_utils


def get_dataset(args, rela_to_id, tokenizer):
    train_dataset = CasrelDataset(args.task, '%s/%s/train.txt' % (args.data_path, args.task), rela_to_id, tokenizer,
                                  do_train=True, debug=args.debug)
    test_dataset = CasrelDataset(args.task, '%s/%s/dev.txt' % (args.data_path, args.task), rela_to_id, tokenizer,
                                 debug=args.debug)

    return train_dataset, test_dataset


def data_collate_fn(data):
    data = np.array(data)

    tokens = data[:, 0].tolist()
    tokens = [torch.LongTensor(np.array(s)) for s in tokens]

    segment_ids = data[:, 1].tolist()
    segment_ids = [torch.LongTensor(np.array(s)) for s in segment_ids]

    mask_ids = data[:, 2].tolist()
    mask_ids = [torch.LongTensor(np.array(s)) for s in mask_ids]

    context_len = data[:, 3].tolist()

    all_subject_heads = data[:, 4].tolist()
    all_subject_heads = [torch.LongTensor(np.array(s)) for s in all_subject_heads]

    all_subject_tails = data[:, 5].tolist()
    all_subject_tails = [torch.LongTensor(np.array(s)) for s in all_subject_tails]

    subject_head = data[:, 6].tolist()
    subject_head = [torch.LongTensor(np.array(s)) for s in subject_head]

    subject_tail = data[:, 7].tolist()
    subject_tail = [torch.LongTensor(np.array(s)) for s in subject_tail]

    object_heads = data[:, 8].tolist()
    object_heads = [torch.LongTensor(np.array(s)) for s in object_heads]

    object_tails = data[:, 9].tolist()
    object_tails = [torch.LongTensor(np.array(s)) for s in object_tails]

    s2o = data[:, 10].tolist()

    return tokens, segment_ids, mask_ids, context_len, all_subject_heads, all_subject_tails, \
           subject_head, subject_tail, object_heads, object_tails, s2o


def train(args, dataset, dataloader, model, optimizer, lr_scheduler):
    model.train()
    loss_sum = 0
    for batch, data in enumerate(dataloader):
        optimizer.zero_grad()
        tokens, segment_ids, mask_ids, _, all_subject_heads, all_subject_tails, \
        subject_head, subject_tail, object_heads, object_tails, _ = data

        tokens = [item.cpu() if args.use_cpu else item.cuda() for item in tokens]
        segment_ids = [item.cpu() if args.use_cpu else item.cuda() for item in segment_ids]
        mask_ids = [item.cpu() if args.use_cpu else item.cuda() for item in mask_ids]
        all_subject_heads = [item.cpu() if args.use_cpu else item.cuda() for item in all_subject_heads]
        all_subject_tails = [item.cpu() if args.use_cpu else item.cuda() for item in all_subject_tails]
        subject_head = [item.cpu() if args.use_cpu else item.cuda() for item in subject_head]
        subject_tail = [item.cpu() if args.use_cpu else item.cuda() for item in subject_tail]
        object_heads = [item.cpu() if args.use_cpu else item.cuda() for item in object_heads]
        object_tails = [item.cpu() if args.use_cpu else item.cuda() for item in object_tails]

        loss = model(tokens, segment_ids, mask_ids, decode=False,
                     all_subject_heads=all_subject_heads, all_subject_tails=all_subject_tails,
                     subject_head=subject_head, subject_tail=subject_tail,
                     object_heads=object_heads, object_tails=object_tails)
        loss = loss.mean()

        loss_sum += loss.item()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

    loss_sum = loss_sum / len(dataset)
    return loss_sum


def evaluate(args, dataloader, model):
    pred_answers = []
    gold_answers = []

    model.eval()

    if args.multi_gpu:
        model = model.module

    with torch.no_grad():
        for batch, data in enumerate(dataloader):
            tokens, segment_ids, mask_ids, _, _, _, _, _, _, _, s2o = data

            tokens = [item.cpu() if args.use_cpu else item.cuda() for item in tokens]
            segment_ids = [item.cpu() if args.use_cpu else item.cuda() for item in segment_ids]
            mask_ids = [item.cpu() if args.use_cpu else item.cuda() for item in mask_ids]

            pred_s2o = model(tokens, segment_ids, mask_ids)

            gold_answers.extend(s2o)
            pred_answers.extend(pred_s2o)

    pre, rec, f1 = calc_measure(gold_answers, pred_answers)
    return pre, rec, f1


def calc_measure(gold_s2os, pred_s2os):
    tp = 0
    tp_fp = 0
    tp_fn = 0
    for i in range(len(gold_s2os)):
        gold_s2o = gold_s2os[i]
        pred_s2o = pred_s2os[i]

        for s2o in gold_s2o:
            tp_fn += len(gold_s2o[s2o])
        for s2o in pred_s2o:
            tp_fp += len(pred_s2o[s2o])

        for gold_sub in gold_s2o:
            for pred_sub in pred_s2o:
                if gold_sub[0] == pred_sub[0]:
                    gold_objs = gold_s2o[gold_sub]
                    pred_objs = pred_s2o[pred_sub]

                    for gold_obj in gold_objs:
                        for pred_obj in pred_objs:
                            if (gold_obj[0] == pred_obj[0]) and (gold_obj[2] == pred_obj[2]):
                                tp += 1

    pre = tp / tp_fp if tp_fp > 0 else 0
    rec = tp / tp_fn if tp_fn > 0 else 0
    f1 = 2 * tp / (tp_fp + tp_fn) if (tp_fp + tp_fn) > 0 else 0
    return pre, rec, f1


def main(args):
    logging.basicConfig(format='%(asctime)s - %(levelname)s: %(message)s', level=logging.INFO)

    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    if args.debug:
        args.batch_size = 3

    if args.multi_gpu:
        logging.info("run on multi GPU")
        torch.distributed.init_process_group(backend="nccl")

    model_utils.setup_seed(0)

    output_path = '%s/%s' % (args.output_path, args.task)
    if args.bert_freeze:
        output_path += '_freeze'
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    model_logger = Logger(data_path=output_path)

    rela_to_id, id_to_rela = load_relation_schema(args.task, '%s/%s' % (args.data_path, args.task))

    logging.info("loading embedding")
    tokenizer = BertTokenizer.from_pretrained('%s/vocab.txt' % args.pretrained_bert_path)

    logging.info("loading dataset")
    train_dataset, test_dataset = get_dataset(args, rela_to_id, tokenizer)

    if args.multi_gpu:
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=data_collate_fn,
                                      sampler=DistributedSampler(train_dataset, shuffle=True))
        test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, collate_fn=data_collate_fn,
                                     sampler=DistributedSampler(test_dataset, shuffle=False))
    else:
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=data_collate_fn,
                                      shuffle=True)
        test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, collate_fn=data_collate_fn,
                                     shuffle=False)

    best_f1 = 0
    epoch = 0

    if args.pretrained_model_path is not None:
        logging.info("loading pretrained model")
        model, optimizer, epoch, best_f1 = model_utils.load(args.pretrained_model_path)
        model = model.cpu() if args.use_cpu else model.cuda()
    else:
        logging.info("creating model")
        model = Casrel('%s/config.json' % args.pretrained_bert_path,
                       '%s/pytorch_model.bin' % args.pretrained_bert_path,
                       len(rela_to_id), args.bert_freeze)
        model = model.cpu() if args.use_cpu else model.cuda()

        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    if args.multi_gpu:
        model = DistributedDataParallel(model, find_unused_parameters=True)

    num_train_steps = int(len(train_dataset) / args.batch_size * args.epoch_size)
    num_warmup_steps = int(num_train_steps * args.lr_warmup_proportion)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=num_warmup_steps, gamma=args.lr_decay_gamma)

    logging.info("begin training")
    while epoch < args.epoch_size:
        epoch += 1

        train_loss = train(args, train_dataset, train_dataloader, model, optimizer, lr_scheduler)
        test_pre, test_rec, test_f1 = evaluate(args, test_dataloader, model)

        logging.info('epoch[%s/%s], train loss: %s' % (epoch, args.epoch_size, train_loss))
        logging.info('epoch[%s/%s], precision: %s, recall: %s, f1: %s' % (
            epoch, args.epoch_size, test_pre, test_rec, test_f1))
        model_utils.save(output_path, 'last.pth', model, optimizer, epoch, test_f1)

        remark = ''
        if test_f1 > best_f1:
            best_f1 = test_f1
            remark = 'best'
            model_utils.save(output_path, 'best.pth', model, optimizer, epoch, best_f1)

        model_logger.write(epoch, train_loss, 0, test_pre, test_rec, test_f1, remark)

    logging.info("complete training")
    model_logger.draw_plot()


if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('--task', type=str, choices=['duie', 'ccks2019'],
                        default='duie')
    parser.add_argument('--data_path', type=str,
                        default='../data/datasets/')
    parser.add_argument('--pretrained_bert_path', type=str,
                        default='../data/bert/bert-base-chinese/')
    parser.add_argument('--pretrained_model_path', type=str,
                        default=None)
    parser.add_argument('--output_path', type=str,
                        default='../runtime/joint/casrel/')
    parser.add_argument('--bert_freeze', type=bool,
                        default=False)
    parser.add_argument('--batch_size', type=int,
                        default=32)
    parser.add_argument('--epoch_size', type=int,
                        default=30)
    parser.add_argument('--learning_rate', type=float,
                        default=1e-5)
    parser.add_argument('--lr_warmup_proportion', type=float,
                        default=0.1)
    parser.add_argument('--lr_decay_gamma', type=float,
                        default=0.9)
    parser.add_argument('--use_cpu', type=bool,
                        default=False)
    parser.add_argument('--multi_gpu', type=bool,
                        help='run with: -m torch.distributed.launch',
                        default=False)
    parser.add_argument('--local_rank', type=int,
                        default=0)
    parser.add_argument('--debug', type=bool,
                        default=False)

    args = parser.parse_args()

    main(args)
