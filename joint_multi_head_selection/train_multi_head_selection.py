import os
import logging
from argparse import ArgumentParser
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch import optim
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel

from joint_multi_head_selection.model import MultiHeadSelection
from utils.dataloader import Tokenizer, MultiHeadDataset, load_relation_schema, load_pretrain_embedding
from utils.logger import Logger
from utils import model_utils


def get_dataset(args, rela_to_id, ner_tags_to_id, tokenizer, bigram_tokenizer):
    train_dataset = MultiHeadDataset(args.task, '%s/%s/train.txt' % (args.data_path, args.task),
                                     args.token_type, rela_to_id, ner_tags_to_id, tokenizer,
                                     use_bigram=args.use_bigram, bigram_tokenizer=bigram_tokenizer,
                                     do_to_id=True, do_sort=True, debug=args.debug)
    test_dataset = MultiHeadDataset(args.task, '%s/%s/dev.txt' % (args.data_path, args.task),
                                    args.token_type, rela_to_id, ner_tags_to_id, tokenizer,
                                    use_bigram=args.use_bigram, bigram_tokenizer=bigram_tokenizer,
                                    do_to_id=True, do_sort=True, debug=args.debug)

    return train_dataset, test_dataset


def data_collate_fn(data):
    data = np.array(data)

    context = data[:, 0].tolist()
    context = [torch.LongTensor(np.array(s)) for s in context]

    context_bigram = data[:, 1].tolist()
    context_bigram = [torch.LongTensor(np.array(s)) for s in context_bigram]

    ner_tags = data[:, 2].tolist()
    ner_tags = [torch.LongTensor(np.array(s)) for s in ner_tags]

    rela_links = data[:, 3].tolist()
    rela_links = [torch.LongTensor(np.array(s)) for s in rela_links]

    context_len = data[:, 4].tolist()

    return context, context_bigram, ner_tags, rela_links, context_len


def train(args, dataset, dataloader, model, optimizer, lr_scheduler):
    model.train()
    loss_sum = 0
    for batch, data in enumerate(dataloader):
        optimizer.zero_grad()
        context, bigram, ner_tags, rela_links, _ = data

        context = [item.cpu() if args.use_cpu else item.cuda() for item in context]
        bigram = [item.cpu() if args.use_cpu else item.cuda() for item in bigram]
        ner_tags = [item.cpu() if args.use_cpu else item.cuda() for item in ner_tags]
        rela_links = [item.cpu() if args.use_cpu else item.cuda() for item in rela_links]

        loss = model(context, bigram=bigram, decode=False, ner_tags=ner_tags, rela_links=rela_links)
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
            context, bigram, ner_tags, rela_links, context_len = data

            context = [item.cpu() if args.use_cpu else item.cuda() for item in context]
            bigram = [item.cpu() if args.use_cpu else item.cuda() for item in bigram]
            ner_tags = [item.cpu() if args.use_cpu else item.cuda() for item in ner_tags]

            _, rela_pred = model(context, bigram=bigram)

            rela_links = [item.cpu() for item in rela_links]
            rela_pred = [item.cpu() for item in rela_pred]

            gold_triplets = get_triplets(rela_links)
            pred_triplets = get_triplets(rela_pred)

            gold_answers.extend(gold_triplets)
            pred_answers.extend(pred_triplets)

    pre, rec, f1 = calc_measure(gold_answers, pred_answers)
    return pre, rec, f1


def get_triplets(rela_links):
    spo_list = []
    for idx in range(len(rela_links)):
        link = rela_links[idx]
        triplets = torch.nonzero(link)
        keep = triplets[:, 2] > 0
        triplets = triplets[keep, :]
        spo_list.append([list(triplet) for triplet in triplets.numpy()])
    return spo_list


def calc_measure(gold_answers, pred_answers):
    tp = 0
    tp_fp = 0
    tp_fn = 0
    for i in range(len(gold_answers)):
        tag_triplets = gold_answers[i]
        pred_triplets = pred_answers[i]

        tp_fn += len(tag_triplets)
        tp_fp += len(pred_triplets)

        for tag_tri in tag_triplets:
            for pred_tri in pred_triplets:
                if pred_tri == tag_tri:
                    tp += 1

    pre = tp / tp_fp if tp_fp > 0 else 0
    rec = tp / tp_fn if tp_fn > 0 else 0
    f1 = 2 * tp / (tp_fp + tp_fn) if (tp_fp + tp_fn) > 0 else 0
    return pre, rec, f1


def main(args):
    logging.basicConfig(format='%(asctime)s - %(levelname)s: %(message)s', level=logging.INFO)

    # os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    if args.debug:
        args.batch_size = 3

    if args.multi_gpu:
        logging.info("run on multi GPU")
        torch.distributed.init_process_group(backend="nccl")

    model_utils.setup_seed(0)

    output_path = '%s/%s/%s_%s' % (args.output_path, args.task, args.token_type, args.embed_type)
    if args.use_bigram:
        output_path += '_bigram'
    if args.use_crf:
        output_path += '_crf'
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    model_logger = Logger(data_path=output_path)

    rela_to_id, id_to_rela = load_relation_schema(args.task, '%s/%s' % (args.data_path, args.task), add_na=True)
    ner_tags_to_id, _ = MultiHeadDataset.generate_ner_tags()

    logging.info("loading embedding")
    token_to_id, pretrain_embed = load_pretrain_embedding(args.pretrained_emb_path,
                                                          has_meta=True if (args.token_type == 'word') else False,
                                                          add_pad=True, add_unk=True, debug=args.debug)
    tokenizer = Tokenizer(token_to_id)

    bigram_tokenizer = None
    bigram_to_id = {}
    if args.use_bigram:
        bigram_to_id, pretrain_bigram_embed = load_pretrain_embedding(args.pretrained_bigram_emb_path,
                                                                      add_pad=True, add_unk=True, debug=args.debug)
        bigram_tokenizer = Tokenizer(bigram_to_id)

    logging.info("loading dataset")
    train_dataset, test_dataset = get_dataset(args, rela_to_id, ner_tags_to_id, tokenizer, bigram_tokenizer)

    if args.multi_gpu:
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=data_collate_fn,
                                      sampler=DistributedSampler(train_dataset, shuffle=False))
        test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, collate_fn=data_collate_fn,
                                     sampler=DistributedSampler(test_dataset, shuffle=False))
    else:
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=data_collate_fn,
                                      shuffle=False)
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
        model = MultiHeadSelection(len(rela_to_id), len(ner_tags_to_id), len(token_to_id),
                                   args.embed_size, args.ner_embed_size, args.hidden_size,
                                   args.input_dropout_rate, args.hidden_dropout_rate,
                                   embed_fix=True if args.embed_type == 'static' else False,
                                   use_bigram=args.use_bigram, bigram_vocab_size=len(bigram_to_id),
                                   bigram_embed_size=args.embed_size, use_crf=args.use_crf)

        if args.embed_type in ['pretrain', 'static']:
            model.init_embedding(np.array(pretrain_embed))
            if args.use_bigram:
                model.init_bigram_embedding(np.array(pretrain_bigram_embed))

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
    parser.add_argument('--pretrained_emb_path', type=str,
                        help='[gigaword_chn.all.a2b.uni.ite50.vec],[news_tensite.pku.words.w2v50]',
                        default='../data/embeddings/gigaword_chn.all.a2b.uni.ite50.vec')
    parser.add_argument('--pretrained_bigram_emb_path', type=str,
                        default='../data/embeddings/gigaword_chn.all.a2b.bi.ite50.vec')
    parser.add_argument('--pretrained_model_path', type=str,
                        default=None)
    parser.add_argument('--output_path', type=str,
                        default='../runtime/joint/multi_head_selection/')
    parser.add_argument('--token_type', type=str, choices=['char', 'word'],
                        default='char')
    parser.add_argument('--embed_type', type=str, choices=['rand', 'pretrain', 'static'],
                        default='rand')
    parser.add_argument('--use_bigram', type=bool,
                        default=False)
    parser.add_argument('--use_crf', type=bool,
                        default=False)
    parser.add_argument('--embed_size', type=int,
                        default=50)
    parser.add_argument('--ner_embed_size', type=int,
                        default=25)
    parser.add_argument('--hidden_size', type=int,
                        default=128)
    parser.add_argument('--input_dropout_rate', type=float,
                        default=0.4)
    parser.add_argument('--hidden_dropout_rate', type=float,
                        default=0.4)
    parser.add_argument('--batch_size', type=int,
                        default=64)
    parser.add_argument('--epoch_size', type=int,
                        default=50)
    parser.add_argument('--learning_rate', type=float,
                        help='0.005 for no crf, 0.01 for crf',
                        default=0.005)
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
