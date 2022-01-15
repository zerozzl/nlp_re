import os
import logging
from argparse import ArgumentParser
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch import optim
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel

from distant_bag_att.model import BAG_ATT
from utils.dataloader import Tokenizer, BagGroupDataset, load_relation_schema, load_pretrain_embedding
from utils.logger import Logger
from utils import model_utils, evaluator


def get_dataset(args, tokenizer, bigram_tokenizer):
    train_dataset = BagGroupDataset(args.task, '%s/%s/' % (args.data_path, args.task), 'train', args.token_type,
                                    tokenizer, use_bigram=args.use_bigram, bigram_tokenizer=bigram_tokenizer,
                                    max_inp_len=args.max_inp_len, bag_size=args.bag_size,
                                    do_train=True, debug=args.debug)
    test_dataset = BagGroupDataset(args.task, '%s/%s/' % (args.data_path, args.task), 'dev', args.token_type,
                                   tokenizer, use_bigram=args.use_bigram, bigram_tokenizer=bigram_tokenizer,
                                   max_inp_len=args.max_inp_len, bag_size=args.bag_size,
                                   do_train=False, debug=args.debug)

    return train_dataset, test_dataset


def data_collate_fn(data):
    data = np.array(data)

    group_relations = data[:, 0].tolist()
    group_relations = torch.LongTensor(np.array(group_relations))

    group_data = []
    groups = data[:, 1]
    for bags in groups:
        bags = np.array(bags)

        bag_relations = bags[:, 0].tolist()
        bag_relations = [torch.LongTensor(np.array(item)) for item in bag_relations]

        context = bags[:, 1].tolist()
        context = [[torch.LongTensor(np.array(item)) for item in list] for list in context]

        bigram = bags[:, 2].tolist()
        bigram = [[torch.LongTensor(np.array(item)) for item in list] for list in bigram]

        subject_dis = bags[:, 5].tolist()
        subject_dis = [[torch.LongTensor(np.array(item)) for item in list] for list in subject_dis]

        object_dis = bags[:, 6].tolist()
        object_dis = [[torch.LongTensor(np.array(item)) for item in list] for list in object_dis]

        pool_masks = bags[:, 7].tolist()
        pool_masks = [[torch.LongTensor(np.array(item)) for item in list] for list in pool_masks]

        group_data.append([bag_relations, context, bigram, subject_dis, object_dis, pool_masks])

    return group_relations, group_data


def train(args, dataset, dataloader, model, optimizer, lr_scheduler):
    model.train()
    loss_sum = 0
    for batch, data in enumerate(dataloader):
        optimizer.zero_grad()
        group_relations, group_data = data

        group_relations = group_relations.cpu() if args.use_cpu else group_relations.cuda()
        for group_idx in range(len(group_data)):
            group = np.array(group_data[group_idx])
            # relations = [item.cpu() if args.use_cpu else item.cuda() for item in group[0]]
            context = [[item.cpu() if args.use_cpu else item.cuda() for item in list] for list in group[1]]
            bigram = [[item.cpu() if args.use_cpu else item.cuda() for item in list] for list in group[2]]
            subject_dis = [[item.cpu() if args.use_cpu else item.cuda() for item in list] for list in group[3]]
            object_dis = [[item.cpu() if args.use_cpu else item.cuda() for item in list] for list in group[4]]
            pool_masks = [[item.cpu() if args.use_cpu else item.cuda() for item in list] for list in group[5]]

            group_data[group_idx] = [context, bigram, subject_dis, object_dis, pool_masks]

        loss = model(group_data, decode=False, tags=group_relations)
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
            group_relations, group_data = data

            group_relations = []
            for group_idx in range(len(group_data)):
                group = np.array(group_data[group_idx])

                relations = [item.cpu().numpy().tolist() for item in group[0]]
                context = [[item.cpu() if args.use_cpu else item.cuda() for item in list] for list in group[1]]
                bigram = [[item.cpu() if args.use_cpu else item.cuda() for item in list] for list in group[2]]
                subject_dis = [[item.cpu() if args.use_cpu else item.cuda() for item in list] for list in group[3]]
                object_dis = [[item.cpu() if args.use_cpu else item.cuda() for item in list] for list in group[4]]
                pool_masks = [[item.cpu() if args.use_cpu else item.cuda() for item in list] for list in group[5]]

                group_relations.append(relations)
                group_data[group_idx] = [context, bigram, subject_dis, object_dis, pool_masks]

            preds = model(group_data)
            preds = [[item.cpu().numpy().tolist() for item in list] for list in preds]

            pred_answers.extend(preds)
            gold_answers.extend(group_relations)

    acc, pre, rec, f1 = calc_measure(gold_answers, pred_answers)
    return acc, pre, rec, f1


def calc_measure(gold_answers, pred_answers):
    gold_amend = []
    pred_amend = []
    for bag_idx in range(len(gold_answers)):
        gold_bags_ans = gold_answers[bag_idx]
        pred_bags_ans = pred_answers[bag_idx]

        for ins_idx in range(len(gold_bags_ans)):
            gold_ans = gold_bags_ans[ins_idx]
            pred_ans = pred_bags_ans[ins_idx]

            for and_idx in range(len(pred_ans)):
                ans = pred_ans[and_idx]
                if ans in gold_ans:
                    gold_amend.append(ans)
                else:
                    gold_amend.append(gold_ans[0])
                pred_amend.append(ans)

    return evaluator.evaluate(gold_amend, pred_amend)


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
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    model_logger = Logger(data_path=output_path)

    rela_to_id, _ = load_relation_schema(args.task, '%s/%s' % (args.data_path, args.task))

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
    train_dataset, test_dataset = get_dataset(args, tokenizer, bigram_tokenizer)

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
        model = BAG_ATT(len(rela_to_id), len(token_to_id), args.embed_size, args.max_inp_len, args.pf_embed_size,
                        args.conv_kernel_num, args.conv_kernel_size, args.input_dropout_rate, args.hidden_dropout_rate,
                        embed_fix=True if args.embed_type == 'static' else False,
                        use_bigram=args.use_bigram, bigram_vocab_size=len(bigram_to_id),
                        bigram_embed_size=args.embed_size, use_cpu=args.use_cpu)

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
        test_acc, test_pre, test_rec, test_f1 = evaluate(args, test_dataloader, model)

        logging.info('epoch[%s/%s], train loss: %s' % (epoch, args.epoch_size, train_loss))
        logging.info('epoch[%s/%s], test accuracy: %s, precision: %s, recall: %s, f1: %s' % (
            epoch, args.epoch_size, test_acc, test_pre, test_rec, test_f1))
        model_utils.save(output_path, 'last.pth', model, optimizer, epoch, test_f1)

        remark = ''
        if test_f1 > best_f1:
            best_f1 = test_f1
            remark = 'best'
            model_utils.save(output_path, 'best.pth', model, optimizer, epoch, best_f1)

        model_logger.write(epoch, train_loss, test_acc, test_pre, test_rec, test_f1, remark)

    logging.info("complete training")
    model_logger.draw_plot()


if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('--task', type=str, choices=['ccks2019_bag', 'cndbpedia'],
                        default='ccks2019_bag')
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
                        default='../runtime/distant/bag_att/')
    parser.add_argument('--token_type', type=str, choices=['char', 'word'],
                        default='char')
    parser.add_argument('--embed_type', type=str, choices=['rand', 'pretrain', 'static'],
                        default='rand')
    parser.add_argument('--use_bigram', type=bool,
                        default=False)
    parser.add_argument('--max_inp_len', type=int,
                        help='128 for ccks2019, 512 for cndbpedia',
                        default=128)
    parser.add_argument('--bag_size', type=int,
                        default=5)
    parser.add_argument('--embed_size', type=int,
                        default=50)
    parser.add_argument('--pf_embed_size', type=int,
                        default=5)
    parser.add_argument('--conv_kernel_num', type=int,
                        default=250)
    parser.add_argument('--conv_kernel_size', type=int,
                        default=3)
    parser.add_argument('--input_dropout_rate', type=float,
                        default=0.5)
    parser.add_argument('--hidden_dropout_rate', type=float,
                        default=0.5)
    parser.add_argument('--batch_size', type=int,
                        default=64)
    parser.add_argument('--epoch_size', type=int,
                        default=30)
    parser.add_argument('--learning_rate', type=float,
                        default=0.0005)
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
