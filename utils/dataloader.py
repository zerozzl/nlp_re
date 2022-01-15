import json
import codecs
import logging
import numpy as np
from random import shuffle
import jieba
from torch.utils.data import Dataset

TOKEN_PAD = '[PAD]'
TOKEN_UNK = '[UNK]'
TOKEN_CLS = '[CLS]'
TOKEN_SEP = '[SEP]'
TOKEN_EDGES_START = '<s>'
TOKEN_EDGES_END = '</s>'
TOKEN_SUBJECT_START = '<e1>'
TOKEN_SUBJECT_END = '</e1>'
TOKEN_OBJECT_START = '<e2>'
TOKEN_OBJECT_END = '</e2>'

DUIE_TRANSLATOR = {
    '人物': 'PER',
    '娱乐人物': 'PER_ENT',
    '历史人物': 'PER_HIS',
    '作品': 'PRODUCTION',
    '文学作品': 'LITERATURE',
    '图书作品': 'BOOK',
    '影视作品': 'FILM',
    '电视综艺': 'TV_SHOW',
    '音乐专辑': 'MUSIC_ALBUM',
    '歌曲': 'MUSIC',
    '国家': 'COUNTRY',
    '城市': 'CITY',
    '行政区': 'DISTRICT',
    '地点': 'SPOT',
    '景点': 'SCENIC_SPOT',
    '企业': 'COMPANY',
    '企业/品牌': 'BRAND',
    '机构': 'ORG',
    '学校': 'SCHOOL',
    '学科专业': 'COURSE',
    '气候': 'CLIMATE',
    '语言': 'LANGUAGE',
    '奖项': 'AWARDS',
    'Text': 'TEXT',
    'Number': 'NUM',
    'Date': 'DATE'
}


class ReRecord:
    def __init__(self, text, relation, subject, subject_type, object, object_type, subject_idx=None, object_idx=None):
        self.text = text
        self.relation = relation
        self.subject = subject
        self.subject_type = subject_type
        self.subject_idx = subject_idx
        self.object = object
        self.object_type = object_type
        self.object_idx = object_idx


class ReJointRecord:
    def __init__(self, text, spos):
        self.text = text
        self.spos = spos


class ReBag:
    def __init__(self, relation, subject, object, records):
        self.relation = relation
        self.subject = subject
        self.object = object
        self.records = records


class ReBagRecord:
    def __init__(self, text, subject_idx, object_idx):
        self.text = text
        self.subject_idx = subject_idx
        self.object_idx = object_idx


class ReDataset(Dataset):
    def __init__(self, task, data_path, token_type, rela_to_id, tokenizer, use_bigram=False, bigram_tokenizer=None,
                 max_inp_len=0, do_pad=False, pad_token=TOKEN_PAD, do_to_id=False, do_sort=False, for_bert=False,
                 add_entity_mark=False, debug=False):
        super(ReDataset, self).__init__()
        self.data = []

        records = []
        if task == 'duie':
            records = self.read_duie(data_path, rela_to_id, debug)
        elif task == 'ccks2019':
            records = self.read_ccks2019(data_path, debug)

        for record in records:
            text = self.dbc_to_sbc(record.text)
            subject = self.dbc_to_sbc(record.subject)
            object = self.dbc_to_sbc(record.object)

            try:
                subject_idx = text.index(subject)
                object_idx = text.index(object)
            except ValueError:
                continue

            if subject_idx < object_idx:
                context = [text[:subject_idx], text[subject_idx:subject_idx + len(subject)],
                           text[subject_idx + len(subject):object_idx], text[object_idx:object_idx + len(object)],
                           text[object_idx + len(object):]]
            else:
                context = [text[:object_idx], text[object_idx:object_idx + len(object)],
                           text[object_idx + len(object):subject_idx], text[subject_idx:subject_idx + len(subject)],
                           text[subject_idx + len(subject):]]

            if token_type == 'char':
                context = [[ch for ch in piece] for piece in context]
                # subject = [ch for ch in subject]
                # object = [ch for ch in object]
            elif token_type == 'word':
                context = [list(jieba.cut(piece)) for piece in context]
                # subject = list(jieba.cut(subject))
                # object = list(jieba.cut(object))

            if for_bert:
                context[0].insert(0, TOKEN_CLS)

            if add_entity_mark:
                if subject_idx < object_idx:
                    context[1].insert(0, TOKEN_SUBJECT_START)
                    context[1].append(TOKEN_SUBJECT_END)
                    context[3].insert(0, TOKEN_OBJECT_START)
                    context[3].append(TOKEN_OBJECT_END)
                else:
                    context[1].insert(0, TOKEN_OBJECT_START)
                    context[1].append(TOKEN_OBJECT_END)
                    context[3].insert(0, TOKEN_SUBJECT_START)
                    context[3].append(TOKEN_SUBJECT_END)

            if subject_idx < object_idx:
                subject_idx = [len(context[0]), len(context[0]) + len(context[1]) - 1]
                object_idx = [len(context[0]) + len(context[1]) + len(context[2]),
                              len(context[0]) + len(context[1]) + len(context[2]) + len(context[3]) - 1]
            else:
                object_idx = [len(context[0]), len(context[0]) + len(context[1]) - 1]
                subject_idx = [len(context[0]) + len(context[1]) + len(context[2]),
                               len(context[0]) + len(context[1]) + len(context[2]) + len(context[3]) - 1]

            context = [ch for piece in context for ch in piece]
            if max_inp_len > 0:
                if (subject_idx[0] >= max_inp_len) or (subject_idx[1] >= max_inp_len) \
                        or (object_idx[0] >= max_inp_len) or (object_idx[1] >= max_inp_len):
                    break
                context = context[:max_inp_len]

            segment = [0] * len(context)
            context_len = len(context)

            subject_dis = []
            object_dis = []
            for token_idx in range(len(context)):
                if (token_idx >= subject_idx[0]) and (token_idx <= subject_idx[1]):
                    subject_dis.append(0)
                else:
                    dis1 = token_idx - subject_idx[0]
                    dis2 = token_idx - subject_idx[1]
                    subject_dis.append(dis1 if abs(dis1) < abs(dis2) else dis2)
                    # subject_dis.append(min(abs(token_idx - subject_idx[0]), abs(token_idx - subject_idx[1])))

                if (token_idx >= object_idx[0]) and (token_idx <= object_idx[1]):
                    object_dis.append(0)
                else:
                    dis1 = token_idx - object_idx[0]
                    dis2 = token_idx - object_idx[1]
                    object_dis.append(dis1 if abs(dis1) < abs(dis2) else dis2)
                    # object_dis.append(min(abs(token_idx - object_idx[0]), abs(token_idx - object_idx[1])))

            bigram = []
            if use_bigram:
                bigram = [TOKEN_EDGES_START] + context + [TOKEN_EDGES_END]
                bigram = [[bigram[i - 1] + bigram[i]] + [bigram[i] + bigram[i + 1]] for i in range(1, len(bigram) - 1)]

            if do_pad and (max_inp_len > 0):
                if use_bigram:
                    bigram = bigram + [[pad_token, pad_token]] * (max_inp_len - len(context))

                context = context + [pad_token] * (max_inp_len - len(context))

            if do_to_id:
                context = tokenizer.convert_tokens_to_ids(context)
                if use_bigram:
                    bigram = bigram_tokenizer.convert_tokens_to_ids(bigram)

            self.data.append([record.relation, context, bigram, context_len,
                              subject_idx, object_idx, subject_dis, object_dis, segment])

        if do_sort:
            self.data = sorted(self.data, key=lambda x: x[3], reverse=True)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def dbc_to_sbc(self, ustring):
        rstring = ''
        for uchar in ustring:
            inside_code = ord(uchar)
            if inside_code == 0x3000:
                inside_code = 0x0020
            else:
                inside_code -= 0xfee0
            if not (0x0021 <= inside_code and inside_code <= 0x7e):
                rstring += uchar
                continue
            rstring += chr(inside_code)
        return rstring

    def read_duie(self, data_path, rela_to_id, debug=False):
        records = []
        with codecs.open(data_path, 'r', 'utf-8') as fin:
            for line in fin:
                line = line.strip()
                if line == '':
                    continue

                line = json.loads(line)

                text = line['text']
                text = text.strip()
                if text == '':
                    continue

                spo_list = line['spo_list']
                for spo in spo_list:
                    predicate = spo['predicate']
                    subject = spo['subject']
                    subject_type = spo['subject_type']
                    object = spo['object']['@value']
                    object_type = spo['object_type']['@value']

                    subject = subject.strip()
                    object = object.strip()
                    if (subject == '') or (object == ''):
                        continue

                    predicate = rela_to_id[predicate]
                    subject_type = DUIE_TRANSLATOR[subject_type]
                    object_type = DUIE_TRANSLATOR[object_type]
                    records.append(ReRecord(text, predicate, subject, subject_type, object, object_type))

                    if debug:
                        if len(records) >= 30:
                            break
                if debug:
                    if len(records) >= 30:
                        break
        return records

    def read_ccks2019(self, data_path, debug=False):
        records = []
        with codecs.open(data_path, 'r', 'utf-8') as fin:
            for line in fin:
                line = line.strip()
                if line == '':
                    continue

                line = line.split('\t')
                if len(line) != 4:
                    continue

                text = line[3]
                text = text.replace(' ', '')

                predicate = int(line[0])
                subject = line[1]
                subject_type = 'PER'
                object = line[2]
                object_type = 'PER'

                records.append(ReRecord(text, predicate, subject, subject_type, object, object_type))

                if debug:
                    if len(records) >= 30:
                        break
        return records

    @staticmethod
    def transform_ccks2019(datapath, output_path):
        logging.info('Transform ccks2019')
        data_split = ['train', 'dev']
        for ds in data_split:
            logging.info('Reading %s data' % ds)
            relations = {}
            with codecs.open('%s/sent_relation_%s.txt' % (datapath, ds), 'r', 'utf-8') as fin:
                for line in fin:
                    line = line.strip()
                    if line == '':
                        continue

                    line = line.split()
                    relations[line[0]] = int(line[1])

            records = []
            with codecs.open('%s/sent_%s.txt' % (datapath, ds), 'r', 'utf-8') as fin:
                for line in fin:
                    line = line.strip()
                    if line == '':
                        continue

                    line = line.split('\t')
                    if len(line) != 4:
                        continue

                    rela = relations[line[0]]
                    records.append([rela, line[1], line[2], line[3]])

            with codecs.open('%s/%s.txt' % (output_path, ds), 'w', 'utf-8') as fout:
                for record in records:
                    fout.write('%s\t%s\t%s\t%s\n' % (record[0], record[1], record[2], record[3]))

        logging.info('Complete. saved data in %s' % output_path)

    @staticmethod
    def statistics_duie(datapath):
        logging.info('========== statistics duie ==========')
        sent_len_dict = {64: 0, 128: 0, 256: 0, 288: 0, 320: 0, 384: 0, 512: 0}

        data_granularity = ['char', 'word']
        data_split = ['train', 'dev']
        for dg in data_granularity:
            for ds in data_split:
                with codecs.open('%s/%s.txt' % (datapath, ds), 'r', 'utf-8') as fin:
                    for line in fin:
                        line = line.strip()
                        if line == '':
                            continue

                        line = json.loads(line)
                        text = line['text']

                        if dg == 'char':
                            text_len = len(text)
                        elif dg == 'word':
                            text_len = len(list(jieba.cut(text)))

                        for sl in sent_len_dict:
                            if text_len <= sl:
                                sent_len_dict[sl] = sent_len_dict[sl] + 1
                                break

                logging.info('%s level %s data count' % (dg, ds))
                logging.info('test length: %s' % str(sent_len_dict))

                for sl in sent_len_dict:
                    sent_len_dict[sl] = 0

    @staticmethod
    def statistics_ccks2019(datapath):
        logging.info('========== statistics ccks2019 ==========')
        sent_len_dict = {64: 0, 128: 0, 256: 0, 288: 0, 320: 0, 384: 0, 512: 0, 1024: 0}

        data_granularity = ['char', 'word']
        data_split = ['train', 'dev']
        for dg in data_granularity:
            for ds in data_split:
                with codecs.open('%s/%s.txt' % (datapath, ds), 'r', 'utf-8') as fin:
                    for line in fin:
                        line = line.strip()
                        if line == '':
                            continue

                        line = line.split('\t')
                        text = line[3]

                        if dg == 'char':
                            text_len = len(text)
                        elif dg == 'word':
                            text_len = len(list(jieba.cut(text)))

                        for sl in sent_len_dict:
                            if text_len <= sl:
                                sent_len_dict[sl] = sent_len_dict[sl] + 1
                                break

                logging.info('%s level %s data count' % (dg, ds))
                logging.info('test length: %s' % str(sent_len_dict))

                for sl in sent_len_dict:
                    sent_len_dict[sl] = 0


class NovelTagDataset(Dataset):
    def __init__(self, task, data_path, token_type, rela_to_id, tag_to_id, tokenizer,
                 use_bigram=False, bigram_tokenizer=None, max_inp_len=0, do_pad=False, pad_token=TOKEN_PAD,
                 do_to_id=False, do_sort=False, debug=False):
        super(NovelTagDataset, self).__init__()
        self.data = []

        records = []
        if task == 'duie':
            records = self.read_duie(data_path, rela_to_id, debug)
        elif task == 'ccks2019':
            records = self.read_ccks2019(data_path, debug)

        for record in records:
            text = self.dbc_to_sbc(record.text)

            entities = []
            spo_list = record.spos
            for spo in spo_list:
                subject = self.dbc_to_sbc(spo.subject)
                object = self.dbc_to_sbc(spo.object)

                try:
                    subject_idx = text.index(subject)
                    object_idx = text.index(object)

                    subject_idx = [subject_idx, subject_idx + len(subject) - 1]
                    object_idx = [object_idx, object_idx + len(object) - 1]

                    overlap = False
                    for item in entities:
                        if ((subject_idx[0] >= item[0]) and (subject_idx[0] <= item[1])) or \
                                ((subject_idx[1] >= item[0]) and (subject_idx[1] <= item[1])) or \
                                ((object_idx[0] >= item[0]) and (object_idx[0] <= item[1])) or \
                                ((object_idx[1] >= item[0]) and (object_idx[1] <= item[1])):
                            overlap = True
                            break

                    if not overlap:
                        entities.append([subject_idx[0], subject_idx[1], spo.relation, 'S'])
                        entities.append([object_idx[0], object_idx[1], spo.relation, 'O'])

                except ValueError:
                    continue

            if len(entities) <= 0:
                continue

            entities.sort(key=lambda x: x[0], reverse=False)

            context = []
            tags = []

            piece = text[:entities[0][0]]
            if token_type == 'char':
                piece = [ch for ch in piece]
            elif token_type == 'word':
                piece = list(jieba.cut(piece))
            context.extend(piece)
            tags.extend(['O'] * len(piece))

            for ent_idx in range(len(entities)):
                if ent_idx > 0:
                    piece = text[entities[ent_idx - 1][1] + 1:entities[ent_idx][0]]
                    if token_type == 'char':
                        piece = [ch for ch in piece]
                    elif token_type == 'word':
                        piece = list(jieba.cut(piece))
                    context.extend(piece)
                    tags.extend(['O'] * len(piece))

                piece = text[entities[ent_idx][0]:entities[ent_idx][1] + 1]
                if token_type == 'char':
                    piece = [ch for ch in piece]
                elif token_type == 'word':
                    piece = list(jieba.cut(piece))
                context.extend(piece)
                tags.extend(self.get_tag(entities[ent_idx], piece))

            piece = text[entities[-1][1] + 1:]
            if token_type == 'char':
                piece = [ch for ch in piece]
            elif token_type == 'word':
                piece = list(jieba.cut(piece))
            context.extend(piece)
            tags.extend(['O'] * len(piece))

            if max_inp_len > 0:
                context = context[:max_inp_len]
                tags = tags[:max_inp_len]
            context_len = len(context)

            bigram = []
            if use_bigram:
                bigram = [TOKEN_EDGES_START] + context + [TOKEN_EDGES_END]
                bigram = [[bigram[i - 1] + bigram[i]] + [bigram[i] + bigram[i + 1]] for i in range(1, len(bigram) - 1)]

            if do_pad and (max_inp_len > 0):
                if use_bigram:
                    bigram = bigram + [[pad_token, pad_token]] * (max_inp_len - len(context))

                tags = tags + ['O'] * (max_inp_len - len(context))
                context = context + [pad_token] * (max_inp_len - len(context))

            if do_to_id:
                context = tokenizer.convert_tokens_to_ids(context)
                if use_bigram:
                    bigram = bigram_tokenizer.convert_tokens_to_ids(bigram)
                tags = [tag_to_id.get(tag) for tag in tags]

            self.data.append([context, bigram, tags, context_len])

        if do_sort:
            self.data = sorted(self.data, key=lambda x: x[3], reverse=True)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def dbc_to_sbc(self, ustring):
        rstring = ''
        for uchar in ustring:
            inside_code = ord(uchar)
            if inside_code == 0x3000:
                inside_code = 0x0020
            else:
                inside_code -= 0xfee0
            if not (0x0021 <= inside_code and inside_code <= 0x7e):
                rstring += uchar
                continue
            rstring += chr(inside_code)
        return rstring

    def get_tag(self, entity, piece):
        entity_type = '%s-%s' % (entity[2], entity[3])
        if len(piece) == 1:
            return ['S-%s' % entity_type]
        return ['B-%s' % entity_type] + ['M-%s' % entity_type] * (len(piece) - 2) + ['E-%s' % entity_type]

    def read_duie(self, data_path, rela_to_id, debug=False):
        records = []
        with codecs.open(data_path, 'r', 'utf-8') as fin:
            for line in fin:
                line = line.strip()
                if line == '':
                    continue

                line = json.loads(line)

                text = line['text']
                text = text.strip()
                if text == '':
                    continue

                record_spo_list = []
                spo_list = line['spo_list']
                for spo in spo_list:
                    predicate = spo['predicate']
                    subject = spo['subject']
                    subject_type = spo['subject_type']
                    object = spo['object']['@value']
                    object_type = spo['object_type']['@value']

                    subject = subject.strip()
                    object = object.strip()
                    if (subject == '') or (object == ''):
                        continue

                    predicate = rela_to_id[predicate]
                    subject_type = DUIE_TRANSLATOR[subject_type]
                    object_type = DUIE_TRANSLATOR[object_type]
                    record_spo_list.append(ReRecord('', predicate, subject, subject_type, object, object_type))

                if len(record_spo_list) > 0:
                    records.append(ReJointRecord(text, record_spo_list))

                if debug:
                    if len(records) >= 10:
                        break

        return records

    def read_ccks2019(self, data_path, debug=False):
        records = []
        with codecs.open(data_path, 'r', 'utf-8') as fin:
            for line in fin:
                line = line.strip()
                if line == '':
                    continue

                line = line.split('\t')
                if len(line) != 4:
                    continue

                predicate = int(line[0])
                if predicate == 0:
                    continue

                text = line[3]
                text = text.replace(' ', '')

                subject = line[1]
                subject_type = 'PER'
                object = line[2]
                object_type = 'PER'

                records.append(
                    ReJointRecord(text, [ReRecord('', predicate, subject, subject_type, object, object_type)]))

                if debug:
                    if len(records) >= 10:
                        break

        return records

    @staticmethod
    def generate_tags(id_to_rela):
        tag_to_id = {'O': 0}
        for id in id_to_rela:
            tag_to_id['B-%s-S' % id] = len(tag_to_id)
            tag_to_id['M-%s-S' % id] = len(tag_to_id)
            tag_to_id['E-%s-S' % id] = len(tag_to_id)
            tag_to_id['S-%s-S' % id] = len(tag_to_id)
            tag_to_id['B-%s-O' % id] = len(tag_to_id)
            tag_to_id['M-%s-O' % id] = len(tag_to_id)
            tag_to_id['E-%s-O' % id] = len(tag_to_id)
            tag_to_id['S-%s-O' % id] = len(tag_to_id)

        id_to_tag = {v: k for k, v in tag_to_id.items()}
        return tag_to_id, id_to_tag


class MultiHeadDataset(Dataset):
    def __init__(self, task, data_path, token_type, rela_to_id, ner_tags_to_id, tokenizer,
                 use_bigram=False, bigram_tokenizer=None, max_inp_len=0, do_pad=False, pad_token=TOKEN_PAD,
                 do_to_id=False, do_sort=False, debug=False):
        super(MultiHeadDataset, self).__init__()
        self.data = []

        records = []
        if task == 'duie':
            records = self.read_duie(data_path, rela_to_id, debug)
        elif task == 'ccks2019':
            records = self.read_ccks2019(data_path, debug)

        for record in records:
            text = self.dbc_to_sbc(record.text)

            entities = []
            entities_link = {}
            spo_list = record.spos
            for spo in spo_list:
                subject = self.dbc_to_sbc(spo.subject)
                object = self.dbc_to_sbc(spo.object)

                try:
                    subject_idx = text.index(subject)
                    object_idx = text.index(object)

                    subject_idx = [subject_idx, subject_idx + len(subject) - 1]
                    object_idx = [object_idx, object_idx + len(object) - 1]

                    subject_exist = False
                    subject_overlap = False
                    for item in entities:
                        if (subject_idx[0] == item[0]) and (subject_idx[1] == item[1]):
                            subject_exist = True
                            break
                        if ((subject_idx[0] >= item[0]) and (subject_idx[0] <= item[1])) or \
                                ((subject_idx[1] >= item[0]) and (subject_idx[1] <= item[1])):
                            subject_overlap = True
                            break
                    if not (subject_exist or subject_overlap):
                        entities.append([subject_idx[0], subject_idx[1], spo.subject_type])

                    object_exist = False
                    object_overlap = False
                    for item in entities:
                        if (object_idx[0] == item[0]) and (object_idx[1] == item[1]):
                            object_exist = True
                            break
                        if ((object_idx[0] >= item[0]) and (object_idx[0] <= item[1])) or \
                                ((object_idx[1] >= item[0]) and (object_idx[1] <= item[1])):
                            object_overlap = True
                            break
                    if not (object_exist or object_overlap):
                        entities.append([object_idx[0], object_idx[1], spo.object_type])

                    if subject_overlap or object_overlap:
                        continue
                    else:
                        if subject_idx[0] in entities_link:
                            entities_link[subject_idx[0]].append([object_idx[0], spo.relation])
                        else:
                            entities_link[subject_idx[0]] = [[object_idx[0], spo.relation]]
                except ValueError:
                    continue

            if len(entities) <= 0:
                continue

            entities.sort(key=lambda x: x[0], reverse=False)

            entities_offset = {}
            context = []
            ner_tags = []

            piece = text[:entities[0][0]]
            if token_type == 'char':
                piece = [ch for ch in piece]
            elif token_type == 'word':
                piece = list(jieba.cut(piece))
            context.extend(piece)
            ner_tags.extend(['O'] * len(piece))

            for ent_idx in range(len(entities)):
                if ent_idx > 0:
                    piece = text[entities[ent_idx - 1][1] + 1:entities[ent_idx][0]]
                    if token_type == 'char':
                        piece = [ch for ch in piece]
                    elif token_type == 'word':
                        piece = list(jieba.cut(piece))
                    context.extend(piece)
                    ner_tags.extend(['O'] * len(piece))

                piece = text[entities[ent_idx][0]:entities[ent_idx][1] + 1]
                if token_type == 'char':
                    piece = [ch for ch in piece]
                elif token_type == 'word':
                    piece = list(jieba.cut(piece))
                entities_offset[entities[ent_idx][0]] = len(context)
                context.extend(piece)
                ner_tags.extend(self.get_ner_tag(entities[ent_idx], piece))

            piece = text[entities[-1][1] + 1:]
            if token_type == 'char':
                piece = [ch for ch in piece]
            elif token_type == 'word':
                piece = list(jieba.cut(piece))
            context.extend(piece)
            ner_tags.extend(['O'] * len(piece))

            entities_link_trans = {}
            for ent_idx in entities_link:
                links_trans = []
                links = entities_link[ent_idx]
                for link in links:
                    links_trans.append([entities_offset[link[0]], link[1]])
                entities_link_trans[entities_offset[ent_idx]] = links_trans
            entities_link = entities_link_trans

            rela_links = np.zeros((len(context), len(context), len(rela_to_id)))
            for ent_idx in range(len(context)):
                if ent_idx in entities_link:
                    links = entities_link[ent_idx]
                    for link in links:
                        rela_links[ent_idx, link[0], link[1]] = 1
                else:
                    rela_links[ent_idx, ent_idx, 0] = 1
            rela_links = list(rela_links)

            if max_inp_len > 0:
                context = context[:max_inp_len]
                ner_tags = ner_tags[:max_inp_len]
                rela_links = rela_links[:max_inp_len]
            context_len = len(context)

            bigram = []
            if use_bigram:
                bigram = [TOKEN_EDGES_START] + context + [TOKEN_EDGES_END]
                bigram = [[bigram[i - 1] + bigram[i]] + [bigram[i] + bigram[i + 1]] for i in range(1, len(bigram) - 1)]

            if do_pad and (max_inp_len > 0):
                if use_bigram:
                    bigram = bigram + [[pad_token, pad_token]] * (max_inp_len - len(context))

                ner_tags = ner_tags + ['O'] * (max_inp_len - len(context))
                rela_links.extend([[[ch_idx, 0]] for ch_idx in range(len(context), max_inp_len)])
                context = context + [pad_token] * (max_inp_len - len(context))

            if do_to_id:
                context = tokenizer.convert_tokens_to_ids(context)
                if use_bigram:
                    bigram = bigram_tokenizer.convert_tokens_to_ids(bigram)

                ner_tags = [ner_tags_to_id.get(tag) for tag in ner_tags]

            self.data.append([context, bigram, ner_tags, rela_links, context_len])

        if do_sort:
            self.data = sorted(self.data, key=lambda x: x[4], reverse=True)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def dbc_to_sbc(self, ustring):
        rstring = ''
        for uchar in ustring:
            inside_code = ord(uchar)
            if inside_code == 0x3000:
                inside_code = 0x0020
            else:
                inside_code -= 0xfee0
            if not (0x0021 <= inside_code and inside_code <= 0x7e):
                rstring += uchar
                continue
            rstring += chr(inside_code)
        return rstring

    def get_ner_tag(self, entity, piece):
        entity_type = entity[2]
        if len(piece) == 1:
            return ['S-%s' % entity_type]
        return ['B-%s' % entity_type] + ['M-%s' % entity_type] * (len(piece) - 2) + ['E-%s' % entity_type]

    def read_duie(self, data_path, rela_to_id, debug=False):
        records = []
        with codecs.open(data_path, 'r', 'utf-8') as fin:
            for line in fin:
                line = line.strip()
                if line == '':
                    continue

                line = json.loads(line)

                text = line['text']
                text = text.strip()
                if text == '':
                    continue

                record_spo_list = []
                spo_list = line['spo_list']
                for spo in spo_list:
                    predicate = spo['predicate']
                    subject = spo['subject']
                    subject_type = spo['subject_type']
                    object = spo['object']['@value']
                    object_type = spo['object_type']['@value']

                    subject = subject.strip()
                    object = object.strip()
                    if (subject == '') or (object == ''):
                        continue

                    predicate = rela_to_id[predicate]
                    subject_type = DUIE_TRANSLATOR[subject_type]
                    object_type = DUIE_TRANSLATOR[object_type]
                    record_spo_list.append(ReRecord('', predicate, subject, subject_type, object, object_type))

                if len(record_spo_list) > 0:
                    records.append(ReJointRecord(text, record_spo_list))

                if debug:
                    if len(records) >= 10:
                        break

        return records

    def read_ccks2019(self, data_path, debug=False):
        records = []
        with codecs.open(data_path, 'r', 'utf-8') as fin:
            for line in fin:
                line = line.strip()
                if line == '':
                    continue

                line = line.split('\t')
                if len(line) != 4:
                    continue

                predicate = int(line[0])
                if predicate == 0:
                    continue

                text = line[3]
                text = text.replace(' ', '')

                subject = line[1]
                subject_type = 'PER'
                object = line[2]
                object_type = 'PER'

                records.append(
                    ReJointRecord(text, [ReRecord('', predicate, subject, subject_type, object, object_type)]))

                if debug:
                    if len(records) >= 10:
                        break

        return records

    @staticmethod
    def generate_ner_tags():
        tag_to_id = {'O': 0}
        for name in DUIE_TRANSLATOR:
            tag = DUIE_TRANSLATOR[name]
            tag_to_id['B-%s' % tag] = len(tag_to_id)
            tag_to_id['M-%s' % tag] = len(tag_to_id)
            tag_to_id['E-%s' % tag] = len(tag_to_id)
            tag_to_id['S-%s' % tag] = len(tag_to_id)

        id_to_tag = {v: k for k, v in tag_to_id.items()}
        return tag_to_id, id_to_tag


class CasrelDataset(Dataset):
    def __init__(self, task, data_path, rela_to_id, tokenizer, max_inp_len=0, do_train=False, do_sort=False,
                 debug=False):
        super(CasrelDataset, self).__init__()
        self.data = []

        records = []
        if task == 'duie':
            records = self.read_duie(data_path, rela_to_id, debug)
        elif task == 'ccks2019':
            records = self.read_ccks2019(data_path, debug)

        for record in records:
            text = self.dbc_to_sbc(record.text)
            tokens = [TOKEN_CLS] + [ch for ch in text]

            if max_inp_len > 0:
                tokens = tokens[: max_inp_len]

            s2o = {}
            spo_list = record.spos
            for spo in spo_list:
                subject = self.dbc_to_sbc(spo.subject)
                object = self.dbc_to_sbc(spo.object)

                subject = [ch for ch in subject]
                object = [ch for ch in object]

                subject_head_idx = self.find_head_idx(tokens, subject)
                object_head_idx = self.find_head_idx(tokens, object)

                if (subject_head_idx != -1) and (object_head_idx != -1):
                    sub = (subject_head_idx, subject_head_idx + len(subject) - 1)
                    if sub not in s2o:
                        s2o[sub] = []
                    s2o[sub].append((object_head_idx, object_head_idx + len(object) - 1, spo.relation))

            if len(s2o) == 0:
                continue

            tokens = tokenizer.convert_tokens_to_ids(tokens)
            context_len = len(tokens)
            segment_ids = [0] * context_len
            mask_ids = [1] * context_len

            if do_train:
                all_subject_heads, all_subject_tails = [0] * context_len, [0] * context_len
                for subject in s2o:
                    all_subject_heads[subject[0]] = 1
                    all_subject_tails[subject[1]] = 1

                for subject in s2o:
                    subject_head, subject_tail = [0] * context_len, [0] * context_len
                    subject_head[subject[0]] = 1
                    subject_tail[subject[1]] = 1

                    object_heads = np.zeros((context_len, len(rela_to_id))).tolist()
                    object_tails = np.zeros((context_len, len(rela_to_id))).tolist()
                    for link in s2o[subject]:
                        object_heads[link[0]][link[2]] = 1
                        object_tails[link[1]][link[2]] = 1

                    self.data.append([tokens, segment_ids, mask_ids, context_len, all_subject_heads, all_subject_tails,
                                      subject_head, subject_tail, object_heads, object_tails, s2o])
            else:
                self.data.append([tokens, segment_ids, mask_ids, context_len, [], [], [], [], [], [], s2o])

        if do_sort:
            self.data = sorted(self.data, key=lambda x: x[3], reverse=True)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def dbc_to_sbc(self, ustring):
        rstring = ''
        for uchar in ustring:
            inside_code = ord(uchar)
            if inside_code == 0x3000:
                inside_code = 0x0020
            else:
                inside_code -= 0xfee0
            if not (0x0021 <= inside_code and inside_code <= 0x7e):
                rstring += uchar
                continue
            rstring += chr(inside_code)
        return rstring

    def find_head_idx(self, source, target):
        target_len = len(target)
        for i in range(len(source)):
            if source[i: i + target_len] == target:
                return i
        return -1

    def read_duie(self, data_path, rela_to_id, debug=False):
        records = []
        with codecs.open(data_path, 'r', 'utf-8') as fin:
            for line in fin:
                line = line.strip()
                if line == '':
                    continue

                line = json.loads(line)

                text = line['text']
                text = text.strip()
                if text == '':
                    continue

                record_spo_list = []
                spo_list = line['spo_list']
                for spo in spo_list:
                    predicate = spo['predicate']
                    subject = spo['subject']
                    subject_type = spo['subject_type']
                    object = spo['object']['@value']
                    object_type = spo['object_type']['@value']

                    subject = subject.strip()
                    object = object.strip()
                    if (subject == '') or (object == ''):
                        continue

                    predicate = rela_to_id[predicate]
                    subject_type = DUIE_TRANSLATOR[subject_type]
                    object_type = DUIE_TRANSLATOR[object_type]
                    record_spo_list.append(ReRecord('', predicate, subject, subject_type, object, object_type))

                if len(record_spo_list) > 0:
                    records.append(ReJointRecord(text, record_spo_list))

                if debug:
                    if len(records) >= 10:
                        break

        return records

    def read_ccks2019(self, data_path, debug=False):
        records = []
        with codecs.open(data_path, 'r', 'utf-8') as fin:
            for line in fin:
                line = line.strip()
                if line == '':
                    continue

                line = line.split('\t')
                if len(line) != 4:
                    continue

                predicate = int(line[0])
                if predicate == 0:
                    continue

                text = line[3]
                text = text.replace(' ', '')

                subject = line[1]
                subject_type = 'PER'
                object = line[2]
                object_type = 'PER'

                records.append(
                    ReJointRecord(text, [ReRecord('', predicate, subject, subject_type, object, object_type)]))

                if debug:
                    if len(records) >= 10:
                        break

        return records


class DistantDataset(Dataset):
    def __init__(self, task, data_path, data_split, token_type, tokenizer,
                 use_bigram=False, bigram_tokenizer=None, max_inp_len=0, do_train=False, debug=False):
        super(DistantDataset, self).__init__()
        self.data = []

        bags = []
        if task == 'ccks2019_bag':
            bags = self.read_ccks2019(data_path, data_split, do_train, debug)
        elif task == 'cndbpedia':
            bags = self.read_cndbpedia(data_path, data_split, debug)

        for bag in bags:
            subject = bag.subject
            object = bag.object
            bag_context = []
            bag_bigram = []
            bag_subject_idx = []
            bag_object_idx = []
            bag_subject_dis = []
            bag_object_dis = []
            bag_pool_masks = []
            for record in bag.records:
                text = record.text
                subject_idx = record.subject_idx
                object_idx = record.object_idx
                if subject_idx < object_idx:
                    context = [text[:subject_idx], text[subject_idx:subject_idx + len(subject)],
                               text[subject_idx + len(subject):object_idx], text[object_idx:object_idx + len(object)],
                               text[object_idx + len(object):]]
                else:
                    context = [text[:object_idx], text[object_idx:object_idx + len(object)],
                               text[object_idx + len(object):subject_idx], text[subject_idx:subject_idx + len(subject)],
                               text[subject_idx + len(subject):]]

                if token_type == 'char':
                    context = [[ch for ch in piece] for piece in context]
                elif token_type == 'word':
                    context = [list(jieba.cut(piece)) for piece in context]

                if subject_idx < object_idx:
                    subject_idx = [len(context[0]), len(context[0]) + len(context[1]) - 1]
                    object_idx = [len(context[0]) + len(context[1]) + len(context[2]),
                                  len(context[0]) + len(context[1]) + len(context[2]) + len(context[3]) - 1]
                else:
                    object_idx = [len(context[0]), len(context[0]) + len(context[1]) - 1]
                    subject_idx = [len(context[0]) + len(context[1]) + len(context[2]),
                                   len(context[0]) + len(context[1]) + len(context[2]) + len(context[3]) - 1]

                pool_mask = [1] * (len(context[0]) + len(context[1])) + [2] * (
                        len(context[2]) + len(context[3])) + [3] * (len(context[4]))
                context = [ch for piece in context for ch in piece]
                if max_inp_len > 0:
                    if (subject_idx[0] >= max_inp_len) or (subject_idx[1] >= max_inp_len) \
                            or (object_idx[0] >= max_inp_len) or (object_idx[1] >= max_inp_len):
                        break
                    pool_mask = pool_mask[:max_inp_len]
                    context = context[:max_inp_len]

                subject_dis = []
                object_dis = []
                for token_idx in range(len(context)):
                    if (token_idx >= subject_idx[0]) and (token_idx <= subject_idx[1]):
                        subject_dis.append(0)
                    else:
                        dis1 = token_idx - subject_idx[0]
                        dis2 = token_idx - subject_idx[1]
                        subject_dis.append(dis1 if abs(dis1) < abs(dis2) else dis2)
                        # subject_dis.append(min(abs(token_idx - subject_idx[0]), abs(token_idx - subject_idx[1])))

                    if (token_idx >= object_idx[0]) and (token_idx <= object_idx[1]):
                        object_dis.append(0)
                    else:
                        dis1 = token_idx - object_idx[0]
                        dis2 = token_idx - object_idx[1]
                        object_dis.append(dis1 if abs(dis1) < abs(dis2) else dis2)
                        # object_dis.append(min(abs(token_idx - object_idx[0]), abs(token_idx - object_idx[1])))

                for dis in subject_dis:
                    if dis >= max_inp_len:
                        print('out')
                for dis in object_dis:
                    if dis >= max_inp_len:
                        print('out')

                bigram = []
                if use_bigram:
                    bigram = [TOKEN_EDGES_START] + context + [TOKEN_EDGES_END]
                    bigram = [[bigram[i - 1] + bigram[i]] + [bigram[i] + bigram[i + 1]] for i in
                              range(1, len(bigram) - 1)]

                context = tokenizer.convert_tokens_to_ids(context)
                if use_bigram:
                    bigram = bigram_tokenizer.convert_tokens_to_ids(bigram)

                bag_context.append(context)
                bag_bigram.append(bigram)
                bag_subject_idx.append(subject_idx)
                bag_object_idx.append(object_idx)
                bag_subject_dis.append(subject_dis)
                bag_object_dis.append(object_dis)
                bag_pool_masks.append(pool_mask)

            self.data.append([bag.relation, bag_context, bag_bigram, bag_subject_idx, bag_object_idx,
                              bag_subject_dis, bag_object_dis, bag_pool_masks])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def dbc_to_sbc(self, ustring):
        rstring = ''
        for uchar in ustring:
            inside_code = ord(uchar)
            if inside_code == 0x3000:
                inside_code = 0x0020
            else:
                inside_code -= 0xfee0
            if not (0x0021 <= inside_code and inside_code <= 0x7e):
                rstring += uchar
                continue
            rstring += chr(inside_code)
        return rstring

    def read_ccks2019(self, data_path, data_split, do_train=False, debug=False):
        sents_dict = {}
        with codecs.open('%s/sent_%s.txt' % (data_path, data_split), 'r', 'utf-8') as fin:
            for line in fin:
                line = line.strip()
                if line == '':
                    continue

                line = line.split('\t')
                if len(line) != 4:
                    continue

                text = line[3].replace(' ', '')
                sents_dict[line[0]] = text

                if debug:
                    if len(sents_dict) >= 1000:
                        break

        bags = []
        with codecs.open('%s/bag_%s.txt' % (data_path, data_split), 'r', 'utf-8') as fin:
            for line in fin:
                line = line.strip()
                if line == '':
                    continue

                line = line.split('\t')
                if len(line) < 5:
                    continue

                if do_train:
                    relation = [int(line[-1])]
                else:
                    relation = [int(x) for x in line[-1].split()]

                subject = self.dbc_to_sbc(line[1])
                object = self.dbc_to_sbc(line[2])
                sents = line[3:-1]

                records = []
                for sent in sents:
                    if sent in sents_dict:
                        sent = self.dbc_to_sbc(sents_dict[sent])
                        try:
                            subject_idx = sent.index(subject)
                            object_idx = sent.index(object)
                        except ValueError:
                            continue

                        records.append(ReBagRecord(sent, subject_idx, object_idx))

                if len(records) > 0:
                    bags.append(ReBag(relation, subject, object, records))

                if debug:
                    if len(bags) >= 10:
                        break

        return bags

    def read_cndbpedia(self, data_path, data_split, debug=False):
        bags = []
        with codecs.open('%s/%s.txt' % (data_path, data_split), 'r', 'utf-8') as fin:
            for line in fin:
                line = line.strip()
                if line == '':
                    continue

                line = json.loads(line)

                relation = [line['relation']]

                subject = self.dbc_to_sbc(line['head'])
                object = self.dbc_to_sbc(line['tail'])
                sents = line['sentences']

                records = []
                for sent in sents:
                    sent = sent.replace(' ', '')
                    sent = self.dbc_to_sbc(sent)
                    try:
                        subject_idx = sent.index(subject)
                        object_idx = sent.index(object)
                    except ValueError:
                        continue

                    records.append(ReBagRecord(sent, subject_idx, object_idx))

                if len(records) > 0:
                    bags.append(ReBag(relation, subject, object, records))

                if debug:
                    if len(bags) >= 10:
                        break

        return bags

    @staticmethod
    def statistics_cndbpedia(datapath):
        logging.info('========== statistics cndbpedia ==========')
        sent_len_dict = {64: 0, 128: 0, 256: 0, 288: 0, 320: 0, 384: 0, 512: 0, 1024: 0}

        data_granularity = ['char', 'word']
        data_split = ['train', 'dev']
        for dg in data_granularity:
            for ds in data_split:
                with codecs.open('%s/%s.txt' % (datapath, ds), 'r', 'utf-8') as fin:
                    for line in fin:
                        line = line.strip()
                        if line == '':
                            continue

                        line = json.loads(line)
                        sents = line['sentences']
                        for text in sents:
                            text = text.replace(' ', '')
                            if dg == 'char':
                                text_len = len(text)
                            elif dg == 'word':
                                text_len = len(list(jieba.cut(text)))

                            for sl in sent_len_dict:
                                if text_len <= sl:
                                    sent_len_dict[sl] = sent_len_dict[sl] + 1
                                    break

                logging.info('%s level %s data count' % (dg, ds))
                logging.info('test length: %s' % str(sent_len_dict))

                for sl in sent_len_dict:
                    sent_len_dict[sl] = 0


class BagGroupDataset(Dataset):
    def __init__(self, task, data_path, data_split, token_type, tokenizer,
                 use_bigram=False, bigram_tokenizer=None, max_inp_len=0, bag_size=5, do_train=False, debug=False):
        super(BagGroupDataset, self).__init__()
        self.data = []
        self.bag_group = {}
        self.bag_size = bag_size

        bags = []
        if task == 'ccks2019_bag':
            bags = self.read_ccks2019(data_path, data_split, do_train, debug)
        elif task == 'cndbpedia':
            bags = self.read_cndbpedia(data_path, data_split, debug)

        for bag in bags:
            subject = bag.subject
            object = bag.object
            bag_context = []
            bag_bigram = []
            bag_subject_idx = []
            bag_object_idx = []
            bag_subject_dis = []
            bag_object_dis = []
            bag_pool_masks = []
            for record in bag.records:
                text = record.text
                subject_idx = record.subject_idx
                object_idx = record.object_idx
                if subject_idx < object_idx:
                    context = [text[:subject_idx], text[subject_idx:subject_idx + len(subject)],
                               text[subject_idx + len(subject):object_idx], text[object_idx:object_idx + len(object)],
                               text[object_idx + len(object):]]
                else:
                    context = [text[:object_idx], text[object_idx:object_idx + len(object)],
                               text[object_idx + len(object):subject_idx], text[subject_idx:subject_idx + len(subject)],
                               text[subject_idx + len(subject):]]

                if token_type == 'char':
                    context = [[ch for ch in piece] for piece in context]
                elif token_type == 'word':
                    context = [list(jieba.cut(piece)) for piece in context]

                if subject_idx < object_idx:
                    subject_idx = [len(context[0]), len(context[0]) + len(context[1]) - 1]
                    object_idx = [len(context[0]) + len(context[1]) + len(context[2]),
                                  len(context[0]) + len(context[1]) + len(context[2]) + len(context[3]) - 1]
                else:
                    object_idx = [len(context[0]), len(context[0]) + len(context[1]) - 1]
                    subject_idx = [len(context[0]) + len(context[1]) + len(context[2]),
                                   len(context[0]) + len(context[1]) + len(context[2]) + len(context[3]) - 1]

                pool_mask = [1] * (len(context[0]) + len(context[1])) + [2] * (
                        len(context[2]) + len(context[3])) + [3] * (len(context[4]))
                context = [ch for piece in context for ch in piece]
                if max_inp_len > 0:
                    if (subject_idx[0] >= max_inp_len) or (subject_idx[1] >= max_inp_len) \
                            or (object_idx[0] >= max_inp_len) or (object_idx[1] >= max_inp_len):
                        break
                    pool_mask = pool_mask[:max_inp_len]
                    context = context[:max_inp_len]

                subject_dis = []
                object_dis = []
                for token_idx in range(len(context)):
                    if (token_idx >= subject_idx[0]) and (token_idx <= subject_idx[1]):
                        subject_dis.append(0)
                    else:
                        dis1 = token_idx - subject_idx[0]
                        dis2 = token_idx - subject_idx[1]
                        subject_dis.append(dis1 if abs(dis1) < abs(dis2) else dis2)
                        # subject_dis.append(min(abs(token_idx - subject_idx[0]), abs(token_idx - subject_idx[1])))

                    if (token_idx >= object_idx[0]) and (token_idx <= object_idx[1]):
                        object_dis.append(0)
                    else:
                        dis1 = token_idx - object_idx[0]
                        dis2 = token_idx - object_idx[1]
                        object_dis.append(dis1 if abs(dis1) < abs(dis2) else dis2)
                        # object_dis.append(min(abs(token_idx - object_idx[0]), abs(token_idx - object_idx[1])))

                bigram = []
                if use_bigram:
                    bigram = [TOKEN_EDGES_START] + context + [TOKEN_EDGES_END]
                    bigram = [[bigram[i - 1] + bigram[i]] + [bigram[i] + bigram[i + 1]] for i in
                              range(1, len(bigram) - 1)]

                context = tokenizer.convert_tokens_to_ids(context)
                if use_bigram:
                    bigram = bigram_tokenizer.convert_tokens_to_ids(bigram)

                bag_context.append(context)
                bag_bigram.append(bigram)
                bag_subject_idx.append(subject_idx)
                bag_object_idx.append(object_idx)
                bag_subject_dis.append(subject_dis)
                bag_object_dis.append(object_dis)
                bag_pool_masks.append(pool_mask)

            if not bag.relation[0] in self.bag_group:
                self.bag_group[bag.relation[0]] = []
            self.bag_group[bag.relation[0]].append([bag.relation, bag_context, bag_bigram,
                                                    bag_subject_idx, bag_object_idx, bag_subject_dis, bag_object_dis,
                                                    bag_pool_masks])

        self.init_data()

    def init_data(self):
        self.data = []
        for rela in self.bag_group:
            bags = self.bag_group[rela]
            shuffle(bags)

            group = []
            for i in range(len(bags)):
                if (i > 0) and ((i % self.bag_size) == 0):
                    self.data.append([rela, group])
                    group = []
                group.append(bags[i])

            self.data.append([rela, group])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def dbc_to_sbc(self, ustring):
        rstring = ''
        for uchar in ustring:
            inside_code = ord(uchar)
            if inside_code == 0x3000:
                inside_code = 0x0020
            else:
                inside_code -= 0xfee0
            if not (0x0021 <= inside_code and inside_code <= 0x7e):
                rstring += uchar
                continue
            rstring += chr(inside_code)
        return rstring

    def read_ccks2019(self, data_path, data_split, do_train=False, debug=False):
        sents_dict = {}
        with codecs.open('%s/sent_%s.txt' % (data_path, data_split), 'r', 'utf-8') as fin:
            for line in fin:
                line = line.strip()
                if line == '':
                    continue

                line = line.split('\t')
                if len(line) != 4:
                    continue

                text = line[3].replace(' ', '')
                sents_dict[line[0]] = text

                if debug:
                    if len(sents_dict) >= 1000:
                        break

        bags = []
        with codecs.open('%s/bag_%s.txt' % (data_path, data_split), 'r', 'utf-8') as fin:
            for line in fin:
                line = line.strip()
                if line == '':
                    continue

                line = line.split('\t')
                if len(line) < 5:
                    continue

                if do_train:
                    relation = [int(line[-1])]
                else:
                    relation = [int(x) for x in line[-1].split()]

                subject = self.dbc_to_sbc(line[1])
                object = self.dbc_to_sbc(line[2])
                sents = line[3:-1]

                records = []
                for sent in sents:
                    if sent in sents_dict:
                        sent = self.dbc_to_sbc(sents_dict[sent])
                        try:
                            subject_idx = sent.index(subject)
                            object_idx = sent.index(object)
                        except ValueError:
                            continue

                        records.append(ReBagRecord(sent, subject_idx, object_idx))

                if len(records) > 0:
                    bags.append(ReBag(relation, subject, object, records))

                if debug:
                    if len(bags) >= 50:
                        break

        return bags

    def read_cndbpedia(self, data_path, data_split, debug=False):
        bags = []
        with codecs.open('%s/%s.txt' % (data_path, data_split), 'r', 'utf-8') as fin:
            for line in fin:
                line = line.strip()
                if line == '':
                    continue

                line = json.loads(line)

                relation = [line['relation']]

                subject = self.dbc_to_sbc(line['head'])
                object = self.dbc_to_sbc(line['tail'])
                sents = line['sentences']

                records = []
                for sent in sents:
                    sent = sent.replace(' ', '')
                    sent = self.dbc_to_sbc(sent)
                    try:
                        subject_idx = sent.index(subject)
                        object_idx = sent.index(object)
                    except ValueError:
                        continue

                    records.append(ReBagRecord(sent, subject_idx, object_idx))

                if len(records) > 0:
                    bags.append(ReBag(relation, subject, object, records))

                if debug:
                    if len(bags) >= 10:
                        break

        return bags


class Tokenizer:
    def __init__(self, token_to_id):
        self.token_to_id = token_to_id
        self.id_to_token = {v: k for k, v in token_to_id.items()}

    def convert_tokens_to_ids(self, tokens, unk_token=TOKEN_UNK):
        ids = []
        for token in tokens:
            if isinstance(token, str):
                ids.append(self.token_to_id.get(token, self.token_to_id[unk_token]))
            else:
                ids.append([self.token_to_id.get(t, self.token_to_id[unk_token]) for t in token])
        return ids

    def convert_ids_to_tokens(self, ids, max_sent_len=0):
        tokens = [self.id_to_token[i] for i in ids]
        if max_sent_len > 0:
            tokens = tokens[:max_sent_len]
        return tokens


def load_relation_schema(task, file_path, add_na=False):
    relations = []
    if task == 'duie':
        if add_na:
            relations.append('NA')
        with codecs.open('%s/schema.txt' % file_path, 'r', 'utf-8') as fin:
            for line in fin:
                line = line.strip()
                if line == '':
                    continue

                schema = json.loads(line)
                predicate = schema['predicate']
                relations.append(predicate)
    elif (task == 'ccks2019') or (task == 'ccks2019_bag'):
        with codecs.open('%s/schema.txt' % file_path, 'r', 'utf-8') as fin:
            for line in fin:
                line = line.strip()
                if line == '':
                    continue

                line = line.split()
                predicate = line[0]
                relations.append(predicate)
    elif task == 'cndbpedia':
        with codecs.open('%s/schema.txt' % file_path, 'r', 'utf-8') as fin:
            for line in fin:
                line = line.strip()
                if line == '':
                    continue

                predicate = line
                relations.append(predicate)

    rela_to_id = {}
    for rela in relations:
        rela_to_id[rela] = len(rela_to_id)

    id_to_rela = {v: k for k, v in rela_to_id.items()}
    return rela_to_id, id_to_rela


def load_pretrain_embedding(filepath, has_meta=False, add_pad=False, pad_token=TOKEN_PAD,
                            add_unk=False, unk_token=TOKEN_UNK, add_entity_mark=False, debug=False):
    with codecs.open(filepath, 'r', 'utf-8', errors='ignore') as fin:
        token_to_id = {}
        embed = []

        if has_meta:
            meta_info = fin.readline().strip().split()

        first_line = fin.readline().strip().split()
        embed_size = len(first_line) - 1

        if add_pad:
            token_to_id[pad_token] = len(token_to_id)
            embed.append([0.] * embed_size)

        if add_unk:
            token_to_id[unk_token] = len(token_to_id)
            embed.append([0.] * embed_size)

        if add_entity_mark:
            token_to_id[TOKEN_SUBJECT_START] = len(token_to_id)
            embed.append([0.] * embed_size)
            token_to_id[TOKEN_SUBJECT_END] = len(token_to_id)
            embed.append([0.] * embed_size)
            token_to_id[TOKEN_OBJECT_START] = len(token_to_id)
            embed.append([0.] * embed_size)
            token_to_id[TOKEN_OBJECT_END] = len(token_to_id)
            embed.append([0.] * embed_size)

        token_to_id[first_line[0]] = len(token_to_id)
        embed.append([float(x) for x in first_line[1:]])

        for line in fin:
            line = line.split()

            if len(line) != embed_size + 1:
                continue
            if line[0] in token_to_id:
                continue

            token_to_id[line[0]] = len(token_to_id)
            embed.append([float(x) for x in line[1:]])

            if debug:
                if len(embed) >= 1000:
                    break

    return token_to_id, embed


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s - %(levelname)s: %(message)s', level=logging.INFO)

    task = {'name': 'duie', 'datapath': '../data/datasets/duie'}
    task = {'name': 'ccks2019', 'datapath': '../data/datasets/ccks2019'}
    task = {'name': 'cndbpedia', 'datapath': '../data/datasets/cndbpedia'}

    # ReDataset.transform_ccks2019('../data/datasets/open_data', '../data/datasets/ccks2019')

    # rela_to_id, id_to_rela = load_relation_schema(task['name'], task['datapath'])

    # ReDataset(task['name'], '%s/dev.txt' % task['datapath'], 'char', None, rela_to_id,
    #           use_bigram=False, bigram_tokenizer=None,
    #           max_inp_len=0, do_pad=False, pad_token=TOKEN_PAD,
    #           do_to_id=False, do_sort=False, do_train=False, for_bert=False, debug=True)

    # ReDataset.statistics_duie('../data/datasets/duie')
    # ReDataset.statistics_ccks2019('../data/datasets/ccks2019')
    # DistantDataset.statistics_cndbpedia('../data/datasets/cndbpedia')
