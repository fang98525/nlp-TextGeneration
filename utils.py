# -*- coding: utf-8 -*-
# @Time : 2021/3/18
# @Author : Chile
# @Email : realchilewang@foxmail.com
# @File : utils.py
import numpy as np
from transformers import BertTokenizer
from tqdm import tqdm
from config import Config
import json
import torch

config = Config()


"""
构建数据pipline  也就是dataloder
"""


def load_data(data_file):
    """
    读取数据
    :param file:
    :return:
    """
    examples = []
    # read src data
    with open(data_file, "r", encoding='utf-8') as f:
        data = json.load(f)
        for en_sample in data:
            for sin_sample in en_sample['annotations']:
                text = list(sin_sample['A'].strip()) + ['✂'] + list(en_sample['text'].strip())
                tag = list(sin_sample['Q'].strip())
                examples.append((text, tag))
    return examples


def create_example(lines):
    examples = []
    for (i, line) in enumerate(lines):
        text = line[0]
        tag = line[1]
        examples.append(InputExample(text=text, tag=tag))
    return examples


def get_examples(data_file):
    return create_example(
        load_data(data_file)
    )


class InputExample(object):
    """a single set of samples of data_src
    """

    def __init__(self, text, tag):
        self.text = text
        self.tag = tag


class DataIterator:
    """
    数据迭代器
    """

    def __init__(self, batch_size, data_file, tokenizer, use_bert=False, seq_length=100, is_test=False, task='iflytek'):
        self.data_file = data_file
        self.data = get_examples(data_file)
        self.batch_size = batch_size
        self.use_bert = use_bert
        self.seq_length = seq_length
        self.num_records = len(self.data)
        self.all_tags = []
        self.idx = 0  # 数据索引
        self.all_idx = list(range(self.num_records))  # 全体数据索引
        self.is_test = is_test
        self.task=task
        # self._tril_matrix = torch.tril(torch.ones((self.seq_length, self.seq_length), dtype=torch.long))
        if not self.is_test:
            self.shuffle()
        self.tokenizer = tokenizer
        print(self.num_records)

    def convert_single_example(self, example_idx):
        # tokenize返回为空则设为[UNK]
        """
        这里构造每一个样本
        """

        example = self.data[example_idx]
        # 构造tokens
        # 加入text
        text_tokens = ['[CLS]']
        tag_tokens = []
        for token in example.text:
            if len(self.tokenizer.tokenize(token)) == 1:
                if token == '✂':
                    text_tokens.append('[SEP]')
                else:
                    text_tokens.append(self.tokenizer.tokenize(token)[0])
            else:
                text_tokens.append('[UNK]')
        # 加入tag
        for token in example.tag + ['[SEP]']:
            if len(self.tokenizer.tokenize(token)) == 1:
                tag_tokens.append(self.tokenizer.tokenize(token)[0])
            else:
                tag_tokens.append('[UNK]')
        # cut off
        if len(text_tokens + ['[SEP]'] + tag_tokens) > self.seq_length:
            cut_len = len(text_tokens + tag_tokens) + 1 - self.seq_length
            text_tokens = text_tokens[:-cut_len]
        # [CLS] A A A [SEP] B B B [SEP]
        input_tokens = text_tokens + ['[SEP]'] + tag_tokens
        token_type_ids = [0] * (len(text_tokens) + 1)
        token_type_ids.extend([1] * len(tag_tokens))  # [SEP]也融入预测

        # 输入文本：[CLS] A A A [SEP] B B B
        input_ids = self.tokenizer.convert_tokens_to_ids(input_tokens[:-1])  # [CLS] A A A [SEP] B B B
        # 输出文本：A A A [SEP] B B B [SEP]
        tag_ids = self.tokenizer.convert_tokens_to_ids(input_tokens[1:])  # A A A [SEP] B B B [SEP]
        assert len(input_ids) == len(tag_ids), 'length of input_ids should be the same as tag_ids!'

        # zero-padding up to the sequence length
        if len(input_ids) < self.seq_length:
            # 补零
            pad_len = self.seq_length - len(input_ids)
            input_ids += [0] * pad_len
            tag_ids += [0] * pad_len
            token_type_ids += [0] * (pad_len - 1)

        assert len(input_ids) == len(token_type_ids)
        return input_ids, token_type_ids, tag_ids

    def shuffle(self):
        np.random.shuffle(self.all_idx)

    def __iter__(self):
        return self

    def __next__(self):
        if self.idx >= self.num_records:  # 迭代停止条件
            self.idx = 0
            if not self.is_test:
                self.shuffle()
            raise StopIteration

        input_ids_list = []
        segment_ids_list = []
        label_list = []
        num_tags = 0
        while num_tags < self.batch_size:  # 每次返回batch_size个数据
            idx = self.all_idx[self.idx]
            res = self.convert_single_example(idx)
            if res is None:
                self.idx += 1
                if self.idx >= self.num_records:
                    break
                continue
            input_ids, segment_ids, labels = res

            # 一个batch的输入
            input_ids_list.append(input_ids)
            segment_ids_list.append(segment_ids)
            label_list.append(labels)
            if self.use_bert:
                num_tags += 1

            self.idx += 1
            if self.idx >= self.num_records:
                break
        #转换为tensor张量
        input_ids_list=torch.from_numpy(np.array(input_ids_list))
        segment_ids_list=torch.from_numpy(np.array(segment_ids_list))
        label_list=torch.from_numpy(np.array(label_list))

        return input_ids_list, segment_ids_list, label_list

if __name__ == '__main__':
    config = Config()
    tokenizer = BertTokenizer.from_pretrained(pretrained_model_name_or_path=config.model_path,
                                              do_lower_case=True,
                                              never_split=["[UNK]", "[SEP]", "[PAD]", "[CLS]", "[MASK]"])
    train_iter = DataIterator(config.batch_size,
                              data_file=config.processed_data + 'train.json',
                              use_bert=config.use_bert,
                              tokenizer=tokenizer, seq_length=config.sequence_length)
    dev_iter = DataIterator(config.batch_size, data_file=config.processed_data + 'val.json',
                            use_bert=config.use_bert,
                            seq_length=config.sequence_length, is_test=True, tokenizer=tokenizer)
    for input_ids_list, segment_ids_list, labels_list in tqdm(train_iter):
        print(input_ids_list[-1])
        print(segment_ids_list[-1])
        print(labels_list[-1])

        break

    for input_ids_list, segment_ids_list, labels_list in tqdm(dev_iter):
        print(input_ids_list[-1])
        print(segment_ids_list[-1])
        print(labels_list[-1])

        break

