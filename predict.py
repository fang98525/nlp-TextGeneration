# /usr/bin/env python
# coding=utf-8
# @Time : 2021/3/18
# @Author : Chile
# @Email : realchilewang@foxmail.com
# @File : utils.py
# @Software: PyCharm
"""Predict"""

from evaluate import beam_search
import torch
from tqdm import tqdm
from config import Config
from transformers import BertTokenizer
from utils import DataIterator
import logging
import os
import numpy as np

gpu_id = 3
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
result_data_dir = Config().result_file
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def list2ts2device(target_list):
    """把utils的数据写入到gpu"""
    target_ts = torch.from_numpy(np.array(target_list))
    return target_ts.to(device)


def predict(test_iter, model_file):
    model = torch.load(model_file)
    device = config.device
    model.to(device)
    logger.info("***** Running Prediction *****")
    logger.info("  Predict Path = %s", model_file)
    idx2word = tokenizer.ids_to_tokens
    model.eval()
    pred = []
    for input_ids_list, segment_ids_list, label_ids_list, seq_length in tqdm(test_iter):
        input_ids, labels, token_type_ids = list2ts2device(input_ids_list), list2ts2device(
                                            label_ids_list), list2ts2device(token_type_ids)
        # inference
        with torch.no_grad():
            # inference
            # 构造predict时的输入
            input_sep_index = torch.nonzero((input_ids == config.sep_id), as_tuple=True)[-1][-1]
            input_ids = input_ids[:, :input_sep_index + 1]
            token_type_ids = token_type_ids[:, :input_sep_index + 1]
            batch_output = beam_search(config, model, input_ids, token_type_ids,
                                       beam_size=config.beam_size,
                                       tgt_seq_len=config.tgt_seq_len)

        batch_output = batch_output.view(-1).to('cpu').numpy().tolist()
        pred.append(''.join([idx2word.get(indices) for indices in batch_output]))

    # write to file
    with open(config.result_file / 'result.txt', 'w', encoding='utf-8') as f:
        for p in pred:
            f.write(p.replace('[SEP]', '') + '\n')


if __name__ == '__main__':
    config = Config()
    tokenizer = BertTokenizer.from_pretrained(pretrained_model_name_or_path=config.model_path,
                                              do_lower_case=True,
                                              never_split=["[UNK]", "[SEP]", "[PAD]", "[CLS]", "[MASK]"])
    config.sep_id = tokenizer._convert_token_to_id('[SEP]')
    config.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Predicting test.txt..........')
    dev_iter = DataIterator(1,
                            config.processed_data + 'val.json',
                            use_bert=config.pretrainning_model,
                            seq_length=config.sequence_length, is_test=True, tokenizer=tokenizer)
    predict(dev_iter, config.checkpoint_path)

