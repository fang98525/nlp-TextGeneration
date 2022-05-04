# -*- coding: utf-8 -*-
# @Time : 2021/3/18
# @Author : Chile
# @Email : realchilewang@foxmail.com
# @File : utils.py
# @Software: PyCharm

import os
import time
from tqdm import tqdm
import torch
from config import Config
import random
import logging
from NEZHA.model_NEZHA import NEZHAConfig
from NEZHA import NEZHA_utils
from model import BertSeq2SeqModel
from utils import DataIterator
import numpy as np
from transformers import BertTokenizer, RobertaConfig, AlbertConfig
from optimization import BertAdam
from rouge import Rouge

gpu_id = 0
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
result_data_dir = Config().result_file
print('GPU ID: ', str(gpu_id))
print('Fine Tune Learning Rate: ', Config().embed_learning_rate)
print('Batch Size: ', Config().batch_size)
print('Use original bert', Config().pretrainning_model)

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

config = Config()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

n_gpu = 0
# n_gpu = torch.cuda.device_count()

# 固定每次结果
seed = 156421
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)


def list2ts2device(target_list):
    """把utils的数据写入到gpu"""
    target_ts = torch.from_numpy(np.array(target_list))
    return target_ts.to(device)


def beam_search(params, model, ori_token_ids, ori_token_type_ids, beam_size=1, tgt_seq_len=30):
    """
    非马尔科夫过程，用beam search解码，局部最优解
    Args:
        ori_token_ids: input ids. (1, src_seq_len)
        ori_token_type_ids: (1, src_seq_len)
        beam_size: size of beam search.
        tgt_seq_len: 生成序列最大长度

    Returns:
        output_ids: ([tgt_seq_len],)
    """
    device = params.device
    sep_id = params.sep_id

    # 用来保存输出序列. (beam_size, 0)
    output_ids = torch.empty((beam_size, 0), dtype=torch.long, device=device)
    # 用来保存累计得分. (beam_size, 1)
    output_scores = torch.zeros((beam_size, 1), device=device)
    # 重复beam-size次. (beam_size, bs * src_seq_len)
    ori_token_ids = ori_token_ids.view(1, -1).repeat(beam_size, 1)
    ori_token_type_ids = ori_token_type_ids.view(1, -1).repeat(beam_size, 1)

    with torch.no_grad():
        for step in range(tgt_seq_len):
            # 第一次迭代
            if step == 0:
                input_ids = ori_token_ids
                token_type_ids = ori_token_type_ids

            # (beam_size, vocab_size)
            scores = model(input_ids, token_type_ids=token_type_ids)
            _, vocab_size = scores.size()

            # 累计得分. (beam_size, vocab_size)
            output_scores = output_scores.view(-1, 1) + scores
            # 确定topk的beam，并获得它们的索引
            # (beam_size,)
            hype_score, hype_pos = torch.topk(output_scores.view(-1), beam_size)
            # 行索引. (beam_size,)
            row_id = (hype_pos // vocab_size)
            # 列索引. (beam_size, 1)
            column_id = (hype_pos % vocab_size).long().reshape(-1, 1)

            # 本次迭代的得分和输出
            # 更新得分. (beam_size,)
            output_scores = hype_score
            # (beam_size, [tgt_seq_len])
            output_ids = torch.cat([output_ids[row_id], column_id], dim=1).long()

            # 下一次迭代的input和token type
            # (beam_size, src_seq_len + [tgt_seq_len])
            input_ids = torch.cat([ori_token_ids, output_ids], dim=1)
            # (beam_size, src_seq_len + [tgt_seq_len])
            token_type_ids = torch.cat([ori_token_type_ids, torch.ones_like(output_ids)], dim=1)

            # 统计每个beam出现的end标记. (beam_size,)
            end_counts = (output_ids == sep_id).sum(dim=1)
            # 最高得分的beam位置
            best_one = output_scores.argmax()
            # 该beam已完成且累计得分最高，直接返回
            if end_counts[best_one] == 1:
                return output_ids[best_one]
            # 将已完成但得分低的beam移除
            else:
                # 标记未完成序列
                flag = (end_counts < 1)
                # 只要有未完成的就为True
                if not flag.all():
                    ori_token_ids = ori_token_ids[flag]
                    ori_token_type_ids = ori_token_type_ids[flag]
                    input_ids = input_ids[flag]
                    token_type_ids = token_type_ids[flag]
                    output_ids = output_ids[flag]
                    output_scores = output_scores[flag]
                    beam_size = flag.sum()  # beam_size相应变化
        # 循环结束未完成则返回得分最高的beam
        return output_ids[output_scores.argmax()]


def train(train_iter, test_iter, config):
    """"""
    # Prepare model
    # Prepare model
    # reload weights from restore_file if specified
    if config.pretrainning_model == 'nezha':
        Bert_config = NEZHAConfig.from_json_file(config.bert_config_file)
        model = BertSeq2SeqModel(config=Bert_config, params=config)
        NEZHA_utils.torch_init_model(model, config.bert_file)
    elif config.pretrainning_model == 'albert':
        Bert_config = AlbertConfig.from_pretrained(config.model_path)
        model = BertSeq2SeqModel.from_pretrained(config.model_path, config=Bert_config)
    else:
        Bert_config = RobertaConfig.from_pretrained(config.model_path, output_hidden_states=True)
        model = BertSeq2SeqModel.from_pretrained(config.model_path, config=Bert_config)
    Bert_config.output_hidden_states = True  # 获取每一层的输出
    model.to(device)

    """多卡训练"""
    if n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Prepare optimizer
    # fine-tuning
    # 取模型权重
    param_optimizer = list(model.named_parameters())
    # pretrain model param
    param_pre = [(n, p) for n, p in param_optimizer if 'bert' in n]
    # downstream model param
    param_downstream = [(n, p) for n, p in param_optimizer if
                        not any([s in n for s in ('bert', 'crf', 'electra')]) or 'dym_weight' in n or 'decoder' in n]
    # 不进行衰减的权重
    no_decay = ['bias', 'LayerNorm', 'dym_weight', 'layer_norm']
    # 将权重分组
    optimizer_grouped_parameters = [
        # pretrain model param
        # 衰减
        {'params': [p for n, p in param_pre if not any(nd in n for nd in no_decay)],
         'weight_decay': config.decay_rate, 'lr': config.embed_learning_rate
         },
        # 不衰减
        {'params': [p for n, p in param_pre if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0, 'lr': config.embed_learning_rate
         },
        # middle model
        # 衰减
        {'params': [p for n, p in param_downstream if not any(nd in n for nd in no_decay)],
         'weight_decay': config.decay_rate, 'lr': config.learning_rate
         },
        # 不衰减
        {'params': [p for n, p in param_downstream if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0, 'lr': config.learning_rate
         },
    ]
    num_train_optimization_steps = train_iter.num_records // config.gradient_accumulation_steps * config.train_epoch
    optimizer = BertAdam(optimizer_grouped_parameters, warmup=config.warmup_proportion, schedule="warmup_cosine",
                         t_total=num_train_optimization_steps)

    logger.info("***** Running training *****")
    # logger.info("  Num examples = %d", len(train_examples))
    logger.info("  Batch size = %d", config.batch_size)
    logger.info("  Num epochs = %d", config.train_epoch)
    logger.info("  Learning rate = %f", config.learning_rate)

    cum_step = 0
    best_acc = 0.5
    timestamp = str(int(time.time()))
    out_dir = os.path.abspath(
        os.path.join(config.save_model, "runs_" + str(gpu_id), timestamp))
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    print("Writing to {}\n".format(out_dir))
    for i in range(config.train_epoch):
        print("当前epoch为{}....".format(i))
        model.train()
        for batch in tqdm(train_iter):
            # 转成张量
            batch = tuple(t.to(config.device) for t in batch)
            input_ids_list, segment_ids_list, label_ids_list = batch
            _,loss = model(input_ids=input_ids_list, token_type_ids=segment_ids_list, labels=label_ids_list)
            if n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu.
            # 梯度累加
            if config.gradient_accumulation_steps > 1:
                loss = loss / config.gradient_accumulation_steps

            if cum_step % 100 == 0:
                format_str = 'step {}, loss {:.4f} lr {:.5f}'
                print(
                    format_str.format(
                        cum_step, loss.item(), config.learning_rate)
                )
            if config.flooding:
                loss = (loss - config.flooding).abs() + config.flooding  # 让loss趋于某个值收敛

            loss.backward()  # 反向传播，得到正常的grad

            if (cum_step + 1) % config.gradient_accumulation_steps == 0:
                # performs updates using calculated gradients
                optimizer.step()
                model.zero_grad()
            cum_step += 1
        if i != 0 and i % 4 == 0:   #没4次验证评估一次计算 rough-L
            val_metrics = set_test(model, test_iter)
            f1 = val_metrics['rouge-l']['f']
        # lr_scheduler学习率递减 step

            print('dev set : step_{},F1_{}'.format(cum_step, f1))
            if f1 > best_acc:
                # Save a trained model
                best_acc = f1
                model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
                output_model_file = os.path.join(
                    os.path.join(out_dir, 'model_{:.4f}_{}'.format(f1, str(cum_step))))
                torch.save(model_to_save, output_model_file)


def set_test(model, test_iter):
    if not test_iter.is_test:
        test_iter.is_test = True
    rouge = Rouge()
    # a running average object for loss
    # word id to word
    idx2word = tokenizer.ids_to_tokens

    pred = []
    ground_truth = []
    model.eval()
    # bs=1
    for batch in tqdm(test_iter):
        # to device
        batch = tuple(t.to(config.device) for t in batch)
        input_ids_list, segment_ids_list, labels = batch

        # inference
        with torch.no_grad():
            # inference
            # 构造predict时的输入
            input_sep_index = torch.nonzero((input_ids_list == config.sep_id), as_tuple=True)[-1][-1]
            input_ids_list = input_ids_list[:, :input_sep_index + 1]
            segment_ids_list = segment_ids_list[:, :input_sep_index + 1]
            batch_output = beam_search(config, model, input_ids_list, segment_ids_list,
                                       beam_size=config.beam_size,
                                       tgt_seq_len=config.tgt_seq_len)

        # List[str]，去掉[CLS]
        # 获取有效label
        start_sep_id = torch.nonzero((labels == config.sep_id), as_tuple=True)[-1][-2]
        end_sep_id = torch.nonzero((labels == config.sep_id), as_tuple=True)[-1][-1]
        labels = labels.view(-1).to('cpu').numpy().tolist()
        labels = labels[start_sep_id + 1:end_sep_id + 1]
        batch_output = batch_output.view(-1).to('cpu').numpy().tolist()

        ground_truth.append(' '.join([idx2word.get(indices) for indices in labels]))
        pred.append(' '.join([idx2word.get(indices) for indices in batch_output]))
        print()
        print('true_text:', ''.join([idx2word.get(indices) for indices in labels]))
        print('pred_text:', ''.join([idx2word.get(indices) for indices in batch_output]))

    # logging loss, rouge-?
    rouge_dict = rouge.get_scores(pred, ground_truth, avg=True)
    metrics = {
        'rouge-1': rouge_dict['rouge-1'],
        'rouge-2': rouge_dict['rouge-2'],
        'rouge-l': rouge_dict['rouge-l']
    }
    metrics_str = "\n".join("{}: {}".format(k, v) for k, v in metrics.items())
    logging.info("- metrics: \n" + metrics_str)

    return metrics


def softmax(x, axis=1):
    """
    自写函数定义softmax
    :param x:
    :param axis:
    :return:
    """
    # 计算每行的最大值
    row_max = x.max(axis=axis)

    # 每行元素都需要减去对应的最大值，否则求exp(x)会溢出，导致inf情况
    row_max = row_max.reshape(-1, 1)
    x = x - row_max
    # 计算e的指数次幂
    x_exp = np.exp(x)
    x_sum = np.sum(x_exp, axis=axis, keepdims=True)
    s = x_exp / x_sum
    return s


if __name__ == '__main__':
    config = Config()
    # config.device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = BertTokenizer.from_pretrained(pretrained_model_name_or_path=config.model_path,
                                              do_lower_case=True,
                                              never_split=["[UNK]", "[SEP]", "[PAD]", "[CLS]", "[MASK]"])
    config.sep_id = tokenizer._convert_token_to_id('[SEP]')
    train_iter = DataIterator(config.batch_size,
                              data_file=config.processed_data + 'train.json',
                              use_bert=config.pretrainning_model,
                              tokenizer=tokenizer, seq_length=config.sequence_length)
    dev_iter = DataIterator(1, data_file=config.processed_data + 'val.json',
                            use_bert=config.pretrainning_model,
                            seq_length=config.sequence_length, is_test=True, tokenizer=tokenizer)
    train(train_iter, dev_iter, config=config)
