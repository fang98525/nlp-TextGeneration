# /usr/bin/env python
# coding=utf-8
# @Time : 2021/3/18
# @Author : Chile
# @Email : realchilewang@foxmail.com
# @File : utils.py
# @Software: PyCharm
"""Evaluate the model"""
import logging
from tqdm import tqdm
import torch
import utils
from rouge import Rouge


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
                # flag矩阵中，False代表继续迭代，True代表已经完成
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


def evaluate(args, model, tokenizer, data_iterator, params, mark='Val'):
    """Evaluate the model on `steps` batches."""
    # set model to evaluation mode
    model.eval()
    # rouge score
    rouge = Rouge()
    # a running average object for loss
    loss_avg = utils.RunningAverage()
    # word id to word
    idx2word = tokenizer.ids_to_tokens
    pred = []
    ground_truth = []
    # bs=1
    for batch in tqdm(data_iterator, unit='Batch', ascii=True):
        # to device
        batch = tuple(t.to(params.device) for t in batch)
        input_ids, labels, token_type_ids = batch

        # inference
        with torch.no_grad():
            # get loss
            _, loss = model(input_ids, labels=labels, token_type_ids=token_type_ids)
            if params.n_gpu > 1 and args.multi_gpu:
                loss = loss.mean()  # mean() to average on multi-gpu.
            # update the average loss
            loss_avg.update(loss.item())

            # inference
            # 构造predict时的输入
            input_sep_index = torch.nonzero((input_ids == params.sep_id), as_tuple=True)[-1][-1]
            input_ids = input_ids[:, :input_sep_index + 1]
            token_type_ids = token_type_ids[:, :input_sep_index + 1]
            batch_output = beam_search(params, model, input_ids, token_type_ids,
                                       beam_size=params.beam_size,
                                       tgt_seq_len=params.tgt_seq_len)

        # List[str]，去掉[CLS]
        # 获取有效label
        start_sep_id = torch.nonzero((labels == params.sep_id), as_tuple=True)[-1][-2]
        end_sep_id = torch.nonzero((labels == params.sep_id), as_tuple=True)[-1][-1]
        labels = labels.view(-1).to('cpu').numpy().tolist()
        labels = labels[start_sep_id + 1:end_sep_id + 1]
        batch_output = batch_output.view(-1).to('cpu').numpy().tolist()

        ground_truth.append(' '.join([idx2word.get(indices) for indices in labels]))
        pred.append(' '.join([idx2word.get(indices) for indices in batch_output]))
        print('true_text:', ''.join([idx2word.get(indices) for indices in labels]))
        print('pred_text:', ''.join([idx2word.get(indices) for indices in batch_output]))

    # logging loss, rouge-?
    rouge_dict = rouge.get_scores(pred, ground_truth, avg=True)
    metrics = {
        'loss': loss_avg(),
        'rouge-1': rouge_dict['rouge-1'],
        'rouge-2': rouge_dict['rouge-2'],
        'rouge-l': rouge_dict['rouge-l']
    }
    metrics_str = "\n".join("{}: {}".format(k, v) for k, v in metrics.items())
    logging.info("- {} metrics: \n".format(mark) + metrics_str)

    return metrics
