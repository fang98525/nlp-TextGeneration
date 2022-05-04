# /usr/bin/env python
# coding=utf-8
# @Time : 2021/3/18
# @Author : Chile
# @Email : realchilewang@foxmail.com
# @File : utils.py
# @Software: PyCharm
"""model"""
import torch
import torch.nn as nn
import math
from NEZHA.model_NEZHA import NEZHAModel
from transformers import BertPreTrainedModel, BertModel


###常见的几种激活函数之一
def mish(x):
    return x * torch.tanh(nn.functional.softplus(x))


def swish(x):
    return x * torch.sigmoid(x)


def gelu(x):
    """Implementation of the gelu activation function currently in Google Bert repo (identical to OpenAI GPT).
    Also see https://arxiv.org/abs/1606.08415
    """
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))


ACT2FN = {"gelu": gelu, "relu": torch.nn.functional.relu, "swish": swish, "mish": mish}


class BertPredictionHeadTransform(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        if isinstance(config.hidden_act, str):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class BertLMPredictionHead(nn.Module):
    def __init__(self, config, bert_model_embedding_weights):
        """
        LM model decoder
        Args:
            bert_model_embedding_weights: Bert Embedding层参数，用于初始化decoder的Linear层
        """
        super().__init__()
        # (vocab_size, hidden_size)
        self.bert_model_embedding_weights = bert_model_embedding_weights
        # init bias for Linear layer of decoder
        self.bias = nn.Parameter(torch.zeros(config.vocab_size), requires_grad=True)

        self.transform = BertPredictionHeadTransform(config)
        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.init_weights()

    def init_weights(self):
        self.decoder.weight = self.bert_model_embedding_weights
        # Need a link between the two variables so that the bias is correctly resized with `resize_token_embeddings`
        self.decoder.bias = self.bias

    def forward(self, hidden_states):
        """
        Args:
            hidden_states: encoder output. (bs, seq_len, hidden_size)

        Returns:
            hidden_states: decoder output. (bs, seq_len, vocab_size)
        """
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states


class BertSeq2SeqModel(BertPreTrainedModel):
    def __init__(self, config, params):
        super().__init__(config)
        # seq2seq encoder(share the weight of pre-train model)
        self.pre_model_type = params.pre_model_type
        if self.pre_model_type.lower() == 'nezha':
            self.bert = NEZHAModel(config)
        elif self.pre_model_type.lower() == 'roberta':
            self.bert = BertModel(config)
        else:
            raise ValueError('Pre-train Model type must be NEZHA or RoBERTa!')
        # seq2seq decoder(share the weight of pre-train model)
        self.decoder = BertLMPredictionHead(config, self.bert.embeddings.word_embeddings.weight)

        # 动态权重
        self.fusion_layers = params.fusion_layers
        self.dym_weight = nn.Parameter(torch.ones((self.fusion_layers, 1, 1, 1)),
                                       requires_grad=True)
        self.vocab_size = config.vocab_size

        self.reset_params()

    def reset_params(self):
        # init weights
        self.init_weights()
        nn.init.xavier_normal_(self.dym_weight)

    def get_dym_layer(self, outputs):
        """获取动态权重融合后的BERT output.
        Args:
            outputs: origin bert output
        Returns:
            sequence_output: 融合后的bert encoder output. (bs, seq_len, hs)
        """
        if self.pre_model_type.lower() in ('electra', 'nezha'):
            fusion_idx = 0
        elif self.pre_model_type.lower() == 'roberta':
            fusion_idx = 2
        else:
            raise ValueError('Pre-train Model type must be NEZHA or ELECTRA or RoBERTa!')

        hidden_stack = torch.stack(outputs[fusion_idx][-self.fusion_layers:],
                                   dim=0)  # (num_layers, bs, seq_len, hs)
        sequence_output = torch.sum(hidden_stack * self.dym_weight,
                                    dim=0)  # (bs, seq_len, hs)
        return sequence_output

    def compute_loss(self, predictions, labels, target_mask):
        """
        计算loss
        Args:
            target_mask: 句子A部分和pad部分全为0，句子B部分为1. (bs, seq_len)
        """
        predictions = predictions.view(-1, self.vocab_size)
        labels = labels.view(-1)
        target_mask = target_mask.view(-1).float()
        loss_func = nn.CrossEntropyLoss(ignore_index=0, reduction="none")
        # average loss
        loss = (loss_func(predictions, labels) * target_mask).sum() / target_mask.sum()
        return loss

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            labels=None,
    ):
        """
        Args:
            input_ids: (batch_size, seq_len)
            attention_mask: 各元素的值为0或1，避免在padding的token上计算attention。(batch_size, seq_len)
            token_type_ids: 就是token对应的句子类型id，值为0或1。为空自动生成全0。(batch_size, seq_len)
            labels: (batch_size, seq_len)

        Returns:
            scores: (bs, seq_len, vocab_size)
        """
        bs, seq_len = input_ids.size()
        # 构造attention mask
        # TODO: 用这种mask会将[PAD]也编码进去
        sum_idxs = torch.cumsum(token_type_ids, dim=1)
        att_mask = (sum_idxs[:, None, :] <= sum_idxs[:, :, None]).float()
        # 将[PAD]部分的attention mask掉
        c_index = torch.argmax(token_type_ids, dim=1)
        tmp = token_type_ids.clone().detach()
        for r_i, c_i in enumerate(c_index):
            tmp[r_i, :c_i] = 1  # 句子A也标1
        tmp1 = tmp.unsqueeze(-1).repeat(1, 1, seq_len)  # (bs, seq_len, seq_len)
        tmp2 = tmp.unsqueeze(1).repeat(1, seq_len, 1)  # (bs, seq_len, seq_len)
        att_mask *= tmp1 * tmp2  # (bs, seq_len, seq_len)

        # pre-train model
        outputs = self.bert(
            input_ids,
            attention_mask=att_mask,
            token_type_ids=token_type_ids,
            output_hidden_states=True
        )  # sequence_output, pooled_output, (hidden_states), (attentions)

        # fusion BERT layers
        sequence_output = self.get_dym_layer(outputs)  # (batch_size, seq_len, hidden_size[embedding_dim])
        # decoder
        predictions = self.decoder(sequence_output)  # (bs, seq_len, vocab_size)

        if labels is not None:   ##训练集
            # 计算loss
            # 需要将句子A的loss mask掉
            predictions = predictions[:, :].contiguous()  # (bs, seq_len, vocab_size)
            # (bs, seq_len+1)
            token_type_ids = torch.cat([token_type_ids.float(), torch.ones(bs, 1, device=token_type_ids.device)], dim=1)
            loss_mask = token_type_ids[:, 1:].contiguous()
            loss = self.compute_loss(predictions, labels, loss_mask)
            return predictions, loss
        else:  #验证or测试集
            # 只取最后一个token的分数（自回归），每次只能预测最后一个token
            scores = torch.log_softmax(predictions[:, -1], dim=-1)  # (bs, vocab_size)
            return scores


if __name__ == '__main__':
    from transformers import BertConfig
    import utils
    import os

    params = utils.Params()
    # Prepare model
    bert_config = BertConfig.from_json_file(os.path.join(params.bert_model_dir, 'bert_config.json'))
    model = BertSeq2SeqModel.from_pretrained(config=bert_config,
                                             pretrained_model_name_or_path=params.bert_model_dir,
                                             params=params)
    # 保存bert config
    model.to(params.device)

    for n, _ in model.named_parameters():
        print(n)
