# /usr/bin/env python
# coding=utf-8
# @Time : 2021/3/18
# @Author : Chile
# @Email : realchilewang@foxmail.com
# @File : utils.py
# @Software: PyCharm
import torch


class Config(object):
    def __init__(self):
        self.base_dir = '/data1/home/fzq/projects/text_generation/'  # 代码存放基础路径
        # 存模型和读数据参数
        self.source_train_dir = '/data1/home/fzq/projects/text_generation/data/'  # 原始数据集
        self.source_test_dir = '/data1/home/fzq/projects/text_generation/data/'  # 原始数据集
        self.processed_data = '/data1/home/fzq/projects/text_generation/data/' # 处理后的数据路径

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.warmup_proportion = 0.05
        self.use_bert = True
        self.pretrainning_model = 'nezha'
        self.pre_model_type = self.pretrainning_model
        self.relation_num = 2
        self.over_sample = True  ##表示的含义

        self.decay_rate = 0.5
        self.decay_step = 5000
        self.num_checkpoints = 5

        self.train_epoch =20
        self.sequence_length = 256

        self.learning_rate = 2e-4
        self.embed_learning_rate = 4e-5
        self.batch_size =16 #24

        self.embed_trainable = True

        self.as_encoder = True

        # if self.pretrainning_model == 'nezha':
        #     model = '/home/wangzhili/pretrained_model/Torch_model/pre_model_nezha_base/'
        # elif self.pretrainning_model == 'roberta':
        #     model = '/home/wangzhili/pretrained_model/Torch_model/pre_model_roberta_base/'
        # else:
        #     model = '/home/wangzhili/pretrained_model/Torch_model/pre_model_electra_base/'
        if self.pretrainning_model == 'roberta':
            model = '/data1/home/fzq/projects/nlpclassification/data/model/RoBERTa_zh_L12_PyTorch/'  # 中文roberta-base
        elif self.pretrainning_model == 'nezha':
            model = '/data1/home/fzq/projects/nlpclassification/data/model/nezha-cn-base/nezha-cn-base/'  # 中文nezha-base
        else:
            raise KeyError('albert nezha roberta bert bert_wwm is need')
        self.model_path = model
        # self.bert_config_file = model + 'bert_config.json'
        # self.bert_file = model + 'pytorch_model.bin'
        self.continue_training = False
        self.bert_file = model + 'pytorch_model.bin'
        self.bert_config_file = model + 'config.json'
        self.vocab_file = model + 'vocab.txt'

        # 解码参数
        self.train_batch_size = 24
        self.val_batch_size = 1
        self.test_batch_size = 1
        self.beam_size = 5   #每一步选取top5 的概率
        self.tgt_seq_len = 30

        self.fusion_layers=4

        self.save_model = self.base_dir + 'Savemodel/'
        self.fold = 1
        self.compare_result = False
        self.kfoldpath = 'Kfold_data/'
        self.result_file = 'result/'
        self.drop_prob = 0.1  # drop_out率
        # 卷积参数
        # self.gru_hidden_dim = 64
        self.rnn_num = 256
        self.dropout = 0.9
        # self.loss_name = 'focal_loss'
        self.loss_name = 'normal'
        self.early_stop = 100
        self.adv = ''
        self.flooding = 0.2
        self.embed_name = 'bert.embeddings.word_embeddings.weight'  # 词
        self.restore_file = None
        self.gradient_accumulation_steps = 1

        self.checkpoint_path = '/home/wangzhili/lei/Ner4torch/Savemodel/runs_0/1608629862/model_0.7402_782'
        """
        实验记录
        """
