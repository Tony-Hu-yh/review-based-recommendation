import torch
from torch import nn
from torch.nn import functional as F
from config import Config

config = Config()

import numpy as np

class AttentionMechanism(nn.Module):
    def __init__(self, config):
        super(AttentionMechanism, self).__init__()
        self.linear1 = nn.Sequential(
            nn.Linear(50, 25),
            nn.ReLU(),
            nn.Dropout(p=config.dropout_prob)
        )

        self.linear2 = nn.Linear(25, 1)

    def forward(self, x):
        w = self.linear2(self.linear1(x))
        a = F.softmax(w, 1)

        return a

class BiGRU(nn.Module):

    def __init__(self, config, word_dim):
        super(BiGRU, self).__init__()
        self.device = config.device
        self.hidden_dim = config.hidden_dim
        self.layer_dim = config.layer_dim
        self.bigru = nn.GRU(word_dim, config.hidden_dim, config.layer_dim, batch_first=True, bidirectional=True)
        self.gru_bn = nn.BatchNorm1d(config.hidden_dim * 2, momentum=config.momentum)
        self.fc = nn.Sequential(
            nn.Linear(config.hidden_dim * 2, config.output_dim),
            nn.ReLU()
        )

    def forward(self, x):
        # 初始化隐藏层状态全为0
        # （layer_dim, batch_size, hidden_dim）
        h0 = torch.zeros(self.layer_dim, x.shape[0], self.hidden_dim).requires_grad_().to(self.device)
        #  初始化cell state
        c0 = torch.zeros(self.layer_dim, x.shape[0], self.hidden_dim).requires_grad_().to(self.device)
        #  分离隐藏状态，以免梯度爆炸
        out, hn = self.bigru(x)
        #  只需要最后一层隐藏层的状态
        out = self.fc(out)
        return out

class DeepFM(nn.Module):

    def __init__(self, p, k):
        super(DeepFM, self).__init__()
        self.v = nn.Parameter(torch.zeros(p, k))
        self.linear = nn.Linear(p, 1, bias=True)
        self.mlp = nn.Sequential(
            nn.Linear(p, p // 2),
            nn.BatchNorm1d(p // 2, momentum=config.momentum),
            nn.ReLU(),

            nn.Linear(p // 2, 1),
            nn.BatchNorm1d(1, momentum=config.momentum),
            nn.ReLU(),
            nn.Dropout(config.dropout_prob),
        )

    def forward(self, x):
        #  FM_part
        linear_part = self.linear(x)  # input shape(batch_size, out_dim), out shape(batch_size, 1)
        inter_part1 = torch.mm(x, self.v)
        inter_part2 = torch.mm(x ** 2, self.v ** 2)
        pair_interactions = torch.sum(inter_part1 ** 2 - inter_part2, dim=1)
        fm_out = 0.5 * pair_interactions

        #  Deep_part
        deep_out = self.mlp(x).transpose(1, 0)

        deepfm_out = linear_part.transpose(1, 0) + fm_out + deep_out
        return deepfm_out.view(-1, 1)  # out shape(batch_size, 1)

class MyModel(nn.Module):
    def __init__(self, config):
        super(MyModel, self).__init__()
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)
        self.attentionMechanism = AttentionMechanism(config)
        self.fully_connect = nn.Linear(config.output_dim, config.output_dim, bias=True)
        self.bigru = nn.Sequential(
            BiGRU(config, word_dim=config.embedding_size),
            nn.Dropout(p=config.dropout_prob)
        )
        self.deepfm = DeepFM(config.output_dim * 2, 10)

    def forward(self, user_review, user_senti, item_review):
        user_embedding = user_review
        item_embedding = item_review

        # senti_weight = self.sigmoid(user_senti).reshape(user_embedding.shape[0], -1, 1)
        senti_weight = self.softmax(user_senti).reshape(user_embedding.shape[0], -1, 1)
        # senti_weight = self.softmax(torch.abs(user_senti)).reshape(user_embedding.shape[0], -1, 1)
        # senti_weight = user_senti.reshape(user_embedding.shape[0], -1, 1)

        user_feature = self.bigru(user_embedding)
        item_feature = self.bigru(item_embedding)

        a_item = self.attentionMechanism(item_feature)
        a_user = self.attentionMechanism(user_feature)

        user_attentive_feature = torch.sum(user_feature.mul(senti_weight), dim=1)
        # user_attentive_feature = torch.sum(user_feature, dim=1)
        # user_attentive_feature = torch.sum(self.fully_connect(user_feature.mul(a_user)), dim=1)
        item_attentive_feature = torch.sum(self.fully_connect(item_feature.mul(a_item)), dim=1)

        concat_feature = torch.cat((user_attentive_feature, item_attentive_feature), dim=1)
        prediction = self.deepfm(concat_feature)
        # prediction = torch.sum(user_attentive_feature.mul(item_attentive_feature), dim=1).reshape(-1,1)

        return prediction, senti_weight

