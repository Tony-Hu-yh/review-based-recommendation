import argparse
import inspect

import torch

class Config:
    assignment = 'train'
    device = torch.device("cuda:0")
    embedding_size = 768 # TODO: 记得修改
    train_epochs = 1000
    batch_size = 300
    learning_rate = 0.001
    l2_regularization = 1e-1  # 权重衰减程度
    learning_rate_decay = 0.99  # 学习率衰减程度
    momentum = 0.9

    word2vec_file = '../DeepCoNN-pytorch-master/embedding/GoogleNews-vectors-negative300.bin'
    dataset = 'VG'
    train_file = 'data/experimentData/VG/train.csv'
    valid_file = 'data/experimentData/VG/valid.csv'
    test_file = 'data/experimentData/VG/test.csv'
    model_file = 'model/best_model_VG_sensitivity.pt'
    # model_file = 'model/best_model_MI_W2V.pt'

    review_count = 10  # max review count
    review_length = 40  # max review length
    lowest_review_count = 3  # reviews wrote by a user/item will be delete if its amount less than such value.
    PAD_WORD = '[PAD]'

    #  BiGRU
    hidden_dim = 128
    layer_dim = 2
    output_dim = 50

    dropout_prob = 0.5

    patience = 5

    def __init__(self):
        attributes = inspect.getmembers(self, lambda a: not inspect.isfunction(a))
        attributes = list(filter(lambda x: not x[0].startswith('__'), attributes))

        parser = argparse.ArgumentParser()
        for key, val in attributes:
            parser.add_argument('--' + key, dest=key, type=type(val), default=val)
        for key, val in parser.parse_args().__dict__.items():
            self.__setattr__(key, val)

    def __str__(self):
        attributes = inspect.getmembers(self, lambda a: not inspect.isfunction(a))
        attributes = list(filter(lambda x: not x[0].startswith('__'), attributes))
        to_str = ''
        for key, val in attributes:
            to_str += '{} = {}\n'.format(key, val)
        return to_str