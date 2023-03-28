import os
import time
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from config import Config
from utils import *
from model import MyModel
from pytorchtools import EarlyStopping

def train(train_dataloader, valid_dataloader, model, config, model_path):
    early_stopping = EarlyStopping(path=config.model_file, patience=config.patience, verbose=True, delta=0, trace_func=print)
    print(f'{date()}## Start the training!')
    train_rmse = predict_rmse(model, train_dataloader, config.device)
    valid_rmse = predict_rmse(model, valid_dataloader, config.device)
    print(f'{date()}#### Initial train mse {train_rmse:.6f}, validation mse {valid_rmse:.6f}')
    start_time = time.perf_counter()
    opt = torch.optim.Adam(model.parameters(), config.learning_rate, weight_decay=config.l2_regularization)
    lr_sch = torch.optim.lr_scheduler.ExponentialLR(opt, config.learning_rate_decay)

    best_loss, best_epoch = 100, 0
    for epoch in range(config.train_epochs):
        model.train()  # 将模型设置为训练状态
        total_loss, total_samples = 0, 0
        for batch in train_dataloader:
            user_reviews, user_senti, item_reviews, ratings = [x.to(config.device) for x in batch]
            # predict = model(user_reviews, item_reviews).to(config.device)
            predict, _ = model(user_reviews, user_senti, item_reviews)
            loss = F.mse_loss(predict.to(config.device), ratings, reduction='sum')  # 平方和误差
            opt.zero_grad()  # 梯度清零
            loss.backward()  # 反向传播计算梯度
            opt.step()  # 根据梯度信息更新所有可训练参数

            total_loss += loss.item()
            total_samples += len(predict)

        lr_sch.step()
        model.eval()  # 停止训练状态
        valid_rmse = predict_rmse(model, valid_dataloader, config.device)
        valid_mae = predict_mae(model, valid_dataloader, config.device)
        train_loss = total_loss / total_samples
        print(f"{date()}#### Epoch {epoch:3d}; train loss {train_loss:.6f}; validation rmse {valid_rmse:.6f}, validation mae {valid_mae:.6f}")

        early_stopping(valid_rmse, model)
        if early_stopping.early_stop:
            print('Early Stopping')
            break

    end_time = time.perf_counter()
    print(f'{date()}## End of training! Time used {end_time - start_time:.0f} seconds.')

def test(dataloader, model):
    print(f'{date()}## Start the testing!')
    start_time = time.perf_counter()
    test_rmse = predict_rmse(model, dataloader, config.device)
    test_mae = predict_mae(model, dataloader, config.device)
    end_time = time.perf_counter()
    print(f"{date()}## Test end, test rmse is {test_rmse:.6f}, test mae is {test_mae:.6f}, time used {end_time - start_time:.0f} seconds.")

if __name__ == '__main__':
    config = Config()
    print(config)

    train_dataset = MyDataSet(config.train_file, config)
    valid_dataset = MyDataSet(config.valid_file, config)
    test_dataset = MyDataSet(config.test_file, config)
    train_dlr = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    valid_dlr = DataLoader(valid_dataset, batch_size=config.batch_size)
    test_dlr = DataLoader(test_dataset, batch_size=config.batch_size)

    '''dataset = config.dataset
    # word_emb, word_dict = load_embedding(word2vec_file=config.word2vec_file, dataset=config.dataset)
    word_emb = torch.load('data/experimentData/' + dataset + '/embedding_w2v.pt')
    word_dict = np.load('data/experimentData/' + dataset + '/word2id_w2v.npy', allow_pickle=True).item()
    train_dataset = MyDataSet_w2v(config.train_file, word_emb, word_dict, config)
    valid_dataset = MyDataSet_w2v(config.valid_file, word_emb, word_dict, config)
    test_dataset = MyDataSet_w2v(config.test_file, word_emb, word_dict, config)
    train_dlr = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    valid_dlr = DataLoader(valid_dataset, batch_size=config.batch_size)
    test_dlr = DataLoader(test_dataset, batch_size=config.batch_size)'''

    model = MyModel(config).to(config.device)

    os.makedirs(os.path.dirname(config.model_file), exist_ok=True)  # 文件夹不存在则创建
    train(train_dlr, valid_dlr, model, config, config.model_file)
    test(test_dlr, torch.load(config.model_file))
    del train_dataset, valid_dataset, test_dataset