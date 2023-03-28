import time
import gensim
import torch
import numpy as np
import pandas as pd
import logging
from config import Config
from torch import nn

config = Config()

logging.basicConfig(level=logging.INFO)
from torch.utils.data import Dataset

embedding_dict = np.load('data/experimentData/VG/embedding_dict.npy', allow_pickle=True).item()
sentiScore_dict = np.load('data/experimentData/VG/sentiScore_dict.npy', allow_pickle=True).item()
# embedding_dict = np.load('data/experimentData/AIV/embedding_dict_d2v.npy', allow_pickle=True).item()
# embedding_dict = np.load('data/experimentData/AIV/embedding_dict_w2v.npy', allow_pickle=True).item()

def date(f='%Y-%m-%d %H:%M:%S'):
    return time.strftime(f, time.localtime())

class MyDataSet(Dataset):
    def __init__(self, data_path, config):
        df = pd.read_csv(data_path, header=None, names=['userID', 'itemID', 'review', 'rating'])
        rating = torch.Tensor(df['rating'].to_list()).view(-1, 1)
        self.config = config
        self.PAD_WORD_idx = config.PAD_WORD
        self.review_length = config.review_length
        self.review_count = config.review_count
        self.user_reviews, self.user_reviews_emb = self.get_reviews(df)  # 收集每个user的评论列表
        self.user_senti = self.get_senti(self.user_reviews)

        _, self.item_reviews_emb = self.get_reviews(df, 'itemID', 'userID')
        self.rating = rating[[idx for idx in range(rating.shape[0])]]

    def __getitem__(self, idx):
        return self.user_reviews_emb[idx], self.user_senti[idx], self.item_reviews_emb[idx], self.rating[idx]

    def __len__(self):
        return self.rating.shape[0]

    def get_reviews(self, df, lead='userID', costar='itemID'):
        # 对于每条训练数据，生成用户的所有评论汇总
        reviews_by_lead = dict(list(df[[costar, 'review']].groupby(df[lead])))  # 每个user/item评论汇总
        lead_reviews_ori = []
        lead_reviews = []
        lead_reviews_emb = []
        for lead_id in df[lead]:
            df_data = reviews_by_lead[lead_id]  # 取出lead的所有评论：DataFrame
            reviews = df_data['review'].to_list()  # 取lead所有评论：列表
            lead_reviews_ori.append(reviews)
            reviews = self.adjust_review_list(reviews, self.review_length, self.review_count)
            lead_reviews.append(reviews)
            reviews_emb = list(map(lambda x: embedding_dict[x], reviews))
            lead_reviews_emb.append(reviews_emb)
        return lead_reviews, torch.tensor(np.array(lead_reviews_emb), dtype=torch.float)

    def get_senti(self, user_reviews):
        result = []
        for user_doc in user_reviews:
            doc_result = []
            for review in user_doc:
                doc_result.append(sentiScore_dict[review])
            result.append(doc_result)
        return torch.tensor(result, dtype=torch.float)

    def adjust_review_list(self, reviews, r_length, r_count):
        reviews = reviews[:r_count] + [' '.join([self.PAD_WORD_idx] * r_length)] * (r_count - len(reviews))  # 评论数量固定
        reviews = [' '.join(r.split(' ')[:r_length] + [self.PAD_WORD_idx] * (r_length - len(r.split(' ')))) for r in reviews]  # 每条评论定长
        return reviews

class MyDataSet_w2v(Dataset):
    def __init__(self, data_path, word_emb, word_dict, config):
        df = pd.read_csv(data_path, header=None, names=['userID', 'itemID', 'review', 'rating'])
        rating = torch.Tensor(df['rating'].to_list()).view(-1, 1)
        self.config = config
        self.word_emb = word_emb
        self.word_dict = word_dict
        self.embedding = nn.Embedding.from_pretrained(self.word_emb)
        self.PAD_WORD_idx = self.word_dict[config.PAD_WORD]
        self.review_length = config.review_length
        self.review_count = config.review_count
        self.user_reviews, self.user_reviews_emb = self.get_reviews(df)  # 收集每个user的评论列表
        self.user_senti = self.get_senti(self.user_reviews)
        _, self.item_reviews_emb = self.get_reviews(df, 'itemID', 'userID')
        self.rating = rating[[idx for idx in range(rating.shape[0])]]

    def __getitem__(self, idx):
        return self.user_reviews_emb[idx], self.user_senti[idx], self.item_reviews_emb[idx], self.rating[idx]

    def __len__(self):
        return self.rating.shape[0]

    def get_reviews(self, df, lead='userID', costar='itemID'):
        # 对于每条训练数据，生成用户的所有评论汇总
        reviews_by_lead = dict(list(df[[costar, 'review']].groupby(df[lead])))  # 每个user/item评论汇总
        lead_reviews = []
        lead_reviews_emb = []
        for lead_id in df[lead]:
            df_data = reviews_by_lead[lead_id]  # 取出lead的所有评论：DataFrame
            reviews = df_data['review'].to_list()  # 取lead所有评论：列表
            reviews = self.adjust_review_list(reviews, self.review_length, self.review_count)
            reviews2id = self.review2id(reviews)
            reviews_emb = self.get_review_emb(reviews2id)
            lead_reviews_emb.append(reviews_emb)
            lead_reviews.append(reviews)
        return lead_reviews, torch.tensor(np.array(lead_reviews_emb), dtype=torch.float)

    def get_senti(self, user_reviews):
        result = []
        for doc in user_reviews:
            doc_result = []
            for sentence in doc:
                doc_result.append(sentiScore_dict[sentence])
            result.append(doc_result)
        return torch.tensor(result, dtype=torch.float)

    def adjust_review_list(self, reviews, r_length, r_count):
        reviews = reviews[:r_count] + [' '.join([config.PAD_WORD] * r_length)] * (r_count - len(reviews))  # 评论数量固定
        reviews = [' '.join(r.split(' ')[:r_length] + [config.PAD_WORD] * (r_length - len(r.split(' ')))) for r in reviews]  # 每条评论定长
        return reviews

    def review2id(self, reviews):  # 将一个评论字符串分词并转为数字
        result = []
        for review in reviews:
            if not isinstance(review, str):
                return []  # 貌似pandas的一个bug，读取出来的评论如果是空字符串，review类型会变成float
            wids = []
            for word in review.split():
                if word in self.word_dict:
                    wids.append(self.word_dict[word])  # 单词映射为数字
                else:
                    wids.append(self.PAD_WORD_idx)
            result.append(wids)
        return result

    def get_review_emb(self, reviews_idx):
        result = []
        for review_idx in reviews_idx:
            emb = self.embedding(torch.tensor(review_idx)).numpy().tolist()
            result.append(emb)
        result = torch.tensor(result, dtype=torch.float)
        result = torch.mean(result, dim=1)
        return result.numpy().tolist()

def load_embedding(word2vec_file, dataset):
    wvmodel = gensim.models.KeyedVectors.load_word2vec_format(word2vec_file,
                                                              binary=True)
    # 需要在字典的位置加上1是需要给UNK添加一个位置
    vocab_size = len(wvmodel.index_to_key) + 1
    vector_size = wvmodel.vector_size

    # 随机生成weight
    weight = torch.randn(vocab_size, vector_size)

    words = wvmodel.index_to_key

    word_to_idx = {word: i + 1 for i, word in enumerate(words)}
    # 定义了一个unknown的词.
    word_to_idx['[PAD]'] = 0
    idx_to_word = {i + 1: word for i, word in enumerate(words)}
    idx_to_word[0] = '[PAD]'

    for i in range(len(wvmodel.index_to_key)):
        try:
            index = word_to_idx[wvmodel.index_to_key[i]]
        except:
            continue
        vector = wvmodel.get_vector(idx_to_word[word_to_idx[wvmodel.index_to_key[i]]])
        weight[index, :] = torch.from_numpy(vector.copy())

    torch.save(weight, 'data/experimentData/' + dataset + '/embedding_w2v.pt')
    np.save('data/experimentData/' + dataset + '/word2id_w2v.npy', word_to_idx)
    return weight, word_to_idx

def predict_mae(model, dataloader, device):
    mae, sample_count = 0, 0
    with torch.no_grad():
        for batch in dataloader:
            user_reviews, user_senti, item_reviews, ratings = [x.to(device) for x in batch]
            # predict = model(user_reviews, item_reviews).to(config.device)
            predict, _ = model(user_reviews, user_senti, item_reviews)
            mae += torch.nn.functional.l1_loss(predict.to(device), ratings, reduction='sum').item()
            sample_count += len(ratings)
    return mae / sample_count  # dataloader上的平均绝对误差

def predict_rmse(model, dataloader, device):
    mse, sample_count = 0, 0
    with torch.no_grad():
        for batch in dataloader:
            user_reviews, user_senti, item_reviews, ratings = [x.to(device) for x in batch]
            # predict = model(user_reviews, item_reviews).to(config.device)
            predict, _ = model(user_reviews, user_senti, item_reviews)
            mse += torch.nn.functional.mse_loss(predict.to(device), ratings, reduction='sum').item()
            sample_count += len(ratings)
    return np.sqrt(mse / sample_count)  # dataloader上的均方误差
