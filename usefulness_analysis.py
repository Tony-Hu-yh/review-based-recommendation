import torch
import random
import pickle
import pandas as pd
import numpy as np
from config import Config

dataset = 'AIV'
json_path = 'data/' + 'Amazon_Instant_Video_5.json'
embedding_dict = np.load('data/experimentData/' + dataset + '/embedding_dict.npy', allow_pickle=True).item()
sentiScore_dict = np.load('data/experimentData/' + dataset + '/sentiScore_dict.npy', allow_pickle=True).item()

config = Config()
PAD_WORD = config.PAD_WORD

def adjust_review_list(reviews, r_length, r_count):
    reviews = reviews[:r_count] + [' '.join([PAD_WORD] * r_length)] * (r_count - len(reviews))  # 评论数量固定
    reviews = [' '.join(r.split(' ')[:r_length] + [PAD_WORD] * (r_length - len(r.split(' ')))) for r in
               reviews]  # 每条评论定长
    return reviews

def get_reviewsANDembeddings(df, lead='userID', costar='itemID'):
    # 对于每条训练数据，生成用户的所有评论汇总
    reviews_by_lead = dict(list(df[[costar, 'review']].groupby(df[lead])))  # 每个user/item评论汇总
    lead_reviews = []
    lead_reviews_ori = []
    lead_reviews_emb = []
    for lead_id in df[lead]:
        df_data = reviews_by_lead[lead_id]  # 取出lead的所有评论：DataFrame
        reviews = df_data['review'].to_list()  # 取lead所有评论：列表
        lead_reviews_ori.append(reviews)
        reviews = adjust_review_list(reviews, config.review_length, config.review_count)
        lead_reviews.append(reviews)
        reviews_emb = list(map(lambda x: embedding_dict[x], reviews))
        lead_reviews_emb.append(reviews_emb)
    return lead_reviews_ori, lead_reviews, torch.tensor(np.array(lead_reviews_emb), dtype=torch.float)

def get_senti(user_reviews):
    result = []
    for user_doc in user_reviews:
        doc_result = []
        for review in user_doc:
            doc_result.append(sentiScore_dict[review])
        result.append(doc_result)
    return torch.tensor(result, dtype=torch.float)

def process_dataset(json_path, select_cols):
    print('#### Read the json file...')
    if json_path.endswith('gz'):
        df = pd.read_json(json_path, lines=True, compression='gzip')
    else:
        df = pd.read_json(json_path, lines=True)
    df = df[select_cols]
    df.columns = ['userID', 'itemID', 'review', 'rating', 'time', 'useful']  # Rename above columns for convenience
    # map user(or item) to number
    df['userID'] = df.groupby(df['userID']).ngroup()
    df['itemID'] = df.groupby(df['itemID']).ngroup()

    def clean_review(review):  # clean a review using stop words
        review = review.lower().strip().replace('  ', ' ')
        return review

    def trans_useful(useful):
        if useful[1] == 0:
            return 0
        elif useful[0] < useful[1] / 2:
            return 0
        else:
            return 1

        '''if useful[0] != 0:
            return 1
        else:
            return 0'''

    df = df.drop(df[[not isinstance(x, str) or len(x) == 0 for x in df['review']]].index)  # erase null reviews
    df['review'] = df['review'].apply(clean_review)

    df['useful'] = df['useful'].apply(trans_useful)
    df = df[['userID', 'itemID', 'review', 'rating', 'time', 'useful']]
    print(f'#### Total: {len(df)} reviews, {len(df.groupby("userID"))} users, {len(df.groupby("itemID"))} items.')
    return df

def cal_precision_k(data, k):
    data_by_item = dict(list(data[['review', 'useful', 'attention', 'time']].groupby(data['itemID'])))
    pre_top1_attn = []
    pre_top1_random = []
    pre_top1_length = []
    pre_top1_time = []

    pre_topk_attn = []
    pre_topk_random = []
    pre_topk_length = []
    pre_topk_time = []

    for item in set(data['itemID']):
        item_data = data_by_item[item]
        attention = list(item_data['attention'])[0]
        item_data['index'] = list(range(len(item_data)))
        item_data['review_length'] = item_data['review'].apply(lambda x: len(x.split()))
        index_useful_dict = dict(zip(item_data['index'], item_data['useful']))
        attn_index_dict = dict(zip(attention, list(range(len(attention)))))
        time_index_dict = dict(zip(list(item_data['time']), list(range(len(list(item_data['time']))))))
        length_index_dict = dict(zip(list(item_data['review_length']), list(range(len(list(item_data['review_length']))))))
        useful_reviews = list(item_data[item_data['useful'] == 1]['index'])

        if k == 1:
            attn_top1 = attention.index(max(attention))
            random_top1 = random.choice(list(item_data['index']))
            length_top1 = list(item_data['review_length']).index(max(item_data['review_length']))
            time_top1 = list(item_data['time']).index(max(item_data['time']))

            try:
                pre_top1_attn.append(index_useful_dict[attn_top1])
            except KeyError:
                pre_top1_attn.append(0)
            pre_top1_random.append(index_useful_dict[random_top1])
            pre_top1_length.append(index_useful_dict[length_top1])
            pre_top1_time.append(index_useful_dict[time_top1])
        else:
            if len(item_data) < k:
                continue
            else:
                attn_sorted_list = list(sorted(attention, reverse=True))
                length_sorted_list = list(sorted(list(item_data['review_length']), reverse=True))
                time_sorted_list = list(sorted(list(item_data['time']), reverse=True))

                attn_topk = list(map(lambda x: attn_index_dict[x], attn_sorted_list[:k]))
                random_topk = random.sample(list(item_data['index']), k)
                length_topk = list(map(lambda x: length_index_dict[x], length_sorted_list[:k]))
                time_topk = list(map(lambda x: time_index_dict[x], time_sorted_list[:k]))

                pre_topk_attn.append(len(set(attn_topk) & set(useful_reviews)) / k)
                pre_topk_random.append(len(set(random_topk) & set(useful_reviews)) / k)
                pre_topk_length.append(len(set(length_topk) & set(useful_reviews)) / k)
                pre_topk_time.append(len(set(time_topk) & set(useful_reviews)) / k)
    if k == 1:
        print('myModel-precision@1={}'.format(np.average(pre_top1_attn)))
        print('random-precision@1={}'.format(np.average(pre_top1_random)))
        print('longest-precision@1={}'.format(np.average(pre_top1_length)))
        print('latest-precision@1={}'.format(np.average(pre_top1_time)))
    else:
        print('myModel-precision@{}={}'.format(k, np.average(pre_topk_attn)))
        print('random-precision@{}={}'.format(k, np.average(pre_topk_random)))
        print('longest-precision@{}={}'.format(k, np.average(pre_topk_length)))
        print('latest-precision@{}={}'.format(k, np.average(pre_topk_time)))

if __name__ == '__main__':
    '''data = process_dataset(json_path, select_cols=['reviewerID', 'asin', 'reviewText', 'overall', 'unixReviewTime', 'helpful'])
    model = torch.load('model/best_model_{}.pt'.format(dataset))
    _, user_reviews, user_embedding = get_reviewsANDembeddings(data)
    user_senti = get_senti(user_reviews)
    item_reviews_ori, _, item_embedding = get_reviewsANDembeddings(data, lead='itemID', costar='userID')
    _, a_item = model(user_embedding.to(config.device), user_senti.to(config.device), item_embedding.to(config.device))
    a_item = a_item.reshape(-1, 10).cpu().detach().numpy().tolist()
    data['attention'] = a_item
    data.to_pickle('data/experimentData/' + dataset + '/data_usefulness.pickle')'''

    data = pd.read_pickle('data/experimentData/' + dataset + '/data_usefulness.pickle')

    k=10
    cal_precision_k(data, k)