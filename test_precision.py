import pandas as pd
import torch
from config import Config
config = Config()

def get_precision(test_data, model, k):
    def adjust_review_list(reviews, r_length, r_count):
        reviews = reviews[:r_count] + [' '.join([config.PAD_WORD] * r_length)] * (r_count - len(reviews))  # 评论数量固定
        reviews = [' '.join(r.split(' ')[:r_length] + [config.PAD_WORD] * (r_length - len(r.split(' ')))) for r in reviews]  # 每条评论定长
        return reviews

    def get_reviews(df, lead, costar):
        # 对于每条训练数据，生成用户的所有评论汇总
        reviews_by_lead = dict(list(df[[costar, 'review']].groupby(df[lead])))  # 每个user/item评论汇总
        lead_reviews = {}
        for lead_id in set(df[lead]):
            df_data = reviews_by_lead[lead_id]  # 取出lead的所有评论：DataFrame
            reviews = df_data['review'].to_list()  # 取lead所有评论：列表
            reviews = adjust_review_list(reviews, config.review_length, config.review_count)
            lead_reviews.setdefault(lead_id, reviews)
        return lead_reviews

    user_reviews = get_reviews(test_data, 'userID', 'itemID')
    item_reviews = get_reviews(test_data, 'itemID', 'userID')

    precision = 0
    for user in set(test_data['userID']):
        user_batch = []
        user_batch.append(user_reviews[user])
        precision_u = 0
        real_set = set(test_data[test_data['userID'] == user]['itemID'])
        recommend = {}
        for item in set(test_data['itemID']):
            item_batch = []
            item_batch.append(item_reviews[item])
            prediction = model(user_batch, item_batch)
            prediction = prediction[0][0].item()
            if prediction > 3:
                recommend.setdefault(item, prediction)
        recommended = set(list(dict(sorted(recommend.items(), key=lambda x: x[1], reverse=True)).keys())[:k])
        precision_u += len(real_set & recommended) / k
        precision += precision_u

    return precision / len(set(test_data['userID']))

if __name__ == '__main__':
    test_data = pd.read_csv(config.test_file, header=None).rename(columns={0:'userID', 1:'itemID', 2:'review', 3:'rating'})
    model = torch.load(config.model_file)
    k = 10
    precision = get_precision(test_data, model, k)
    print(precision)
