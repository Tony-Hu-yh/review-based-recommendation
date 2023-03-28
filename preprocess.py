import torch
import argparse
import os
os.environ['NUMEXPR_MAX_THREADS'] = '16'
import sys
import time
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk import word_tokenize #分词函数
from nltk.corpus import sentiwordnet as swn #得到单词情感得分
from sklearn.model_selection import train_test_split
os.chdir(sys.path[0])

# OPTIONAL: if you want to have more information on what's happening, activate the logger as follows
import logging
logging.basicConfig(level=logging.INFO)
from pytorch_pretrained_bert import convert_tf_checkpoint_to_pytorch
from pytorch_pretrained_bert import BertTokenizer, BertModel
from pytorch_pretrained_bert import BertConfig

# Translate model from tensorflow to pytorch
BERT_MODEL_PATH = '../../bert/uncased_L-12_H-768_A-12/'
convert_tf_checkpoint_to_pytorch.convert_tf_checkpoint_to_pytorch(
    BERT_MODEL_PATH + 'bert_model.ckpt',
BERT_MODEL_PATH + 'bert_config.json',
BERT_MODEL_PATH + 'pytorch_model.bin')

# Load pre-trained model tokenizer (vocabulary)
tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_PATH)

# Load pre-trained model (weights)
bert_model = BertModel.from_pretrained(BERT_MODEL_PATH)

bert_config = BertConfig('../../bert/uncased_L-12_H-768_A-12/'+'bert_config.json')

review_count = 10  # max review count
review_length = 40  # max review length
PAD_WORD = '[PAD]'

def getBertVector(text):
    marked_text = "[CLS] " + text + " [SEP]"

    # Tokenize our sentence with the BERT tokenizer.
    tokenized_text = tokenizer.tokenize(marked_text)

    # Map the token strings to their vocabulary indeces.
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)

    # Mark each of the 22 tokens as belonging to sentence "1".
    segments_ids = [1] * len(tokenized_text)

    # Convert inputs to PyTorch tensors
    tokens_tensor = torch.tensor([indexed_tokens])
    segments_tensors = torch.tensor([segments_ids])

    # Put the model in "evaluation" mode, meaning feed-forward operation.
    bert_model.eval()

    # Run the text through BERT, and collect all of the hidden states produced
    # from all 12 layers.
    with torch.no_grad():
        outputs = bert_model(tokens_tensor, segments_tensors)

        # Evaluating the model will return a different number of objects based on
        # how it's  configured in the `from_pretrained` call earlier. In this case,
        # because we set `output_hidden_states = True`, the third item will be the
        # hidden states from all layers. See the documentation for more details:
        # https://huggingface.co/transformers/model_doc/bert.html#bertmodel
        hidden_states = outputs[0]  # 12*1*22*768

    '''get sentence_vec'''
    # `hidden_states` has shape [12 x 1 x 22 x 768]

    # `token_vecs` is a tensor with shape [22 x 768]
    token_vecs = hidden_states[-2][0]

    # Calculate the average of all 22 token vectors.
    sentence_embedding = torch.mean(token_vecs, dim=0)
    sentence_embedding = sentence_embedding.numpy().tolist()

    return sentence_embedding

def adjust_review_list(review):
    # reviews = reviews[:review_count] + [' '.join([PAD_WORD] * review_length)] * (review_count - len(reviews))  # 评论数量固定
    reviews = ' '.join(review.split(' ')[:review_length] + [PAD_WORD] * (review_length - len(review.split(' ')))) # 每条评论定长
    return reviews

def embedding(review):
    review = adjust_review_list(review)
    vector = getBertVector(review)
    return vector

def sentiwordnet_analysis(sentence):
    # create单词表
    # nltk.pos_tag是打标签
    ttt = nltk.pos_tag([i for i in word_tokenize(str(sentence).lower())])
    word_tag_fq = nltk.FreqDist(ttt)
    wordlist = word_tag_fq.most_common()

    # 变为dataframe形式
    key = []
    part = []
    frequency = []
    for i in range(len(wordlist)):
        key.append(wordlist[i][0][0])
        part.append(wordlist[i][0][1])
        frequency.append(wordlist[i][1])
    textdf = pd.DataFrame({'key': key,
                           'part': part,
                           'frequency': frequency},
                          columns=['key', 'part', 'frequency'])

    # 编码
    n = ['NN', 'NNP', 'NNPS', 'NNS', 'UH']
    v = ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']
    a = ['JJ', 'JJR', 'JJS']
    r = ['RB', 'RBR', 'RBS', 'RP', 'WRB']

    for i in range(len(textdf['key'])):
        z = textdf.iloc[i, 1]

        if z in n:
            textdf.iloc[i, 1] = 'n'
        elif z in v:
            textdf.iloc[i, 1] = 'v'
        elif z in a:
            textdf.iloc[i, 1] = 'a'
        elif z in r:
            textdf.iloc[i, 1] = 'r'
        else:
            textdf.iloc[i, 1] = ''

        # 计算单个评论的单词分数
    score = []
    for i in range(len(textdf['key'])):
        m = list(swn.senti_synsets(textdf.iloc[i, 0], textdf.iloc[i, 1]))
        s = 0
        ra = 0
        if len(m) > 0:
            for j in range(len(m)):
                s += (m[j].pos_score() - m[j].neg_score()) / (j + 1)
                ra += 1 / (j + 1)
            score.append(s / ra)
        else:
            score.append(0)
    return sum(score)

def process_dataset(json_path, select_cols, train_rate, csv_path):
    print('#### Read the json file...')
    if json_path.endswith('gz'):
        df = pd.read_json(json_path, lines=True, compression='gzip')
    else:
        df = pd.read_json(json_path, lines=True)
    df = df[select_cols]
    df.columns = ['userID', 'itemID', 'review', 'rating']  # Rename above columns for convenience
    # map user(or item) to number
    df['userID'] = df.groupby(df['userID']).ngroup()
    df['itemID'] = df.groupby(df['itemID']).ngroup()
    # stop_words = stopwords.words('english')
    def clean_review(review):  # clean a review using stop words
        review = review.lower().strip().replace('  ', ' ')
        '''tokens = word_tokenize(review)
        print(tokens)
        result = []
        for token in tokens:
            if token in stop_words:
                continue
            else:
                result.append(token)
        result = ' '.join(result)'''
        return review

    df = df.drop(df[[not isinstance(x, str) or len(x) == 0 for x in df['review']]].index)  # erase null reviews
    df['review'] = df['review'].apply(clean_review)
    PAD_sentence = ' '.join([PAD_WORD] * review_length)

    df['embedding'] = df['review'].apply(embedding)
    review_embedding_dict = dict(zip(list(df['review'].apply(adjust_review_list)), list(df['embedding'])))

    review_embedding_dict[PAD_sentence] = embedding(PAD_sentence)
    np.save(os.path.join(csv_path, 'embedding_dict.npy'), review_embedding_dict)

    df['senti_score'] = df['review'].apply(sentiwordnet_analysis)
    review_sentiScore_dict = dict(zip(list(df['review'].apply(adjust_review_list)), list(df['senti_score'])))
    review_sentiScore_dict[PAD_sentence] = sentiwordnet_analysis(PAD_sentence)
    np.save(os.path.join(csv_path, 'sentiScore_dict.npy'), review_sentiScore_dict)

    df = df[['userID', 'itemID', 'review', 'rating']]

    train, valid = train_test_split(df, test_size=1 - train_rate, random_state=3)  # split dataset including random
    valid, test = train_test_split(valid, test_size=0.5, random_state=4)
    os.makedirs(csv_path, exist_ok=True)
    train.to_csv(os.path.join(csv_path, 'train.csv'), index=False, header=False)
    valid.to_csv(os.path.join(csv_path, 'valid.csv'), index=False, header=False)
    test.to_csv(os.path.join(csv_path, 'test.csv'), index=False, header=False)
    print(f'#### Split and saved dataset as csv: train {len(train)}, valid {len(valid)}, test {len(test)}')
    print(f'#### Total: {len(df)} reviews, {len(df.groupby("userID"))} users, {len(df.groupby("itemID"))} items.')
    return train, valid, test


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', dest='data_path',
                        default='Video_Games_5.json',
                        help='Selected columns of above dataset in json format.')
    parser.add_argument('--select_cols', dest='select_cols', nargs='+',
                        default=['reviewerID', 'asin', 'reviewText', 'overall'])
    parser.add_argument('--train_rate', dest='train_rate', default=0.8)
    parser.add_argument('--save_dir', dest='save_dir', default='./experimentData/VG')
    args = parser.parse_args()

    start_time = time.perf_counter()
    process_dataset(args.data_path, args.select_cols, args.train_rate, args.save_dir)
    end_time = time.perf_counter()
    print(f'## preprocess.py: Data loading complete! Time used {end_time - start_time:.0f} seconds.')