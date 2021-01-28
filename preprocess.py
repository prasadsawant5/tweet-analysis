import re
import numpy as np
import os
import pandas as pd
from tqdm import tqdm
import json

# nltk
import nltk
from nltk.corpus import stopwords
from  nltk.stem import SnowballStemmer

# DATASET
DATA = 'data.csv'
CLEAN_DATA = 'deta-clean.csv'
DATASET_COLUMNS = ["target", "ids", "date", "flag", "user", "text"]
DATASET_ENCODING = "ISO-8859-1"
WORD_DICT = 'word_2_vec.json'

# TEXT CLENAING
TEXT_CLEANING_RE = "@\S+|https?:\S+|http?:\S|[^A-Za-z0-9]+"

stop_words = stopwords.words("english")
stemmer = SnowballStemmer("english")

decode_map = {0: "NEGATIVE", 2: "NEUTRAL", 4: "POSITIVE"}
def decode_sentiment(label):
    return decode_map[int(label)]

def preprocess(text, stem=False):
    # Remove link,user and special characters
    text = re.sub(TEXT_CLEANING_RE, ' ', str(text).lower()).strip()
    tokens = []
    for token in text.split():
        if token not in stop_words:
            if stem:
                tokens.append(stemmer.stem(token))
            else:
                tokens.append(token)
    return " ".join(tokens)

def one_hot(label):
    if label == 'POSITIVE':
        return [0, 0, 1]
    elif label == 'NEGATIVE':
        return [1, 0, 0]
    else:
        return [0, 1, 0]

if __name__ == '__main__':
    nltk.download('stopwords')

    # Add text based labels and clean the tweets for any unwanted strings
    df = pd.read_csv(DATA, encoding =DATASET_ENCODING , names=DATASET_COLUMNS)
    df.target = df.target.apply(lambda x: decode_sentiment(x))
    df.text = df.text.apply(lambda x: preprocess(x))

    # loop over all the words in the tweet and create a set of words
    words = set()
    tweet_len = set()
    for index, row in tqdm(df.iterrows(), total=df.shape[0]):
        txt = row['text']
        tweet_len.add(len(txt.split(' ')))
        
        for word in txt.split(' '):
            words.add(word)

    # create a word to vector representation dictionary and save it to word_2_vec.json
    word_2_vec = {}
    if not os.path.exists(WORD_DICT):
        word_2_vec['<PAD>'] = 0     # pad vector (to be used for padding all the tweets which are less than max length)
        counter = 1
        for word in words:
            word_2_vec[word] = counter
            counter += 1

        with open(WORD_DICT, 'w+') as f:
            json.dump(word_2_vec, f)
    else:
        with open(WORD_DICT, 'r') as f:
            word_2_vec = json.load(f)


    vectorized_data = []
    max_len_tweet = max(tweet_len)
    print('Max tweet length: {}'.format(max_len_tweet))
    one_hot_labels = []

    # one hot the labels and transform the fetures into a numpy array
    for index, row in tqdm(df.iterrows(), total=df.shape[0]):
        txt = row['text']
        label = one_hot(row['target'])
        one_hot_labels.append(label)

        vector = np.zeros(max_len_tweet)
        i = 0
        for word in txt.split(' '):
            vec = word_2_vec[word]
            vector[i] = vec
            i += 1


        vectorized_data.append(np.array(vector))

    # Save an numpy array
    np.save('vectorized_data.npy', np.array(vectorized_data))
    np.save('labels.npy', np.array(one_hot_labels))
    




