#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import tensorflow as tf
import ktrain 
from ktrain import text
import json, csv, nltk

def load_data():
    raw_data = []
    for line in open('./dataset/tweets_DM.json', 'r'):
         raw_data.append(json.loads(line))

    with open('./dataset/emotion.csv') as f:
        data = csv.reader(f, delimiter=',')
         # use dictionary to store the {id:emotion} so that I can build the DataFrame later easily
        id_emotion_dict = {row[0]:row[1] for row in data}  

    with open('./dataset/data_identification.csv') as f:
        data = csv.reader(f, delimiter=',')
        ident_dict = {row[0]:row[1] for row in data}
    return raw_data, id_emotion_dict, ident_dict

def train_data(data, id_emotion, ident):
    # build DataFrame, including '_score', '_source', '_index', '_crawldate', '_type' columns
    X = pd.DataFrame(data)
    X['id'] = [row['tweet']['tweet_id'] for row in X['_source']]
    X['text'] = [row['tweet']['text'] for row in X['_source']]
    X.drop(['_score', '_type', '_index', '_source', '_crawldate'], axis=1, inplace=True)
    X ['emotion'] = [id_emotion[row] if row in id_emotion else '' for row in X['id']]
    X ['ident'] = [ident[row] if row in ident else '' for row in X['id']]
    train_df = X.query('ident == "train"')
    test_df = X.query('ident == "test"')
    return train_df, test_df

def training(train_frame):
    train_frame = train_frame.sample(frac=1)
    train_test_part = int(len(train_frame)*0.9)
    train_df, self_test_df = train_frame[:train_test_part], train_frame[train_test_part:]

    # text.texts_from_df return two tuples
    # maxlen=50 and rest of them are getting trucated
    # preprocess_mode: choose to use BERT model
    (X_train, y_train), (X_test, y_test), preprocess = text.texts_from_df(train_df = train_df,
                                                                           text_column = 'text',
                                                                           label_columns = 'emotion',
                                                                           val_df = self_test_df,
                                                                           maxlen = 50,
                                                                           preprocess_mode = 'bert',)
    # using BERT model
    model = text.text_classifier(name='bert', train_data = (X_train, y_train), preproc = preprocess)
    learner = ktrain.get_learner(model = model, train_data = (X_train, y_train), val_data = (X_test, y_test), batch_size = 32)

    # fit one cycle uses the one cycle policy callback
    learner.fit_onecycle(lr = 3e-5, epochs = 2, checkpoint_folder='checkpoint')

    # get predictor and save
    predictor = ktrain.get_predictor(learner.model, preproc = preprocess)
    predictor.save('predictor')
    
    
def main():
    all_data, id_emotion, ident = load_data()
    train_df_all, test_df_all = train_data(all_data, id_emotion, ident)
    training(train_df_all)
    predictor = ktrain.load_predictor('predictor')
    predict_result = predictor.predict([row for row in test_df_all['text']])
    test_df_all['emotion'] = predict_result
    test_df_all.drop(['text', 'ident'], axis=1, inplace=True)
    test_df_all.to_csv('result.csv', index=False, sep=',', header=True)
    
if __name__ == '__main__':
    main()