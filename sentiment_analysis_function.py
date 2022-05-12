# -*- coding: utf-8 -*-
"""
Created on Thu May 12 09:53:24 2022

@author: aceso
"""
#%%Module
import pandas as pd
import re
import os
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, Embedding, Bidirectional
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import json

TOKENIZER_PATH = os.path.join(os.getcwd(), 'tokenizer_data2.json')
URL = "https://raw.githubusercontent.com/Ankit152/IMDB-sentiment-analysis/master/IMDB-Dataset.csv"

#%% Data Cleaning

class ExploratoryDataAnalysis():
    
    def __init__(self):
        pass
    
    # Remove tags
    def remove_tags(self,data):
        for i, text in enumerate(data):
            data[i] = re.sub("<.*?>", "", text)
            
        return data
    
    # lower
    def lower_split(self,data):
        for i, text in enumerate(data):    
            data[i] = re.sub("[^a-zA-Z]", " ", text).lower().split()
            
        return data
    
    # Tokenizer
    def sentiment_tokenizer(self, data, token_save_path, num_words=10000, 
                         oov_token="<oov>"):
        # Build Token
        tokenizer = Tokenizer(num_words=num_words, oov_token=(oov_token))
        tokenizer.fit_on_texts(data)
        
        # Save Token
        token_jason = tokenizer.to_json()
        with open(TOKENIZER_PATH , "w") as json_file:
            json.dump(token_jason, json_file)
        word_index = tokenizer.word_index # give address to all words
        print(word_index)
        data = tokenizer.texts_to_sequences(data)
        
        return data
    
    # Pad sequence    
    def sentiment_pad_sequences(self, data):
        data = pad_sequences(data, maxlen=200, padding="post", truncating="post")
        
        return data

#%% Model Creation

class ModelBuilding():
    
    def lstm_layer(self, num_words, nb_categories,
                   embedding_output=64, nodes=32, dropout=0.2):
        
        model = Sequential()
        model.add(Embedding(num_words, embedding_output)) # added the embedding layer, embedding doesn't need input, much faster and better with this approach
        model.add(Bidirectional(LSTM(nodes, return_sequences=True))) # added bidirectional
        model.add(Dropout(dropout))
        model.add(Bidirectional(LSTM(nodes)))
        model.add(Dropout(dropout))
        model.add(Dense(nb_categories, activation="softmax")) # softmax because classification between +ve & -ve
        model.summary()
        
        return model

    def simple_lstm_layer(self, num_words, nb_categories,
                   embedding_output=64, nodes=32, dropout=0.2):
        
        model = Sequential()
        model.add(Embedding(num_words, embedding_output)) 
        model.add(LSTM(nodes, return_sequences=True))
        model.add(Dropout(dropout))
        model.add(Dense(nb_categories, activation="softmax")) 
        model.summary()
        
        return model
    
#%% Model evaluation

class ModelEvaluation():
    
    def evaluation(self, y_true, y_pred):
        print(classification_report(y_true, y_pred)) # classification report
        print(confusion_matrix(y_true, y_pred)) # confusion matrix
        print(accuracy_score(y_true, y_pred))

#%% only for testing purpose

if __name__ == "__main__": # this code to make sure below line doesnt get into another folder after export this file

    # 1) Load data
    df = pd.read_csv(URL)
    review = df["review"]
    
    # Data cleaning class testing
    eda = ExploratoryDataAnalysis()
    
    # Remove tag
    test = eda.remove_tags(review)
    # lower the case
    test = eda.lower_split(test) # function 2
    
    # Tokenization testing
    test = eda.sentiment_tokenizer(test, None)
    
    # Pad sequence
    test = eda.sentiment_pad_sequences(test)
    # convert to lowercase and split it and remove numerical text
    #%% model testing
    nn = ModelBuilding()
    model = nn.lstm_layer(10000, 2)
