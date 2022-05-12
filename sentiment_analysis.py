# -*- coding: utf-8 -*-
"""
Created on Wed May 11 09:15:13 2022

Other than time series, NLP also consider as sequence data

@author: aceso
"""
#%% Reference
# Replace string

a = "I am a boy 12345"
print(a.replace("12345", "")) # (old, new)

# or we can use regex
import re

# remove numerical data
print(re.sub("[^a-zA-Z]", " ", a)) # (old, new, data), ^ start with, $ end with
# [] <-- something target to analyse
# ^ not included

#%% Modules
import pandas as pd
import numpy as np
import re
import os
import datetime
from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, Embedding, Bidirectional
from tensorflow.keras.callbacks import TensorBoard
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import json

# Constant
URL = "https://raw.githubusercontent.com/Ankit152/IMDB-sentiment-analysis/master/IMDB-Dataset.csv"

PATH_LOGS = os.path.join(os.getcwd(), 'log')

log_dir = os.path.join(PATH_LOGS, datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))

TOKENIZER_PATH = os.path.join(os.getcwd(), 'tokenizer_data.json')

MODEL_SAVE_PATH = os.path.join(os.getcwd(), 'model.h5')


#%% EDA

# 1) Load data
df = pd.read_csv(URL)

review = df["review"] # data with all words, this will be features
df_review = review.copy()

sentiment = df["sentiment"] # +ve / -ve review, this will be label
df_sentiment = sentiment.copy()

#%% Step 2) DATA INSPECTION

# let's see review num 3
df_review[3] 
df_sentiment[3]

#%% Step 3) DATA CLEANING
# since in review there's "<br /><br />" so we need to remove it

# remove funny character/ html character
for i, text in enumerate(df_review):
    df_review[i] = re.sub("<.*?>", "", text) # .* will select everything inside < >, ? not to go beyond it

# convert to lowercase and split it and remove numerical text
for i, text in enumerate(df_review):    
    df_review[i] = re.sub("[^a-zA-Z]", " ", text).lower().split()

#%% Step 4 ) FEATURE SELECTION <- NOTHING TO SELECT

#%% Step 5 ) DATA PREPROCESSING

# Data Vectorization for reviews(tokenization)
num_words = 10000 # pick carefully refers on data shape
oov_token = "<OOV>" # this use when new unknown word(eg:spaghetification) will be replace with oov

tokenizer = Tokenizer(num_words=num_words, oov_token=(oov_token))
tokenizer.fit_on_texts(df_review)

# Tokenizer need to be saved for deployment purpose
token_jason = tokenizer.to_json()
with open(TOKENIZER_PATH , "w") as json_file:
    json.dump(token_jason, json_file)

word_index = tokenizer.word_index # give address to all words
print(word_index) # all words will be assign with address/index

df_review = tokenizer.texts_to_sequences(df_review) # vectorize the sequences of text

# every index of sequence have diff length so we need to pad sequence
df_review = pad_sequences(df_review, maxlen=200, padding="post", truncating="post")
# maxlen 200, means in a line max length is 200, why 200 because each line have avg 200 words
# post padding means if a line have fewer than 200 then it will pad remaining with 0


# One-hot encoding for label
hot_encoder = OneHotEncoder(sparse=False)
df_sentiment.unique() # only 2(positive and negative)
encoded_sentiment = hot_encoder.fit_transform(np.expand_dims(df_sentiment, axis=-1))

# Train test split
X_train, X_test, y_train, y_test = train_test_split(df_review, 
                                                    encoded_sentiment, 
                                                    test_size=0.3, 
                                                    random_state=123)

# expand training data into 3D array
X_train = np.expand_dims(X_train, axis=-1)
X_test = np.expand_dims(X_test, axis=-1)


#%% Step 4) Model Building

# Old but slow method without embedding and biderectional
#model = Sequential()
#model.add(LSTM(128, input_shape=(np.shape(X_train)[1:]),return_sequences=True))
#model.add(Dropout(0.2))
#model.add(LSTM(128))
#model.add(Dropout(0.2))
#model.add(Dense(2, activation="softmax")) # softmax because classification between +ve & -ve
#model.summary()

# New method
model = Sequential()
model.add(Embedding(num_words, 64)) # added the embedding layer, embedding doesn't need input, much faster and better with this approach
model.add(Bidirectional(LSTM(32, return_sequences=True))) # added bidirectional
model.add(Dropout(0.2))
model.add(Bidirectional(LSTM(32)))
model.add(Dropout(0.2))
model.add(Dense(2, activation="softmax")) # softmax because classification between +ve & -ve
model.summary()

#%% callbacks
tensorboard = TensorBoard(log_dir, histogram_freq=1)

#%% Compile and Training
model.compile(optimizer="adam", 
              loss="categorical_crossentropy", 
              metrics="acc")

model.fit(X_train, y_train, epochs=3, 
          validation_data=(X_test, y_test),
          callbacks=tensorboard)

#%% Step 5) Model evaluation

# Method 1) append approach
#predicted = []
#for i in X_test:
#    predicted.append(model.predict(np.expand_dims(i, axis=0))) #testing data need to change dimension

#predicted = np.array(predicted) # convert from list to array
# above process took so long, here is altenative(pre-allocated memory and overide)

# Method 2) preallocation of memory approach <-- NOTE: this approach faster
predicted_advanced = np.empty([len(X_test), 2]) # make 2 empty label columns based on the len of X_test
for i, test in enumerate(X_test):
    predicted_advanced[i,:] = model.predict(np.expand_dims(test, axis=0))

#%% Step 6) Model analysis

# we use np.argmax just to convert float into int for better analysis
y_pred = np.argmax(predicted_advanced, axis=1) # argmax to convert hot encoding data into single value(eg:0 or 1)
y_true = np.argmax(y_test, axis=1)

# see the report
print(classification_report(y_true, y_pred)) # classification report
print(confusion_matrix(y_true, y_pred)) # confusion matrix
print(accuracy_score(y_true, y_pred)) # accuracy

#%% Model Saving
model.save(MODEL_SAVE_PATH)



