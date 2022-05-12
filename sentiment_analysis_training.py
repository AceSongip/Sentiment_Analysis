# -*- coding: utf-8 -*-
"""
Created on Thu May 12 11:44:28 2022

-This class is used to do model training
- We can also use transfer learning but there are 
two things to consider(input_size, output_size)

@author: aceso
"""
#%% module
import pandas as pd
import os 
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import datetime
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import TensorBoard
from sentiment_analysis_function import ExploratoryDataAnalysis, ModelBuilding # import file that we just created
from sentiment_analysis_function import ModelEvaluation


#%% Constant
URL = "https://raw.githubusercontent.com/Ankit152/IMDB-sentiment-analysis/master/IMDB-Dataset.csv"
TOKENIZER_PATH = os.path.join(os.getcwd(), 'tokenizer_data2.json')
PATH_LOGS = os.path.join(os.getcwd(), 'log')
log_dir = os.path.join(PATH_LOGS, datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
MODEL_SAVE_PATH = os.path.join(os.getcwd(), 'model.h5')

#%% EDA 
# Step 1) Import data
df = pd.read_csv(URL)
review = df["review"]
sentiment = df["sentiment"]

# Step 2) Data cleaning

eda = ExploratoryDataAnalysis() # class

review = eda.remove_tags(review) # Remove tags
review = eda.lower_split(review) #lowercase

# Step 3) Feature Selection
# Step 4) Data vectorization
# Tokenization
review = eda.sentiment_tokenizer(review, TOKENIZER_PATH)
print(review[2])

# Pad Sequence
review = eda.sentiment_pad_sequences(review)

# Step 5) Data preprocessing <-- not in function bcos only use once
# One hot encoding
hot_encoder = OneHotEncoder(sparse=False)
nb_categories = len(sentiment.unique())
encoded_sentiment = hot_encoder.fit_transform(np.expand_dims(sentiment, axis=-1))

# Train test split(X = review, y = sentiment)
X_train, X_test, y_train, y_test = train_test_split(review, 
                                                    encoded_sentiment, 
                                                    test_size=0.3, 
                                                    random_state=123)

# expand training data into 3D array
X_train = np.expand_dims(X_train, axis=-1)
X_test = np.expand_dims(X_test, axis=-1)

# inverse to see either it postive or negative
# positive
print(y_train[0]) #[0,1]
print(hot_encoder.inverse_transform(np.expand_dims(y_train[0], axis=0))) 
# negative
print(y_train[1]) #[1,0]
print(hot_encoder.inverse_transform(np.expand_dims(y_train[1], axis=0))) 

#%% Model Building

mb = ModelBuilding()
num_words = 10000
nb_categories = len(sentiment.unique())

model = mb.lstm_layer(num_words, nb_categories)
model.compile(optimizer="adam", 
              loss="categorical_crossentropy", 
              metrics="acc")

tensorboard = TensorBoard(log_dir, histogram_freq=1)

model.fit(X_train, y_train, epochs=3, 
          validation_data=(X_test, y_test),
          callbacks=tensorboard)

#%% Model Evaluation & Analysis
predicted_advanced = np.empty([len(X_test), 2]) 
for i, test in enumerate(X_test):
    predicted_advanced[i,:] = model.predict(np.expand_dims(test, axis=0))

# Model analysis
y_pred = np.argmax(predicted_advanced, axis=1) 
y_true = np.argmax(y_test, axis=1)

evals = ModelEvaluation()
result = evals.evaluation(y_true, y_pred)

#%% Model Saving
model.save(MODEL_SAVE_PATH)



