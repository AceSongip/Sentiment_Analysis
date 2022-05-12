# -*- coding: utf-8 -*-
"""
Created on Thu May 12 14:21:18 2022

@author: aceso
"""

from tensorflow.keras.models import load_model
import os
import json
import numpy as np
from sentiment_analysis_function import ExploratoryDataAnalysis
from tensorflow.keras.preprocessing.text import tokenizer_from_json
import warnings

warnings.filterwarnings("ignore")

MODEL_PATH = os.path.join(os.getcwd(), 'model.h5')
JSON_PATH = os.path.join(os.getcwd(), "tokenizer_data2.json")

#%% model loading
sentiment_clf = load_model(MODEL_PATH)
sentiment_clf.summary()

#%% tokenizer loading
with open(JSON_PATH, 'r') as json_file:
    token = json.load(json_file)
    
#%% Example review

new_review = [input("What is your review about this movie?: ")]

            
#%% 
# Step 1) clean the data
eda = ExploratoryDataAnalysis()
clean_review = eda.remove_tags(new_review)
clean_review = eda.lower_split(clean_review)

#%% Data preprocessing

loaded_tokenizer = tokenizer_from_json(token)

token_review = loaded_tokenizer.texts_to_sequences(clean_review)
pad_review = eda.sentiment_pad_sequences(token_review)

#%% model prediction

result = sentiment_clf.predict(np.expand_dims(pad_review, axis=-1)) # make sure new review in 3D(1,200,1)
print(np.argmax(result)) # only give num result

sentiment_dict = {1:"Positive", 0:"Negative"}
print(f"This review is {sentiment_dict[np.argmax(result)]}")

