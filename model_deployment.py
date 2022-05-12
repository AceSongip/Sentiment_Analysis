# -*- coding: utf-8 -*-
"""
Created on Wed May 11 15:08:16 2022

@author: aceso
"""
#%% module
from tensorflow.keras.models import load_model
import os
import re
import numpy as np
import json
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences

#%% constant
MODEL_PATH = os.path.join(os.getcwd(), 'model.h5')
JSON_PATH = os.path.join(os.getcwd(), "tokenizer_data.json")

#%% model loading
sentiment_clf = load_model(MODEL_PATH)
sentiment_clf.summary()

#%% tokenizer loading
with open(JSON_PATH, 'r') as json_file:
    loaded_tokenizer = json.load(json_file)

# deploy
new_review = ["I think the first one hour is interesting but\
    the second half of the movie is <br /> boring. This movie just wasted my precious\
        time and hard earned money. This movie should be banned to avoid\
            time being wasted."]         
           
# remove funny character/ html character
for i, text in enumerate(new_review):
    new_review[i] = re.sub("<.*?>", "", text) # .* will select everything inside < >, ? not to go beyond it

# convert to lowercase and split it and remove numerical text
for i, text in enumerate(new_review):    
    new_review[i] = re.sub("[^a-zA-Z]", " ", text).lower().split()

# to vectorize the new review
loaded_tokenizer = tokenizer_from_json(loaded_tokenizer)
new_review = loaded_tokenizer.texts_to_sequences(new_review)
new_review = pad_sequences(new_review,
                           maxlen=200,
                           truncating= "post",
                           padding = "post")

#%% Model Prediction
result = sentiment_clf.predict(np.expand_dims(new_review, axis=-1)) # make sure new review in 3D(1,200,1)
print(np.argmax(result)) # only give num result

sentiment_dict = {1:"Positive", 0:"Negative"}
print(f"This review is {sentiment_dict[np.argmax(result)]}") # convert num in result into str 