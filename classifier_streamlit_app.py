import streamlit as st
import tensorflow as tf
import matplotlib.pyplot as plt
import datetime
import numpy as np
import pandas as pd
import h5py

from tensorflow import keras as tfkeras
from keras import layers, losses
from keras.models import load_model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
# from keras.preprocessing.sequence import pad_sequences

#data
reddit_data = 'Reddit_Data.csv'
dataset = pd.read_csv(reddit_data)

comments = dataset['clean_comment']
categories = dataset['category']

sentences = []
labels = []

for sentence in comments:
  sentences.append(str(sentence))

for category in categories:
  labels.append(category)

train_size = 28000

train_text = sentences[0:train_size]
test_text = sentences[train_size:]

tokenizer = Tokenizer(num_words=48244, oov_token='<OOV>')
tokenizer.fit_on_texts(train_text)


# Streamlit App



def preprocess_text(text):
  text = text.lower()
  input_sequence = tokenizer.texts_to_sequences([text])
  input_padded = pad_sequences(input_sequence, maxlen=100, padding='post')
  
  return input_padded

classifier = load_model('sentiment_model_tf.h5')

input_text = st.text_input('Type in sentence')
input = preprocess_text(input_text)

score = classifier.predict(input)

sentiment_classes = {0:'neutral', 1:'positive', 2:'negative'}
predicted_class = np.argmax(score)
certainty = 100 * np.max(score)

if st.button('Classify Text'):
    st.write(f'Input Sentence: {input_text}')
    st.write(f'Predicted: {sentiment_classes[predicted_class]}')
    st.write(f'Certainty: {certainty:.2f}%')