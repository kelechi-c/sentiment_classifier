#Import statements
import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import time

from selenium.webdriver.chrome.options import Options
from selenium import webdriver
from bs4 import BeautifulSoup
from keras.models import load_model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences


# Data preparation
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

# Overview
st.title('Text Sentiment Classification')
st.subheader('Overview')
st.markdown('''
         This is a simple **machine learning** web app to detect positive/negative sentiment in the given text.
         It is an application of Natural language processing using Tensorflow, and web scraping(for Tweets). 
    ''')

#Process text for classification
def preprocess_text(text):
  text = text.lower()
  input_sequence = tokenizer.texts_to_sequences([text])
  input_padded = pad_sequences(input_sequence, maxlen=100, padding='post')
  
  return input_padded

# Load and initialize pretrained model 
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
    
    
st.write('')
st.write('')


# Tweet scraping functionality

# Driver headless mode
chrome_opts = Options()
chrome_opts.add_argument('--headless')
st.subheader('Tweet URL input')
tweet_url = st.text_input('Paste tweet URL to extract tweet')

def scrape_tweet_url(url):
        
    driver = webdriver.Chrome(options=chrome_opts) # Initialize web driver
    driver.minimize_window()
    driver.get(url)
    time.sleep(2)
    
    response = driver.page_source
    driver.close()
    
    tweet_soup = BeautifulSoup(response, 'html.parser')
    
    try:
        # Get Tweet text with bs4 
        tweet_source = tweet_soup.find("div",{"data-testid":"tweetText"})
        tweet_text = tweet_source.find('span', class_='css-1qaijid r-bcqeeo r-qvutc0 r-poiln3').text

    except:
        tweet_text = None
        st.write('      404. Tweet Not Found')
        
    return tweet_text



def scrape_and_classify(scrape_function):
    try: # Exception handling
        if st.button('Check tweet'):
          tweet = scrape_function(tweet_url)
          processed_tweet = preprocess_text(tweet)
          score = classifier.predict(processed_tweet)
          predicted_class = np.argmax(score)
          certainty = 100 * np.max(score)
          
          # Print output
          st.write(f'Tweet text: {tweet}')
          st.write(f'Predicted: {sentiment_classes[predicted_class]}')
          st.write(f'Certainty: {certainty:.2f}%')
    
    except:
        st.write('404 Not found. Error occured in retrieving tweet')
        # Fallback message(In case of error)
    
scrape_and_classify(scrape_tweet_url) # Final function

# Footer

st.write('')
st.write('')
st.write('')
st.write('')


st.markdown("<hr style='border: 1px dashed #ddd; margin: 2rem;'>", unsafe_allow_html=True) #Horizontal line

st.markdown("""
    <div style="text-align: center; padding: 1rem;">
        Project by <a href="https://github.com/ChibuzoKelechi" target="_blank" style="color: white; font-weight: bold; text-decoration: none;">
         kelechi_tensor</a>
    </div>
    
    <div style="text-align: center; padding: 1rem;">
        Data from <a href="https://kaggle.com" target="_blank" style="color: lightblue; font-weight: bold; text-decoration: none;">
         Kaggle</a>
    </div>
""",
unsafe_allow_html=True)

# Peace Out :)