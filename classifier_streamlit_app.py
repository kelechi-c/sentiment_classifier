#Import statements
import streamlit as st
import numpy as np
import pandas as pd
import time
import tracemalloc

from bs4 import BeautifulSoup
from keras.models import load_model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.firefox.service import Service
from selenium import webdriver

# Memory allocation tracing
tracemalloc.start()


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

# Text tokenizer
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

# Process text for classification
def preprocess_text(text):
  text = text.lower()
  input_sequence = tokenizer.texts_to_sequences([text])
  input_padded = pad_sequences(input_sequence, maxlen=100, padding='post')
  
  return input_padded

# Load and initialize pretrained model 
classifier = load_model('sentiment_model_tf.h5')

input_text = st.text_input('Type in sentence')
input = preprocess_text(input_text)

# Classifier function
score = classifier.predict(input)

# def get_output():
sentiment_classes = {0:'neutral', 1:'positive', 2:'negative'}
predicted_class = np.argmax(score)
certainty = 100 * np.max(score)
    
    # return sentiment_classes, predicted_class, certainty

if st.button('Classify Text'):
    with st.spinner("Running..."):
        sent_class = predicted_class
        if sent_class:
            st.markdown(f'''
            <div style="background-color: black; color: white; font-weight: bold; padding: 1rem; border-radius: 10px;">
            <h4>Results</h4>
                <h5>Tweet text: </h5>
                <p>{input_text}</p>
                <p>
                  Predicted connotation => <span style="font-weight: bold;">{sentiment_classes[predicted_class]}</span> with <span style="font-weight: bold;">{certainty:.2f}% </span>certainty
                </p>
            </div>
                ''', unsafe_allow_html=True)
            st.success('Successful')
    
    
    
st.write('')
st.write('')


# Tweet scraping functionality

# Driver headless mode
opts = Options()
opts.add_argument('--headless')
opts.add_argument('--disable-gpu')
st.subheader('Tweet URL input')
tweet_url = st.text_input('Paste tweet URL to extract tweet')


def get_driver():
    return webdriver.Chrome(service=Service("driver/geckodriver"), options=opts)


def scrape_tweet_url(url):
    driver = get_driver() 
    driver.get(url)
    
    response = driver.page_source
    driver.quit()
    
    tweet_soup = BeautifulSoup(response, 'html.parser')
    
    try:
        # Get Tweet text with bs4 
        tweet_source = tweet_soup.find("div",{"data-testid":"tweetText"})
        
        tweet_text = tweet_source.find('span', class_='css-1qaijid r-bcqeeo r-qvutc0 r-poiln3').text

    except:
        tweet_text = None
        st.markdown('''<span style='background: darkred; color: white;'> 404. Tweet Not Found</span>''', unsafe_allow_html=True)
        
        
    return tweet_text


def scrape_and_classify(scrape_function):
    try: # Exception handling
        tweet = scrape_function(tweet_url)
        processed_tweet = preprocess_text(tweet)
        score = classifier.predict(processed_tweet)
        predict_class = np.argmax(score)
        percent_certainty = 100 * np.max(score)
        
        # Print output
        st.markdown(f'''
            <div style="background-color: black; color: white; font-weight: bold; padding: 1rem; border-radius: 10px;">
               <h4>Results</h4>
               <h5>Tweet text: </h5>
                <p>{tweet}</p>
                <p>
                 Predicted connotation => <span style="font-weight: bold;">{sentiment_classes[predict_class]}</span> with <span style="font-weight: bold;">{percent_certainty:.2f}% </span>certainty
                </p>
            </div>
        ''', unsafe_allow_html=True)
    
    except:
        st.markdown('''
                    <p style='background: maroon; color: white; font-weight: bold; padding: 1rem; border-radius: 10px;'> 404. Tweet Not Found. Enter a valid link </p>
        ''', unsafe_allow_html=True)
        # Fallback message(In case of error)


if st.button('Check tweet'):
  result = scrape_and_classify(scrape_tweet_url)
  scrape_and_classify(scrape_tweet_url(tweet_url)) # Final function
  
# Process Running signal
with st.spinner("Running..."):
  result = scrape_and_classify(scrape_function=scrape_tweet_url)
  if result:
      st.success('Successful')
  

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