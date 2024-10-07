#!/usr/bin/env python
# coding: utf-8

import streamlit as st
import pickle
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk
nltk.data.path.append('./nltk_data')
nltk.download('punkt', download_dir='/home/appuser/.nltk_data')
nltk.download('punkt_tab', download_dir='/home/appuser/.nltk_data')
nltk.download('stopwords', download_dir='/home/appuser/.nltk_data')
nltk.download('vader_lexicon', download_dir='/home/appuser/.nltk_data')

# Title of the app
st.title("📧 Email Classification: Spam or Ham")

model_path = r"NBModel.pkl"
vectorizer_path = r"tfidf.pkl"

with open(model_path, 'rb') as model_file:
    model = pickle.load(model_file)

with open(vectorizer_path, 'rb') as vectorizer_file:
    tfidf = pickle.load(vectorizer_file)

def preprocess_text(text):
    text = text.strip().lower()  # Remove whitespace and convert to lowercase
    text = ''.join([char for char in text if char not in string.punctuation])  # Remove punctuation
    tokens = word_tokenize(text)  # Tokenization
    stopwords_list = stopwords.words('english')  # English stopwords
    tokens = [word for word in tokens if word not in stopwords_list]  # Remove stopwords
    ps = PorterStemmer()  # Stemmer instance
    stems = [ps.stem(word) for word in tokens]  # Stem the tokens
    return " ".join(stems)

def vaderAnalyzer(text):
    """Analyze the sentiment of the input text using VADER."""
    vader_analyzer = SentimentIntensityAnalyzer()
    sentiment_score = vader_analyzer.polarity_scores(text)
    if sentiment_score['compound'] > 0:
        return "Positive"
    elif sentiment_score['compound'] < 0:
        return "Negative"
    else:
        return "Neutral"


input_text = st.text_area("✍️ Enter your email text here:", height=150)


if input_text:
    processed_text = preprocess_text(input_text)  
    x = tfidf.transform([processed_text])  
    output = model.predict(x)  

    sentiment = vaderAnalyzer(input_text) 

    # Customized output messages
    if output[0].lower() == 'spam':
        st.markdown("### 🛑 Classification Result:")
        st.success("The email is classified as **Spam**!")
        st.markdown(f"**Sentiment Analysis:** The email carries a **{sentiment}** sentiment.")
        st.markdown("**Warning:** This email may contain unsolicited offers or phishing attempts. Exercise caution!")
    else:
        st.markdown("### ✅ Classification Result:")
        st.success("The email is classified as **Ham (Not Spam)**!")
        st.markdown(f"**Sentiment Analysis:** The email carries a **{sentiment}** sentiment.")
        st.markdown("**Note:** This email appears to be legitimate. You can proceed with caution.")

