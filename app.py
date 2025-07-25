import streamlit as st
import joblib
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob

# Download NLTK resources (only needed first time)
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Load models and vectorizer
vectorizer = joblib.load('tfidf_vectorizer.joblib')
nb = joblib.load('naive_bayes_model.joblib')
lr = joblib.load('logreg_model.joblib')

# Preprocessing function (should match your training pipeline)
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    if pd.isnull(text):
        return ''
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = [word for word in text.split() if word not in stop_words]
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(tokens)

# VADER and TextBlob functions
analyzer = SentimentIntensityAnalyzer()

def vader_sentiment(text):
    score = analyzer.polarity_scores(str(text))['compound']
    if score >= 0.5:
        return 'positive'
    elif score <= -0.5:
        return 'negative'
    else:
        return 'neutral'

def textblob_sentiment(text):
    polarity = TextBlob(str(text)).sentiment.polarity
    if polarity >= 0.5:
        return 'positive'
    elif polarity <= -0.5:
        return 'negative'
    else:
        return 'neutral'

# Streamlit UI
st.title("Sentiment Analysis Demo")

user_input = st.text_area("Enter a product review:")

if st.button("Predict Sentiment"):
    if user_input.strip() == "":
        st.warning("Please enter a review.")
    else:
        # Preprocess and vectorize
        cleaned = clean_text(user_input)
        X_input = vectorizer.transform([cleaned])

        # Classical models
        pred_nb = nb.predict(X_input)[0]
        pred_lr = lr.predict(X_input)[0]

        # Pretrained models
        pred_vader = vader_sentiment(user_input)
        pred_textblob = textblob_sentiment(user_input)

        st.subheader("Predictions:")
        st.write(f"**Logistic Regression:** {pred_lr}")
        st.write(f"**Naive Bayes:** {pred_nb}")
        st.write(f"**VADER:** {pred_vader}")
        st.write(f"**TextBlob:** {pred_textblob}")