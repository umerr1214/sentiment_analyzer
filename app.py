import streamlit as st
import joblib
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
import plotly.express as px

# Download NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Load models and vectorizer
@st.cache_resource
def load_models():
    vectorizer = joblib.load('tfidf_vectorizer.joblib')
    nb = joblib.load('naive_bayes_model.joblib')
    lr = joblib.load('logreg_model.joblib')
    return vectorizer, nb, lr

vectorizer, nb, lr = load_models()

# Model performance statistics (from your training results)
MODEL_STATS = {
    "Logistic Regression": {
        "accuracy": 0.93,
        "precision": 0.65,
        "recall": 0.39,
        "f1_score": 0.42,
        "recall_positive": 1.00,
        "recall_negative": 0.14,
        "recall_neutral": 0.04,
        "description": "Machine learning model trained on TF-IDF features with balanced class weights to handle imbalanced data.",
        "strengths": "Best overall performance, handles class imbalance well",
        "weaknesses": "Requires training data, may overfit on small datasets"
    },
    "Naive Bayes": {
        "accuracy": 0.93,
        "precision": 0.81,
        "recall": 0.34,
        "f1_score": 0.33,
        "recall_positive": 1.00,
        "recall_negative": 0.01,
        "recall_neutral": 0.01,
        "description": "Probabilistic classifier based on Bayes theorem, very fast and simple.",
        "strengths": "Very fast, simple to understand, good for large datasets",
        "weaknesses": "Struggles with class imbalance, assumes feature independence"
    },
    "VADER": {
        "accuracy": 0.76,
        "precision": 0.44,
        "recall": 0.47,
        "f1_score": 0.41,
        "recall_positive": 0.79,
        "recall_negative": 0.18,
        "recall_neutral": 0.46,
        "description": "Lexicon and rule-based sentiment analyzer specifically designed for social media and informal text.",
        "strengths": "No training required, understands context and emojis, very fast",
        "weaknesses": "Limited to English, may miss nuanced sentiments"
    },
    "TextBlob": {
        "accuracy": 0.36,
        "precision": 0.41,
        "recall": 0.41,
        "f1_score": 0.22,
        "recall_positive": 0.34,
        "recall_negative": 0.02,
        "recall_neutral": 0.87,
        "description": "Simple rule-based sentiment analyzer using pattern matching and lexicon.",
        "strengths": "Very simple, no training required, good for basic sentiment",
        "weaknesses": "Less accurate than other models, may miss context"
    }
}

# Preprocessing
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

def get_metric_explanation(metric_name, value):
    """Return simple explanation of what each metric means"""
    explanations = {
        "accuracy": f"**Accuracy ({value:.1%})**: Out of 100 predictions, this model gets {value*100:.0f} correct on average. This tells us how often the model is right overall.",
        "precision": f"**Precision ({value:.1%})**: When this model predicts a sentiment, it's correct {value*100:.0f}% of the time. Higher precision means fewer false alarms.",
        "recall": f"**Recall ({value:.1%})**: This model finds {value*100:.0f}% of all actual instances of each sentiment. Higher recall means it misses fewer cases.",
        "f1_score": f"**F1-Score ({value:.1%})**: This is the balance between precision and recall. A score of {value:.1%} means the model has a good balance of being accurate and finding most cases."
    }
    return explanations.get(metric_name, f"{metric_name}: {value:.1%}")

def explain_model_disagreement(predictions):
    """Explain why models might disagree based on their recall characteristics"""
    if len(set(predictions.values())) <= 1:
        return None
    
    explanations = []
    
    # Check for Naive Bayes bias towards positive
    if "Naive Bayes" in predictions and predictions["Naive Bayes"] == "positive":
        nb_stats = MODEL_STATS["Naive Bayes"]
        if nb_stats["recall_positive"] == 1.00 and nb_stats["recall_neutral"] == 0.01:
            explanations.append({
                "model": "Naive Bayes",
                "prediction": "positive",
                "reason": f"Naive Bayes has a strong bias towards predicting 'positive' (100% recall) and rarely predicts 'neutral' (only 1% recall). This means it tends to classify ambiguous or mildly positive text as 'positive' rather than 'neutral'."
            })
    
    # Check for TextBlob bias towards neutral
    if "TextBlob" in predictions and predictions["TextBlob"] == "neutral":
        tb_stats = MODEL_STATS["TextBlob"]
        if tb_stats["recall_neutral"] == 0.87 and tb_stats["recall_positive"] == 0.34:
            explanations.append({
                "model": "TextBlob",
                "prediction": "neutral",
                "reason": f"TextBlob has a high recall for 'neutral' (87%) but low recall for 'positive' (34%). This means it's more likely to classify mildly positive text as 'neutral' rather than 'positive'."
            })
    
    # Check for VADER's balanced approach
    if "VADER" in predictions:
        vader_stats = MODEL_STATS["VADER"]
        if vader_stats["recall_neutral"] == 0.46 and vader_stats["recall_positive"] == 0.79:
            explanations.append({
                "model": "VADER",
                "prediction": predictions["VADER"],
                "reason": f"VADER has more balanced recall scores: 79% for 'positive', 46% for 'neutral', and 18% for 'negative'. This makes it less biased towards any single class."
            })
    
    # Check for Logistic Regression's characteristics
    if "Logistic Regression" in predictions:
        lr_stats = MODEL_STATS["Logistic Regression"]
        if lr_stats["recall_positive"] == 1.00 and lr_stats["recall_neutral"] == 0.04:
            explanations.append({
                "model": "Logistic Regression",
                "prediction": predictions["Logistic Regression"],
                "reason": f"Logistic Regression has high recall for 'positive' (100%) but very low recall for 'neutral' (4%). However, it was trained with balanced class weights, which should help reduce bias."
            })
    
    return explanations

# Page config and style
st.set_page_config(page_title="Sentiment Studio", page_icon="💬", layout="wide")

st.markdown("""
    <style>
        h1 {
            color: #1f77b4;
            text-align: center;
            margin-top: 0;
        }
        .card {
            padding: 1rem;
            background: #f7f9fa;
            border-radius: 12px;
            border-left: 5px solid #1f77b4;
            margin-bottom: 1rem;
        }
        .sentiment {
            font-size: 1.2rem;
            font-weight: bold;
        }
        .positive { color: #28a745; }
        .negative { color: #dc3545; }
        .neutral { color: #6c757d; }
        .stats-card {
            background: white;
            padding: 1rem;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin: 0.5rem 0;
        }
        .metric-highlight {
            background: #e3f2fd;
            padding: 0.5rem;
            border-radius: 5px;
            margin: 0.5rem 0;
        }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown("<h1>Sentiment Studio</h1>", unsafe_allow_html=True)
st.markdown("Analyze your product reviews using multiple sentiment analysis techniques.")

# Sidebar
st.sidebar.header("⚙️ Choose Model")
model_choice = st.sidebar.selectbox(
    "Select a model:",
    ["All Models", "Logistic Regression", "Naive Bayes", "VADER", "TextBlob"]
)

# Input section
user_input = st.text_area(
    "📝 Enter your product review:",
    height=150,
    placeholder="Example: This product exceeded my expectations. The quality is amazing!"
)

# Button
if st.button("Analyze Sentiment 🚀", use_container_width=True):
    if user_input.strip() == "":
        st.error("⚠️ Please enter a review.")
    else:
        with st.spinner("Analyzing sentiment, please wait..."):
            predictions = {}

            if model_choice in ["All Models", "Logistic Regression"]:
                cleaned = clean_text(user_input)
                X_input = vectorizer.transform([cleaned])
                predictions["Logistic Regression"] = lr.predict(X_input)[0]

            if model_choice in ["All Models", "Naive Bayes"]:
                cleaned = clean_text(user_input)
                X_input = vectorizer.transform([cleaned])
                predictions["Naive Bayes"] = nb.predict(X_input)[0]

            if model_choice in ["All Models", "VADER"]:
                predictions["VADER"] = vader_sentiment(user_input)

            if model_choice in ["All Models", "TextBlob"]:
                predictions["TextBlob"] = textblob_sentiment(user_input)

            # Display results
            st.subheader("🔍 Predictions")
            cols = st.columns(len(predictions))
            for i, (model, pred) in enumerate(predictions.items()):
                sentiment_class = "positive" if pred == "positive" else "negative" if pred == "negative" else "neutral"
                cols[i].markdown(f"""
                    <div class="card">
                        <h4>{model}</h4>
                        <p class="sentiment {sentiment_class}">{pred.upper()}</p>
                    </div>
                """, unsafe_allow_html=True)

            # Model Statistics and Explanations
            st.subheader("📊 Model Performance & Explanation")
            
            for model_name, pred in predictions.items():
                if model_name in MODEL_STATS:
                    stats = MODEL_STATS[model_name]
                    
                    st.markdown(f"### {model_name} Analysis")
                    
                    # Model description
                    st.markdown(f"**About this model:** {stats['description']}")
                    
                    # Performance metrics
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("**Performance Metrics:**")
                        st.markdown(f"<div class='metric-highlight'>{get_metric_explanation('accuracy', stats['accuracy'])}</div>", unsafe_allow_html=True)
                        st.markdown(f"<div class='metric-highlight'>{get_metric_explanation('precision', stats['precision'])}</div>", unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown("**More Metrics:**")
                        st.markdown(f"<div class='metric-highlight'>{get_metric_explanation('recall', stats['recall'])}</div>", unsafe_allow_html=True)
                        st.markdown(f"<div class='metric-highlight'>{get_metric_explanation('f1_score', stats['f1_score'])}</div>", unsafe_allow_html=True)
                    
                    # Strengths and weaknesses
                    col3, col4 = st.columns(2)
                    with col3:
                        st.markdown("**✅ Strengths:**")
                        st.info(stats['strengths'])
                    
                    with col4:
                        st.markdown("**⚠️ Limitations:**")
                        st.warning(stats['weaknesses'])
                    
                    # Prediction confidence explanation
                    if model_name in ["Logistic Regression", "Naive Bayes"]:
                        st.markdown("**🤔 Why this prediction?**")
                        st.markdown("This machine learning model was trained on thousands of product reviews. It learned patterns in words and phrases that typically indicate positive, negative, or neutral sentiment.")
                    elif model_name == "VADER":
                        st.markdown("**🤔 Why this prediction?**")
                        st.markdown("VADER analyzed your text using a pre-built dictionary of words with sentiment scores, plus rules for handling punctuation, capitalization, and context.")
                    elif model_name == "TextBlob":
                        st.markdown("**🤔 Why this prediction?**")
                        st.markdown("TextBlob used simple pattern matching and a basic sentiment lexicon to determine the overall polarity of your text.")
                    
                    st.markdown("---")

            # Pie chart with model labels
            st.subheader("📈 Sentiment Distribution")
            
            # Create a more detailed visualization showing which models predicted each sentiment
            sentiment_model_mapping = {}
            for model, sentiment in predictions.items():
                if sentiment not in sentiment_model_mapping:
                    sentiment_model_mapping[sentiment] = []
                sentiment_model_mapping[sentiment].append(model)
            
            # Create pie chart data
            pie_data = []
            for sentiment, models in sentiment_model_mapping.items():
                pie_data.append({
                    'sentiment': sentiment,
                    'count': len(models),
                    'models': ', '.join(models)
                })
            
            # Create the pie chart
            fig = px.pie(
                data_frame=pd.DataFrame(pie_data),
                values='count',
                names='sentiment',
                color='sentiment',
                color_discrete_map={
                    'positive': '#28a745',
                    'negative': '#dc3545',
                    'neutral': '#6c757d'
                },
                title="Sentiment Distribution by Model Predictions"
            )
            
            # Update hover template to show which models predicted each sentiment
            fig.update_traces(
                hovertemplate="<b>%{label}</b><br>" +
                            "Count: %{value}<br>" +
                            "Models: %{customdata}<br>" +
                            "<extra></extra>",
                customdata=[item['models'] for item in pie_data]
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Show detailed breakdown below the chart
            st.markdown("**Detailed Breakdown:**")
            for sentiment, models in sentiment_model_mapping.items():
                sentiment_color = {
                    'positive': '🟢',
                    'negative': '🔴', 
                    'neutral': '⚪'
                }.get(sentiment, '⚪')
                
                st.markdown(f"{sentiment_color} **{sentiment.title()}** ({len(models)} model{'s' if len(models) > 1 else ''}): {', '.join(models)}")

            # Agreement check
            if model_choice == "All Models":
                st.subheader("🤝 Model Agreement")
                unique_predictions = set(predictions.values())
                if len(unique_predictions) == 1:
                    st.success("✅ All models agree on the sentiment!")
                else:
                    st.warning("⚠️ Models do not agree.")
                    df_agree = []
                    for i, (m1, p1) in enumerate(predictions.items()):
                        for j, (m2, p2) in enumerate(predictions.items()):
                            if i < j:
                                df_agree.append({
                                    "Model 1": m1,
                                    "Model 2": m2,
                                    "Agreement": "✅" if p1 == p2 else "❌"
                                })
                    st.dataframe(pd.DataFrame(df_agree), use_container_width=True)

            # Explain model disagreements
            st.subheader("🤔 Why might models disagree?")
            explanations = explain_model_disagreement(predictions)
            if explanations:
                # Remove duplicates based on model name
                seen_models = set()
                unique_explanations = []
                for exp in explanations:
                    if exp['model'] not in seen_models:
                        unique_explanations.append(exp)
                        seen_models.add(exp['model'])
                
                for exp in unique_explanations:
                    st.markdown(f"**{exp['model']} prediction for '{exp['prediction']}':** {exp['reason']}")
            else:
                st.info("All models predict the same sentiment or no disagreement to explain.")

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; font-size: 0.9rem; color: #6c757d;'>Made with ❤️ using Streamlit · Sentiment Studio</div>",
    unsafe_allow_html=True
)
