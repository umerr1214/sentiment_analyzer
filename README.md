# Sentiment Studio

**Sentiment Studio** is an  web application for analyzing the sentiment of product reviews using both classical machine learning models and pretrained sentiment analysis tools. 

---

## Features

### 📝 Single Review Analysis
- Enter a product review in a text box.
- Get sentiment predictions from four models:
  - **Logistic Regression** 
  - **Naive Bayes** 
  - **VADER** 
  - **TextBlob**

### 📁 Batch Review Analysis
- Upload a CSV or Excel file containing a column named `reviews.text`.
- The app will analyze all reviews in the file using the VADER model.
- View:
  - Sentiment distribution (pie chart and summary stats)
  - Color-coded table of all reviews and their predicted sentiment
  - Option to download the results as a CSV file


## Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/umerr1214/sentiment_analyzer.git
cd sentiment_analyzer
```

### 2. Install Requirements

```bash
pip install -r requirements.txt
```

### 3. Run the App

```bash
streamlit run app.py
```

---

## Usage

### Single Review
- Enter your review in the text box.
- Select which model(s) to use from the sidebar.
- Click **Analyze Sentiment** to see predictions, explanations, and visualizations.

### Batch Analysis
- Go to the sidebar and upload a `.csv` or `.xlsx` file with a column named `reviews.text`.
- Click **Analyze All Reviews (VADER)**.
- View the results, download the output, and explore the sentiment distribution.

---

## File Format for Batch Analysis

- The file must have a column named `reviews.text`.
- Supported formats: `.csv`, `.xlsx`, `.xls`
- Example:

| reviews.text                        |
|-------------------------------------|
| This product is amazing!            |
| Not worth the price.                |
| It's okay, does the job.            |

---

## Requirements

- Python 3.7+
- Streamlit
- pandas
- scikit-learn
- nltk
- vaderSentiment
- textblob
- plotly
- openpyxl

(Install all dependencies with `pip install -r requirements.txt`)


---

**Enjoy analyzing your product reviews with Sentiment Studio!**   
