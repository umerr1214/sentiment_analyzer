# Sentiment Studio

**Sentiment Studio** is an  web application for analyzing the sentiment of product reviews using both classical machine learning models and pretrained sentiment analysis tools. 

---

## Features

### 📝 Single Review Analysis
- Enter a product review in a text box.
- Get sentiment predictions from four models:
  - **Logistic Regression** (trained on your data)
  - **Naive Bayes** (trained on your data)
  - **VADER** (pretrained, rule-based)
  - **TextBlob** (pretrained, rule-based)
- See a side-by-side comparison of model predictions.
- Get easy-to-understand explanations for each model’s decision, including model strengths, weaknesses, and performance stats.

### 📁 Batch Review Analysis
- Upload a CSV or Excel file containing a column named `reviews.text`.
- The app will analyze all reviews in the file using the VADER model.
- View:
  - Sentiment distribution (pie chart and summary stats)
  - Color-coded table of all reviews and their predicted sentiment
  - Option to download the results as a CSV file

### 📊 Visualizations & Explanations
- Interactive pie charts for sentiment distribution.
- Color-coded tables for easy review of results.
- Model agreement and disagreement explanations.
- Hover tooltips on charts show which models predicted each sentiment.

---

## Getting Started

### 1. Clone the Repository

```bash
git clone <your-repo-url>
cd <your-repo-directory>
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

## Screenshots

*(Add screenshots here if you like!)*

---

## License

MIT License

---

**Enjoy analyzing your product reviews with Sentiment Studio!**  
For questions or suggestions, feel free to open an issue or pull request. 