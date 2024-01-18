# Advanced Twitter Sentiment Analysis and Evaluation Dashboard

## Project Overview

This project presents an advanced dashboard for Twitter Sentiment Analysis and Evaluation. It utilizes Python libraries like Pandas, Streamlit, TextBlob, CleanText, and Google Translate. The dashboard allows users to crawl Twitter data, perform sentiment analysis, and evaluate the sentiment analysis model using actual sentiments.

## Features

1. **Twitter Data Crawling**: Enter search keywords and parameters to fetch relevant tweets.
2. **Sentiment Analysis**: Analyze the sentiment of tweets (positive, negative, neutral) using TextBlob after cleaning and translating the text.
3. **Data Visualization**: Visualize the distribution of sentiments among the analyzed tweets.
4. **Model Evaluation**: Upload a dataset with actual sentiments to evaluate the performance of the sentiment analysis model.
5. **Performance Metrics**: Display accuracy, precision, recall, and F1 score along with a confusion matrix and classification report.

## Setup and Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/JeremyLewi/NLP.git
   ```
2. Install required Python packages:
   ```bash
   pip install pandas streamlit textblob clean-text googletrans sklearn matplotlib
   ```
3. Run the Streamlit app:
   ```bash
   streamlit run main.py
   ```

## Usage

- Tweet Crawling and Analysis Options: Set parameters for crawling Twitter data.
- Crawl Data: Fetch tweets based on search criteria.
  Upload CSV for Analysis: Optionally, upload a CSV containing tweets.
- Sentiment Analysis: Analyze the sentiment of crawled or uploaded tweets.
- Model Evaluation: Upload a CSV with actual sentiments to evaluate the model.
