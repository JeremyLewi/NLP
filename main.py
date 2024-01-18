import pandas as pd
import streamlit as st
import subprocess
import os
import glob
from textblob import TextBlob
import matplotlib.pyplot as plt
import cleantext
from googletrans import Translator
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, recall_score, precision_score, f1_score


# Fungsi untuk melakukan pembersihan teks
def clean_text(text):
    return cleantext.clean(text, clean_all=True, extra_spaces=True, stopwords=True, lowercase=True, numbers=True, punct=True)

# Fungsi untuk menerjemahkan teks ke bahasa Inggris
def translate_text_to_english(text):
    try:
        translated = translator.translate(text, src='id', dest='en')
        return translated.text
    except Exception as e:
        print("Error:", e)
        return text

# Fungsi untuk menghitung skor polaritas
def score(text):
    blob = TextBlob(text)
    return blob.sentiment.polarity

# Fungsi untuk menghitung sentimen
def analyze_sentiment(polarity):
    return 'Positive' if polarity > 0 else ('Negative' if polarity < 0 else 'Neutral')

# Inisialisasi Translator
translator = Translator()

# Menambahkan judul aplikasi
st.title('Advanced Twitter Sentiment Analysis and Evaluation Dashboard')

# Adding User Instructions
st.markdown("""
    ### Instructions for Use:
    1. **Tweet Crawling and Analysis Options**: Enter your search keyword, start and end dates, and limit for tweet crawling.
    2. **Crawl Data**: Click the 'Crawl Data' button to fetch tweets based on your search criteria.
    3. **Upload a CSV File for Analysis**: Optionally, upload a CSV file containing tweets for analysis.
    4. **Perform Sentiment Analysis**: Click the 'Sentiment Analysis' button to analyze the sentiment of crawled or uploaded tweets.
    5. **Sentiment Analysis Model Evaluation**: Upload another CSV file with actual sentiments for model evaluation.
    6. **Evaluate Model**: After uploading, click the 'Evaluate Model' button to see the performance metrics and comparison.
""")

# Sidebar Options
st.sidebar.header("Tweet Crawling and Analysis Options")
search_keyword = st.sidebar.text_input("Enter search keyword:", 'Debat Pemilihan Presiden 2024')
start_date = st.sidebar.date_input("Enter start date:", pd.to_datetime('2023-12-01'))
end_date = st.sidebar.date_input("Enter end date:", pd.to_datetime('2024-01-13'))
limit = st.sidebar.number_input("Enter limit:", 5, 1000, 100, 100)




# Mendapatkan direktori saat ini
current_directory = os.getcwd()

# Subdirektori yang ingin ditambahkan
subdirectory = 'tweets-data'

# Membuat direktori lengkap dengan subdirektori
file_directory = os.path.join(current_directory, subdirectory)

# Tombol untuk Crawling Data
if st.sidebar.button('Crawl Data'):
    token = "ae32da70061dd43aa6371090cdc45d010cad878d"
    subprocess.run(['npx', '--yes', 'tweet-harvest@2.2.8', '-s', search_keyword, '-l', str(limit), '--token', token])

    files = glob.glob(file_directory + '/*.csv')
    if files:
        latest_file = max(files, key=os.path.getctime)
        crawled_df = pd.read_csv(latest_file, delimiter=";")
        st.session_state.crawled_df = crawled_df
        st.write("Crawled Data:")
        st.dataframe(crawled_df)
        st.success('Tweet crawling completed!')
    else:
        st.error('No new tweet data file found.')

# File Uploader
uploaded_file = st.sidebar.file_uploader("Upload a CSV file for analysis:", type=['csv'])
if uploaded_file is not None:
    uploaded_df = pd.read_csv(uploaded_file, delimiter=";")
    st.session_state.uploaded_df = uploaded_df
    st.write("Uploaded Data for Analysis:")
    st.dataframe(uploaded_df)

# Tombol untuk Analisis Sentimen
if st.sidebar.button('Sentiment Analysis'):
    data_df = None
    if 'crawled_df' in st.session_state:
        data_df = st.session_state.crawled_df
    elif 'uploaded_df' in st.session_state:
        data_df = st.session_state.uploaded_df

    if data_df is not None:
        data_df['cleaned_text'] = data_df['full_text'].apply(clean_text)
        data_df['translated_text'] = data_df['cleaned_text'].apply(translate_text_to_english)
        data_df['polarity'] = data_df['translated_text'].apply(score)
        data_df['sentiment'] = data_df['polarity'].apply(analyze_sentiment)

        st.write("Data with Sentiment Analysis:")
        st.dataframe(data_df[['full_text', 'translated_text', 'sentiment', 'polarity']])
        sentiment_counts = data_df['sentiment'].value_counts()
        fig, ax = plt.subplots()
        sentiment_counts.plot(ax=ax, kind='bar')
        st.pyplot(fig)
        st.success('Sentiment analysis completed!')
    else:
        st.warning('Please crawl or upload data first.')

# subheader "Sentiment Analysis Model Evaluation"
st.sidebar.subheader("Sentiment Analysis Model Evaluation")


# File Uploader for Model Evaluation
evaluation_file = st.sidebar.file_uploader("Upload a CSV file for model evaluation (with actual sentiments):", type=['csv'])
if evaluation_file is not None:
    evaluation_df = pd.read_csv(evaluation_file,  delimiter=";")
    st.session_state.evaluation_df = evaluation_df
    st.write("Uploaded Data for Evaluation:")
    st.dataframe(evaluation_df)


# Button for Model Evaluation
if st.sidebar.button('Evaluate Model'):
    if 'evaluation_df' in st.session_state:
        eval_df = st.session_state.evaluation_df
        if 'actual_sentiment' in eval_df.columns and 'full_text' in eval_df.columns:
            eval_df['predicted_sentiment'] = eval_df['full_text'].apply(lambda x: analyze_sentiment(score(translate_text_to_english(clean_text(x)))))
            
            # Calculating metrics
            accuracy = accuracy_score(eval_df['actual_sentiment'], eval_df['predicted_sentiment'])
            precision = precision_score(eval_df['actual_sentiment'], eval_df['predicted_sentiment'], average='weighted')
            recall = recall_score(eval_df['actual_sentiment'], eval_df['predicted_sentiment'], average='weighted')
            f1 = f1_score(eval_df['actual_sentiment'], eval_df['predicted_sentiment'], average='weighted')


            # Display metrics
            st.write("Evaluation Metrics:")
            st.write("Accuracy: ", accuracy)
            st.write("Precision: ", precision)
            st.write("Recall: ", recall)
            st.write("F1 Score: ", f1)


            # Confusion Matrix
            cm = confusion_matrix(eval_df['actual_sentiment'], eval_df['predicted_sentiment'])
            st.write("Confusion Matrix:")
            st.dataframe(pd.DataFrame(cm))

            # Classification Report
            st.write("Classification Report:")
            st.text(classification_report(eval_df['actual_sentiment'], eval_df['predicted_sentiment']))
        else:
            st.error("The uploaded file for evaluation must contain 'actual_sentiment' and 'full_text' columns.")
    else:
        st.warning("Please upload a file for model evaluation.")


# Final Layout Adjustments
st.sidebar.markdown("---")
st.sidebar.write("Developed by KIYOWO Team") 
st.sidebar.write("© 2024")