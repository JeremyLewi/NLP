import pandas as pd
import streamlit as st
import subprocess
import os
import glob
from textblob import TextBlob
import matplotlib.pyplot as plt
import cleantext
from googletrans import Translator



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
st.title('Sentiment Analysis of Tweets using TextBlob')

# Sidebar Options
st.sidebar.header("Tweet Crawling Options")
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

    # Mengganti file_directory dengan file_directory yang telah dibuat
    files = glob.glob(file_directory + '/*.csv')

    if files:
        latest_file = max(files, key=os.path.getctime)
        crawled_df = pd.read_csv(latest_file, delimiter=";")
        st.session_state.crawled_df = crawled_df  # Menyimpan DataFrame dalam state sesi
        st.write("Crawled Data:")
        st.dataframe(crawled_df[['full_text']])
        st.success('Tweet crawling completed!')
    else:
        st.error('No new tweet data file found.')

# Tombol untuk Analisis Sentimen
if st.sidebar.button('Sentiment Analysis'):
    if 'crawled_df' in st.session_state:
        crawled_df = st.session_state.crawled_df  # Mengambil DataFrame dari state sesi
        crawled_df['cleaned_text'] = crawled_df['full_text'].apply(clean_text)
        crawled_df['translated_text'] = crawled_df['cleaned_text'].apply(translate_text_to_english)
        crawled_df['polarity'] = crawled_df['translated_text'].apply(score)
        crawled_df['sentiment'] = crawled_df['polarity'].apply(analyze_sentiment)

        st.write("Crawled Data with Sentiment Analysis:")
        st.dataframe(crawled_df[['full_text', 'translated_text', 'sentiment', 'polarity']])
        sentiment_counts = crawled_df['sentiment'].value_counts()
        fig, ax = plt.subplots()
        sentiment_counts.plot(ax=ax, kind='bar')
        st.pyplot(fig)
        st.success('Sentiment analysis completed!')
    else:
        st.warning('Please crawl data first.')

