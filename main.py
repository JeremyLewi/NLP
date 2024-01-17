import pandas as pd
import streamlit as st
import subprocess
import os
import glob

# Menambahkan judul aplikasi
st.title('Sentiment Analysis')

# Tweet Crawling Integration
if st.button('Crawl Tweets'):
    search_keyword = 'Debat Pemilihan Presiden 2024 until:2024-01-13 since:2023-12-01'
    limit = 10
    token = "ae32da70061dd43aa6371090cdc45d010cad878d"

    # Running the tweet-harvest command
    subprocess.run(['npx', '--yes', 'tweet-harvest@2.2.8', '-s', search_keyword, '-l', str(limit), '--token', token])

    # Assuming the file is saved in a known directory, adjust the path as needed
    file_directory = '/Users/jeremylewi/Downloads/textblob_sentiment_analysis-main 2/tweets-data/'
    files = glob.glob(file_directory + '/*.csv')

    # Check if any new CSV file is created
    if files:
        latest_file = max(files, key=os.path.getctime)
        # Read the latest file
        crawled_df = pd.read_csv(latest_file, delimiter=";")
        
        # Display the result
        st.write("Crawled Data:")
        st.dataframe(crawled_df.head(10))  # Adjust the number of rows as needed
        st.success('Tweet crawling completed!')
    else:
        st.error('No new tweet data file found.')
