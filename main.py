import string
from textblob import TextBlob
import pandas as pd
import streamlit as st
import cleantext
import os
from googletrans import Translator
import matplotlib.pyplot as plt

st.header('Sentiment Analysis')

def score(x):
    blob1 = TextBlob(x)
    return blob1.sentiment.polarity

def analyze(x):
    if x >= 0.5:
        return 'Positive'
    elif x <= -0.5:
        return 'Negative'
    else:
        return 'Neutral'

if 'crawled_df' not in st.session_state:
    st.session_state.crawled_df = pd.DataFrame()

st.sidebar.header("Tweet Crawling Options")
search_keyword = st.sidebar.text_input("Enter search keyword:", 'Debat Pemilihan Presiden 2024')
start_date = st.sidebar.date_input("Enter start date:", pd.to_datetime('2023-12-01'))
end_date = st.sidebar.date_input("Enter end date:", pd.to_datetime('2024-01-13'))
limit = st.sidebar.number_input("Enter limit:", 10, 1000, 100, 100)







if st.sidebar.button('Process'):
    filename = 'Pilpres.csv'
    search_query_dates = f'until:{end_date.strftime("%Y-%m-%d")} since:{start_date.strftime("%Y-%m-%d")}'
    current_directory = os.getcwd()
    tweets_data_directory = os.path.join(current_directory, 'tweets-data')
    os.makedirs(tweets_data_directory, exist_ok=True)
    file_path = os.path.join(tweets_data_directory, filename)

    os.system(f'npx --yes tweet-harvest@2.2.8 -o "{file_path}" -s "{search_keyword} {search_query_dates}" -l {limit} --token "ae32da70061dd43aa6371090cdc45d010cad878d"')

    st.success('Tweet crawling completed!')
    # thiyara tambahan
    # st.session_state.crawled_df = pd.read_csv(file_path)
    # st.write("Crawled Data:")
    # st.write(st.session_state.crawled_df.head(10))


    st.sidebar.download_button(
            label="Download Crawled Data",
            key="download_button",
            data=st.session_state.crawled_df.to_csv(index=False).encode(),
            file_name=filename,
            mime="text/csv"
        )
# hiyara tambahan
    


# ## Pra-pemrosesan (Preprocessing):
    
#     # Clean the tweets
#     st.session_state.crawled_df['full_text'] = st.session_state.crawled_df['full_text'].apply(lambda x: cleantext.clean(x, clean_all=True, extra_spaces=True, stopwords=True, lowercase=True, numbers=True, punct=True))

#     st.write("Cleaned Data:")

#     st.write(st.session_state.crawled_df[['full_text']].head(10))  # Displaying the first 10 rows, adjust as needed


# ## Analisis Sentimen
#     # Check if tweets have been crawled before analyzing
#     if st.session_state.crawled_df.empty:
#         st.warning("Please crawl tweets before analyzing.")
#     else:
#         st.write("Translating tweets to English...")
    
#       # Translate tweets to English
#         translator = Translator()
#         st.session_state.crawled_df['translated_text'] = st.session_state.crawled_df['full_text'].apply(lambda x: translator.translate(x).text)

#         st.write("Performing sentiment analysis...")

#         # Perform sentiment analysis on the translated text
#         st.session_state.crawled_df['score'] = st.session_state.crawled_df['translated_text'].apply(score)
#         st.session_state.crawled_df['analysis'] = st.session_state.crawled_df['score'].apply(analyze)

#         st.write("Analyzed Data:")
#         st.write(st.session_state.crawled_df[['full_text', 'score', 'analysis']].head(10))  # Displaying the first 10 rows, adjust as needed


# # Ekstraksi Fitur (Feature Extraction):
# if st.button('Extract Features'):
#     # Check if tweets have been crawled before extracting features
#     if st.session_state.crawled_df.empty:
#         st.warning("Please crawl tweets before extracting features.")
#     else:
#         st.write("Extracting features...")

#         # Extract features from the tweets
#         st.session_state.crawled_df['word_count'] = st.session_state.crawled_df['full_text'].apply(lambda x: len(str(x).split(" ")))
#         st.session_state.crawled_df['char_count'] = st.session_state.crawled_df['full_text'].str.len()
#         st.session_state.crawled_df['word_density'] = st.session_state.crawled_df['char_count'] / (st.session_state.crawled_df['word_count'] + 1)
#         st.session_state.crawled_df['punctuation_count'] = st.session_state.crawled_df['full_text'].apply(lambda x: len([c for c in str(x) if c in string.punctuation]))
#         st.session_state.crawled_df['title_word_count'] = st.session_state.crawled_df['full_text'].apply(lambda x: len([wrd for wrd in str(x).split() if wrd.istitle()]))
#         st.session_state.crawled_df['upper_case_word_count'] = st.session_state.crawled_df['full_text'].apply(lambda x: len([wrd for wrd in str(x).split() if wrd.isupper()]))

#         st.write("Extracted Data:")
#         st.write(st.session_state.crawled_df[['full_text', 'word_count', 'char_count', 'word_density', 'punctuation_count', 'title_word_count', 'upper_case_word_count']].head(10))  # Displaying the first 10 rows, adjust as needed


    
# # Pemilihan Model:
# if st.button('Select Model'):
#     # Check if tweets have been crawled before selecting model
#     if st.session_state.crawled_df.empty:
#         st.warning("Please crawl tweets before selecting model.")
#     else:
#         st.write("Selecting model...")

#         # Select model
#         st.session_state.crawled_df['model'] = st.session_state.crawled_df['score'].apply(lambda x: 'model1' if x >= 0.5 else ('model2' if x <= -0.5 else 'model3'))

#         st.write("Selected Model:")
#         st.write(st.session_state.crawled_df[['full_text', 'score', 'model']].head(10))  # Displaying the first 10 rows, adjust as needed


# # Evaluasi Model:
# if st.button('Evaluate Model'):
#     # Check if tweets have been crawled before evaluating model
#     if st.session_state.crawled_df.empty:
#         st.warning("Please crawl tweets before evaluating model.")
#     else:
#         st.write("Evaluating model...")

#         # Evaluate model
#         st.session_state.crawled_df['evaluation'] = st.session_state.crawled_df['score'].apply(lambda x: 'correct' if x >= 0.5 else ('incorrect' if x <= -0.5 else 'neutral'))

#         st.write("Evaluated Model:")
#         st.write(st.session_state.crawled_df[['full_text', 'score', 'model', 'evaluation']].head(10))  # Displaying the first 10 rows, adjust as needed




        


# # Disable the warning about Matplotlib's global figure object
# st.set_option('deprecation.showPyplotGlobalUse', False)

# # Assuming 'crawled_df' is your DataFrame containing the data
# if st.button('Visualize Data'):
#     # Check if tweets have been crawled before visualizing data
#     if st.session_state.crawled_df.empty:
#         st.warning("Please crawl tweets before visualizing data.")
#     else:
#         st.write("Visualizing data...")

#         # Create a figure and axis
#         fig, ax = plt.subplots()

#         # Visualize data
#         st.session_state.crawled_df['model'].value_counts().plot(kind='bar', ax=ax)

#         # Display the plot
#         st.pyplot(fig)

        

