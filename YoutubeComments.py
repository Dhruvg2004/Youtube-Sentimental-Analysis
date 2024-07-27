import streamlit as st
from joblib import load
import re
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from googleapiclient.discovery import build
from langdetect import detect
import pandas as pd

api_key = ''  

# Load SVM classifier and TF-IDF vectorizer from joblib file
svm_classifier, tfidf_vectorizer = load('sent1.joblib')

# Function to preprocess text data
def preprocess_text(text):
    """
    Preprocesses text data by removing punctuation, converting to lowercase,
    removing stopwords, and tokenizing.
    
    Parameters:
    text (str): The input text to preprocess.
    
    Returns:
    str: Preprocessed text.
    """
    text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
    text = text.lower()  # Convert to lowercase
    tokens = word_tokenize(text)  # Tokenize the text
    stop_words = set(stopwords.words('english'))  # Get English stopwords
    tokens = [word for word in tokens if word not in stop_words]  # Remove stopwords
    text = ' '.join(tokens)  # Join tokens back into text
    return text

# Function to analyze sentiment of a single comment
def analyze_sentiment(text):
    """
    Analyzes sentiment of a single comment using loaded SVM model and TF-IDF vectorizer.
    
    Parameters:
    text (str): The comment text to analyze.
    
    Returns:
    str: Predicted sentiment ('positive', 'negative', 'neutral').
    """
    text = preprocess_text(text)  # Preprocess the text
    text_transformed = tfidf_vectorizer.transform([text])  # Transform text using TF-IDF vectorizer
    sentiment = svm_classifier.predict(text_transformed)[0]  # Predict sentiment
    return sentiment

# Function to extract YouTube video comments
def extract_youtube_video_comments(video_id):
    """
    Extracts English comments from a YouTube video using its video ID.
    
    Parameters:
    video_id (str): The YouTube video ID.
    
    Returns:
    list: List of English comments from the YouTube video.
    """
    youtube = build('youtube', 'v3', developerKey=api_key)  # Initialize YouTube API client
    video_response = youtube.commentThreads().list(
        part='snippet,replies',
        videoId=video_id
    ).execute()  # Execute API request to get video comments
    
    comments = []
    while len(comments) <= 100:  # Limit to 100 comments
        for item in video_response['items']:
            comment = item['snippet']['topLevelComment']['snippet']['textDisplay']
            try:
                if detect(comment) == 'en':  # Check if comment is in English
                    comments.append(comment)
            except:
                pass
        
        if 'nextPageToken' in video_response:
            video_response = youtube.commentThreads().list(
                part='snippet,replies',
                videoId=video_id,
                pageToken=video_response['nextPageToken']
            ).execute()  # Get next page of comments if available
        else:
            break
    
    return comments

# Function to analyze sentiment of YouTube video comments
def analyze_yt_comments(video_id):
    """
    Analyzes sentiment of comments from a YouTube video and displays overall sentiment and counts.
    
    Parameters:
    video_id (str): The YouTube video ID.
    """
    comments = extract_youtube_video_comments(video_id)  # Extract comments from YouTube video
    sentiments = [analyze_sentiment(comment) for comment in comments]  # Analyze sentiment of each comment
    sentiment_counts = pd.Series(sentiments).value_counts()  # Count occurrences of each sentiment
    
    # Determine overall sentiment based on sentiment counts
    if 'positive' in sentiment_counts and sentiment_counts['positive'] > sentiment_counts.get('negative', 0) and sentiment_counts['positive'] > sentiment_counts.get('neutral', 0):
        st.success("Overall Sentiment: Positive")
    elif 'negative' in sentiment_counts and sentiment_counts['negative'] > sentiment_counts.get('positive', 0) and sentiment_counts['negative'] > sentiment_counts.get('neutral', 0):
        st.error("Overall Sentiment: Negative")
    else:
        st.warning("Overall Sentiment: Neutral")
    
    st.write("Sentiment counts:")  # Display sentiment counts
    st.write(sentiment_counts)

# Function to extract YouTube video ID from URL
def extract_video_id(youtube_link):
    """
    Extracts YouTube video ID from a YouTube video URL.
    
    Parameters:
    youtube_link (str): The YouTube video URL.
    
    Returns:
    str: YouTube video ID extracted from URL.
    """
    pattern = r'(?:https?://)?(?:www\.)?(?:youtube\.com/watch\?v=|youtu\.be/)([a-zA-Z0-9_-]{11})'
    match = re.search(pattern, youtube_link)
    if match:
        return match.group(1)  # Return video ID if found
    else:
        return None  

# Streamlit app title and user input fields
st.title("YouTube Comments Sentiment Analysis")
url = st.text_input("Enter YouTube video URL")  
analyze_button = st.button("Analyze")  

if "select" not in st.session_state:
    st.session_state["select"] = False  # Initialize session state variable

if analyze_button:  
    video_id = extract_video_id(url)  # Extract video ID from URL
    if video_id:
        analyze_yt_comments(video_id)  # Analyze YouTube video comments if valid video ID found
    else:
        st.error("Please enter a valid YouTube video URL")  
