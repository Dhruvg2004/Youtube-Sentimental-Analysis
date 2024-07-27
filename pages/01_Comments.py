import streamlit as st
from joblib import load
import re

svm_classifier, tfidf_vectorizer = load('sent1.joblib')

def analyze_single_comment(comment):
    """
    Analyzes the sentiment of a single comment using the loaded SVM model.
    
    Parameters:
    comment (str): The comment to analyze.
    
    Returns:
    str: The predicted sentiment ('positive', 'negative', 'neutral').
    """
    # Check if comment is empty or contains only special characters
    if not comment.strip() or re.match(r'^[\W_]+$', comment) or comment.strip().isdigit():
        return 'invalid'
    
    # Transform the input text using the loaded vectorizer
    text_transformed = tfidf_vectorizer.transform([comment])
    
    # Predict the sentiment using the loaded model
    sentiment = svm_classifier.predict(text_transformed)[0]
    
    return sentiment 

st.title("Sentiment Analysis")
comment = st.text_input("Enter any comment")
b = st.button("Analyze")

if "select" not in st.session_state:
    st.session_state["select"] = False

if not st.session_state["select"]:
    if b:
        output = analyze_single_comment(comment)
        if output == 'positive':
            st.success("Positive")
        elif output == 'negative':
            st.error("Negative")
        elif output == 'neutral':
            st.warning("Neutral")
        else:
            st.error("Invalid input: Please enter a valid comment")
else:
    st.error("Please enter a valid video URL")
