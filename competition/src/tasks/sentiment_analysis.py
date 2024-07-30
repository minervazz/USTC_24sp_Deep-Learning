# src/tasks/sentiment_analysis.py

from transformers import pipeline

def analyze_sentiment(text, model_name="distilbert-base-uncased-finetuned-sst-2-english"):
    sentiment_analyzer = pipeline("sentiment-analysis", model=model_name)
    result = sentiment_analyzer(text)
    return result[0]
