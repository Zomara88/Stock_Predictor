import requests
import praw
from textblob import TextBlob
import pandas as pd
from dotenv import load_dotenv
import os
from datetime import datetime

load_dotenv()

# Initialize PRAW with your Reddit API credentials
reddit = praw.Reddit(client_id=os.getenv('REDDIT_CLIENT_ID'),
                     client_secret=os.getenv('REDDIT_CLIENT_SECRET'),
                     user_agent=os.getenv('REDDIT_USER_AGENT'))

def fetch_reddit_data(stock_ticker):
    subreddit = reddit.subreddit('all')
    comments = []
    
    for submission in subreddit.search(stock_ticker, limit=100):
        for comment in submission.comments:
            if isinstance(comment, praw.models.Comment):
                date = datetime.fromtimestamp(comment.created_utc)
                comments.append({'text': comment.body, 'date': date})
    
    return comments

def fetch_news_data(stock_ticker, api_key):
    url = f"https://newsapi.org/v2/everything?q={stock_ticker}&apiKey={api_key}"
    response = requests.get(url)
    news_data = []
    
    if response.status_code == 200:
        articles = response.json().get('articles', [])
        for article in articles:
            title = article['title'] or ""
            description = article['description'] or ""
            text = title + " " + description
            date = article.get('publishedAt')
            news_data.append({'text': text, 'date': date})
    
    return news_data

from textblob import TextBlob

def analyze_sentiment(data):
    for item in data:
        text = item['text']  # Extract the text field
        blob = TextBlob(text)
        sentiment = blob.sentiment.polarity
        item['sentiment'] = sentiment
    return data

import pandas as pd

def combine_data(stock_ticker, news_api_key):
    # Fetch data from different sources
    reddit_data = fetch_reddit_data(stock_ticker)
    news_data = fetch_news_data(stock_ticker, news_api_key)
    
    # Combine data from all sources and include date
    all_data = reddit_data + news_data

    # Analyze sentiment and create a DataFrame with text, sentiment, and date
    data = []
    for item in all_data:
        text = item['text']
        date = item.get('date')  # Extract date if available
        sentiment = TextBlob(text).sentiment.polarity
        data.append({'Text': text, 'Sentiment': sentiment, 'Date': date})

    # Create DataFrame and check the output
    df = pd.DataFrame(data)
    print(df[['Text', 'Sentiment', 'Date']].head())  # Print sample rows with Date column to verify

    return df

# Example: Use the functions
news_api_key = 'fe2cb004c56947e2b0a25259cef6ac44'

stock_ticker = 'NVDA'
combined_data = combine_data(stock_ticker, news_api_key)

print(combined_data)
