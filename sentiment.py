import requests
import praw
import pandas as pd
import os
import pickle
from datetime import datetime
from textblob import TextBlob
from dotenv import load_dotenv

load_dotenv('key.env')

# Initialize PRAW with your Reddit API credentials
reddit = praw.Reddit(client_id=os.getenv('REDDIT_CLIENT_ID'),
                     client_secret=os.getenv('REDDIT_CLIENT_SECRET'),
                     user_agent=os.getenv('REDDIT_USER_AGENT'))

def fetch_reddit_data(stock_ticker, cache_path):
    try:
        # Try loading cached data
        with open(cache_path, 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        # If no cache, fetch data
        subreddit = reddit.subreddit('all')
        comments = []
        
        for submission in subreddit.search(stock_ticker, limit=100):
            for comment in submission.comments:
                if isinstance(comment, praw.models.Comment):
                    date = datetime.fromtimestamp(comment.created_utc)
                    comments.append({'text': comment.body, 'date': date})
        
        # Cache data
        with open(cache_path, 'wb') as f:
            pickle.dump(comments, f)
        
        return comments

def fetch_news_data(stock_ticker, api_key, cache_path):
    try:
        with open(cache_path, 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
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
        
        with open(cache_path, 'wb') as f:
            pickle.dump(news_data, f)
        
        return news_data

def combine_data(stock_ticker, news_api_key):
    reddit_data = fetch_reddit_data(stock_ticker, f'{stock_ticker}_reddit_cache.pkl')
    news_data = fetch_news_data(stock_ticker, news_api_key, f'{stock_ticker}_news_cache.pkl')
    
    all_data = reddit_data + news_data
    data = []
    for item in all_data:
        text = item['text']
        date = item.get('date')
        sentiment = TextBlob(text).sentiment.polarity
        data.append({'Text': text, 'Sentiment': sentiment, 'Date': date})
    
    df = pd.DataFrame(data)
    print(df[['Text', 'Sentiment', 'Date']].head())
    return df

# Example: Use the functions
news_api_key = 'fe2cb004c56947e2b0a25259cef6ac44'
stock_ticker = 'NVDA'
combined_data = combine_data(stock_ticker, news_api_key)

print(combined_data)

