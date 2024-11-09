import requests
import os
from dotenv import load_dotenv
import praw

load_dotenv()

# Initialize PRAW with your Reddit API credentials
reddit = praw.Reddit(client_id=os.getenv('REDDIT_CLIENT_ID'),
                     client_secret=os.getenv('REDDIT_CLIENT_SECRET'),
                     user_agent=os.getenv('REDDIT_USER_AGENT'))

def fetch_reddit_data(stock_ticker):
    subreddit = reddit.subreddit('all')  # Or specify a specific subreddit
    comments = []
    
    for submission in subreddit.search(stock_ticker, limit=10):  # Limit to 10 submissions
        submission.comments.replace_more(limit=0)  # Avoids fetching "MoreComments" objects
        for comment in submission.comments.list():  # Flatten the comment tree
            if isinstance(comment, praw.models.Comment):
                comments.append(comment.body)
    
    return comments

def fetch_news_data(stock_ticker, api_key):
    url = f"https://newsapi.org/v2/everything?q={stock_ticker}&apiKey={api_key}"
    response = requests.get(url)
    
    if response.status_code == 200:
        articles = response.json().get('articles', [])
        news_data = []
        for article in articles:
            title = article.get('title', '')  # Get title or default to empty string
            description = article.get('description', '')  # Get description or default to empty string
            
            # Check if both title and description are not empty
            if title and description:
                news_data.append(f"{title} {description}")
            elif title:  # If only title is available
                news_data.append(title)
            elif description:  # If only description is available
                news_data.append(description)
        return news_data
    else:
        print(f"Failed to fetch News API data. Status code: {response.status_code}")
        return []

from textblob import TextBlob

def analyze_sentiment(text_data):
    sentiments = []
    for text in text_data:
        blob = TextBlob(text)
        sentiment = blob.sentiment.polarity  # Get polarity (-1 to 1)
        sentiments.append(sentiment)
    return sentiments

import pandas as pd

def combine_data(stock_ticker, news_api_key):
    # Fetch data from different sources
    reddit_data = fetch_reddit_data(stock_ticker)
    news_data = fetch_news_data(stock_ticker, news_api_key)
    
    # Combine data from all sources
    all_data = reddit_data + news_data
    
    # Perform sentiment analysis
    sentiments = analyze_sentiment(all_data)
    
    # Create DataFrame
    data = pd.DataFrame({
        'Text': all_data,
        'Sentiment': sentiments
    })
    
    return data

# Example: Use the functions
news_api_key = 'NEWS_API_KEY'

stock_ticker = 'AAPL'
combined_data = combine_data(stock_ticker, news_api_key)

print(combined_data)
