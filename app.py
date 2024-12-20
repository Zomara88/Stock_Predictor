from flask import Flask, jsonify, request, render_template, send_from_directory, redirect, url_for, session
from flask_cors import CORS
import yfinance as yf
import praw
from textblob import TextBlob
import pandas as pd
import requests
from dotenv import load_dotenv
import os
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import traceback
import hashlib
import firebase_admin
from firebase_admin import credentials, firestore
import re

# Load environment variables from .env file
load_dotenv()

# Initialize Flask app and enable CORS
app = Flask(__name__)
CORS(app)

# Set secret key for session management
app.secret_key = os.getenv('SECRET_KEY', 'supersecretkey')

# Initialize Firebase Admin SDK
firebase_cert_path = os.getenv('FIREBASE_CERT_PATH')
if not os.path.exists(firebase_cert_path):
    print(f"Error: Firebase certificate file not found: {firebase_cert_path}")
    exit(1)
cred = credentials.Certificate(firebase_cert_path)
firebase_admin.initialize_app(cred)

# Initialize Firestore
db = firestore.client()


@app.route('/register', methods=['POST'])
def register():
    """Register a new user."""
    try:
        data = request.form
        username = data.get('username')
        email = data.get('email')
        password = data.get('password')

        # Validate input
        if not username or len(username) < 3 or len(username) > 10:
            return jsonify({'error': 'Username must be between 3 and 10 characters'}), 400
        if not email or not re.match(r"[^@]+@[^@]+\.[^@]+", email):
            return jsonify({'error': 'Invalid email format'}), 400
        if not password or len(password) < 8 or len(password) > 100:
            return jsonify({'error': 'Password must be between 8 and 100 characters'}), 400

        # Hash the password before storing it
        hashed_password = hashlib.sha256(password.encode()).hexdigest()

        # Store user data in Firestore
        users_ref = db.collection('users')
        user_doc = users_ref.document(username)
        user_doc.set({
            'username': username,
            'email': email,
            'password': hashed_password
        })

        # Set session user_id
        session['user_id'] = username
        return redirect(url_for('login_page'))
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/login', methods=['POST'])
def login():
    """Log in an existing user."""
    try:
        data = request.form
        username = data.get('username')
        password = data.get('password')

        # Validate input
        if not username or not password:
            return jsonify({'error': 'Username and password are required'}), 400

        # Hash the password to compare with stored password
        hashed_password = hashlib.sha256(password.encode()).hexdigest()

        # Check if user exists in Firestore
        users_ref = db.collection('users')
        user_doc = users_ref.document(username).get()

        if user_doc.exists:
            user_data = user_doc.to_dict()
            if user_data['password'] == hashed_password:
                session['user_id'] = username
                return render_template('index.html')
            else:
                return jsonify({'message': 'Invalid credentials!'}), 401
        else:
            return jsonify({'message': 'User not found!'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/logout', methods=['GET', 'POST'])
def logout():
    """Log out the current user."""
    session.pop('user_id', None)
    return redirect(url_for('login_page'))

# Initialize PRAW with your Reddit API credentials
reddit = praw.Reddit(client_id=os.getenv('REDDIT_CLIENT_ID'),
                     client_secret=os.getenv('REDDIT_CLIENT_SECRET'),
                     user_agent=os.getenv('REDDIT_USER_AGENT'))

def fetch_reddit_data(stock_ticker):
    """Fetch comments from Reddit related to the stock ticker."""
    subreddit = reddit.subreddit('all')
    comments = []
    
    try:
        for submission in subreddit.search(stock_ticker, limit=5):
            submission.comments.replace_more(limit=0)
            for comment in submission.comments.list()[:5]:
                if isinstance(comment, praw.models.Comment):
                    date = datetime.fromtimestamp(comment.created_utc)
                    comments.append({'text': comment.body, 'date': date})
    except Exception as e:
        print(f"Error fetching Reddit data: {e}")

    return comments

def fetch_news_data(stock_ticker, api_key):
    """Fetch news articles related to the stock ticker."""
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
            if date:
                date = pd.to_datetime(date).tz_localize(None)
            news_data.append({'text': text, 'date': date})
    
    return news_data

def analyze_sentiment(data):
    """Analyze sentiment of the given text data."""
    for item in data:
        text = item['text']
        blob = TextBlob(text)
        sentiment = blob.sentiment.polarity
        item['sentiment'] = sentiment
    return data

def combine_data(stock_ticker, news_api_key):
    """Combine Reddit and news data for the stock ticker."""
    reddit_data = fetch_reddit_data(stock_ticker)
    news_data = fetch_news_data(stock_ticker, news_api_key)

    all_data = reddit_data + news_data

    data = []
    for item in all_data:
        text = item['text']
        date = item.get('date')
        if date:
            date = pd.to_datetime(date).tz_localize(None)
        sentiment = TextBlob(text).sentiment.polarity
        data.append({'Text': text, 'Sentiment': sentiment, 'Date': date})

    df = pd.DataFrame(data)
    return df

def create_features(historical_data, combined_data):
    """Create features for the model from historical and combined data."""
    historical_data['MA_5'] = historical_data['Close'].rolling(window=5).mean()
    historical_data['MA_10'] = historical_data['Close'].rolling(window=10).mean()

    combined_data['Date'] = pd.to_datetime(combined_data['Date']).dt.tz_localize(None)
    historical_data.reset_index(inplace=True)
    historical_data['Date'] = pd.to_datetime(historical_data['Date']).dt.tz_localize(None)

    merged_data = pd.merge(historical_data, combined_data, left_on='Date', right_on='Date', how='left')
    merged_data.fillna(0, inplace=True)

    return merged_data

def train_model(data):
    """Train a RandomForest model on the given data."""
    data['Target'] = data['Close'].shift(-1)
    data.dropna(inplace=True)

    features = ['MA_5', 'MA_10', 'Sentiment']
    X = data[features]
    y = data['Target']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    return model

def determine_risk_level(prediction):
    """Determine the risk level based on the prediction."""
    if prediction > 100:
        return "Low"
    elif prediction > 50:
        return "Moderate"
    else:
        return "High"

@app.route('/')
def home():
    """Redirect to the login page."""
    return redirect(url_for('login_page'))

@app.route('/news')
def news():
    """Render the news page."""
    return render_template('news.html')

@app.route('/news-data', methods=['GET'])
def news_data():
    """Fetch news data for the given stock ticker."""
    ticker = request.args.get('ticker', 'NVDA')
    news_api_key = os.getenv('NEWS_API_KEY')

    try:
        news_data = fetch_news_data(ticker, news_api_key)
        return jsonify(news_data)
    except Exception as e:
        print(f"Error fetching news data: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/predict', methods=['GET'])
def predict():
    """Predict the stock price for the given ticker."""
    ticker = request.args.get('ticker', 'NVDA')
    news_api_key = os.getenv('NEWS_API_KEY')

    try:
        company = yf.Ticker(ticker)
        historical_data = company.history(period="max")
        historical_data.reset_index(inplace=True)
        historical_data['Date'] = pd.to_datetime(historical_data['Date']).dt.tz_localize(None)

        historical_data = historical_data[['Date', 'Close']]

        combined_data = combine_data(ticker, news_api_key)

        data = create_features(historical_data, combined_data)

        model = train_model(data)

        latest_data = data.iloc[-1][['MA_5', 'MA_10', 'Sentiment']].values.reshape(1, -1)
        latest_data_df = pd.DataFrame(latest_data, columns=['MA_5', 'MA_10', 'Sentiment'])
        prediction = model.predict(latest_data_df)[0]

        risk_level = determine_risk_level(prediction)

        historical_data_json = historical_data.to_json(orient='records', date_format='iso')
        combined_data_json = combined_data.to_json(orient='records', date_format='iso')

        news_data = fetch_news_data(ticker, news_api_key)

        response = {
            'prediction': prediction,
            'risk_level': risk_level,
            'historical_data': historical_data_json,
            'combined_data': combined_data_json,
            'news_data': news_data
        }
        return jsonify(response)

    except Exception as e:
        print(f"Error during prediction: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/market-overview')
def market_overview():
    """Render the market overview page."""
    return render_template('market_overview.html')

@app.route('/market-performance', methods=['GET'])
def market_performance():
    """Fetch market performance data."""
    try:
        performance_data = {}  # Replace with actual data fetching logic
        return jsonify(performance_data)
    except Exception as e:
        print(f"Error fetching market performance data: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/trending-stocks', methods=['GET'])
def trending_stocks():
    """Fetch trending stocks data."""
    try:
        trending_data = {}  # Replace with actual data fetching logic
        return jsonify(trending_data)
    except Exception as e:
        print(f"Error fetching trending stocks data: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/trending-stocks-data', methods=['GET'])
def trending_stocks_data():
    """Fetch trending stocks data from yfinance."""
    try:
        tickers = ['AAPL', 'MSFT', 'GOOGL']
        companies = []
        dates = []

        for ticker in tickers:
            company = yf.Ticker(ticker)
            history = company.history(period="1mo")
            if history.empty:
                continue
            dates = history.index.strftime('%Y-%m-%d').tolist()
            prices = history['Close'].tolist()
            name = company.info.get('shortName', ticker)

            companies.append({
                'name': name,
                'prices': prices
            })

        data = {
            'dates': dates,
            'companies': companies
        }

        return jsonify(data)
    except Exception as e:
        print(f"Error fetching trending stocks data: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/sector-analysis', methods=['GET'])
def sector_analysis():
    """Fetch sector analysis data."""
    try:
        sector_data = {}  # Replace with actual data fetching logic
        return jsonify(sector_data)
    except Exception as e:
        print(f"Error fetching sector analysis data: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/sector-data', methods=['GET'])
def sector_data():
    """Fetch sector data for the given stock ticker."""
    ticker = request.args.get('ticker')
    try:
        company = yf.Ticker(ticker)
        latest_close = round(company.history(period="1d")['Close'].iloc[0], 2)
        eps = company.info.get('trailingEps', None)
        pe_ratio = round(latest_close / eps, 2) if eps else None
        market_cap = company.info.get('marketCap', 'N/A')
        sector = company.info.get('sector', 'N/A')
        analyst_rating = company.info.get('recommendationMean', 'N/A')

        data = {
            'market_cap': market_cap,
            'latest_close': latest_close,
            'pe_ratio': pe_ratio,
            'sector': sector,
            'analyst_rating': analyst_rating
        }
        return jsonify(data)
    except Exception as e:
        print(f"Error fetching sector data for {ticker}: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/sorted-companies', methods=['GET'])
def sorted_companies():
    """Fetch and sort companies data based on the given criteria."""
    sort_by = request.args.get('sort_by')
    order = request.args.get('order', 'asc')
    try:
        tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA']
        companies = []

        for ticker in tickers:
            try:
                company = yf.Ticker(ticker)
                history = company.history(period="1d")
                if history.empty:
                    continue
                latest_close = round(history['Close'].iloc[0], 2)
                eps = company.info.get('trailingEps', None)
                pe_ratio = round(latest_close / eps, 2) if eps else None
                market_cap = company.info.get('marketCap', 'N/A')
                name = company.info.get('shortName', ticker)

                companies.append({
                    'name': name,
                    'market_cap': market_cap,
                    'pe_ratio': pe_ratio,
                    'latest_close': latest_close
                })
            except Exception as e:
                print(f"Error fetching data for {ticker}: {e}")

        companies.sort(key=lambda x: x[sort_by], reverse=(order == 'desc'))

        return jsonify(companies)
    except Exception as e:
        print(f"Error fetching sorted companies data: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/static/<path:filename>')
def static_files(filename):
    """Serve static files."""
    return send_from_directory(os.path.join(app.root_path, 'static'), filename)

@app.route('/login', methods=['GET'])
def login_page():
    """Render the login page."""
    return render_template('login.html')

@app.route('/register', methods=['GET'])
def register_page():
    """Render the registration page."""
    return render_template('register.html')

@app.route('/index')
def index():
    """Render the index page."""
    return render_template('index.html')

@app.route('/users', methods=['GET'])
def get_users():
    """Fetch all users from Firestore."""
    try:
        users_ref = db.collection('users')
        users = users_ref.stream()
        users_data = [{'username': user.id, 'email': user.to_dict().get('email')} for user in users]
        return jsonify(users_data), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/check-db', methods=['GET'])
def check_db():
    """Check the database connection."""
    try:
        # Perform a simple query to check the database connection
        users_ref = db.collection('users')
        users = users_ref.stream()
        return jsonify({'message': 'Database connection successful', 'result': [user.id for user in users]}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)