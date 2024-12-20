<h1>Problem Statement</h1>

<p>Investors encounter challenges in trading and investment decisions due to the complexity of modern financial markets. The vast data, rapid market shifts, and global events can obscure opportunities and increase risks. Consequently, investors seek advanced tools and strategies to navigate these challenges and make informed decisions.</p>

<h1>Prompt</h1>

<p>Develop a tool that predicts daily stock performance using historical market data, social media trends, and news events. Users should input a stock ticker to receive performance predictions and actionable insights. The deliverable can be a website, dashboard, or mobile app.</p>

<p>Teams can focus on data sources such as historical market data, social media trends, and news events. External data can be used if compliant with terms and conditions.</p>

<p>The tool should provide retail investors with informative insights to support decision-making. There is no expectation to cover all market factors.</p>

<h1>Features</h1>

<p>The flask-based web application provides several functionalities, including user registration, login, stock prediction, and market data analysis.</p>

<p>Imports Flask and Flask-CORS libraries for setting up the Flask web server and handling cross-origin requests. TextBlob, yfinance, and praw are used for stock data retrieval, Reddit API access, sentiment analysis, and data handling.</p>

<p>Firebase Admin SDK is initialized to manage Firestore, where user data is stored. A Firebase certificate file path is retrieved from environment variables, and the app exits if the certificate is missing.</p>

<h1>Stock Data and Sentiment Analysis:</h1>

<p>Reddit and News API: Functions like fetch_reddit_data and fetch_news_data fetch related comments and articles for a specified stock ticker, then analyze_sentiment uses TextBlob to calculate sentiment polarity.</p>

<p>Data Combination: combine_data merges data from Reddit and news sources.</p>

<p>Feature Creation: create_features generates moving averages for stock prices to use in machine learning models.</p>

<p>Prediction: train_model trains a RandomForest model, and the predict endpoint provides stock price predictions and risk levels based on these features.</p>

<h1>Screenshots</h1>

<p>Stock prediction page:</p>
<img src="images/screenshot1.png" alt="Project Screenshot">

<p>Stock news page:</p>
<img src="images/screenshot2.png" alt="Project Screenshot">

<p>Market overview page:</p>
<img src="images/screenshot3.png" alt="Project Screenshot">



 
