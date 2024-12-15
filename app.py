from flask import Flask, request, render_template
import datetime
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
from bs4 import BeautifulSoup
import pandas_market_calendars as mcal
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        user_date = request.form['date']
        # Call your prediction function here with user_date
        predictions = perform_predictions(user_date)
        return render_template('index.html', predictions=predictions)
    return render_template('index.html')

# Helper function: Get next business days
def get_next_business_days(start_date, num_days=5):
    """
    Get the next `num_days` business days starting from `start_date`.
    Excludes weekends and US market holidays.
    """
    # Ensure start_date is a datetime object
    if isinstance(start_date, str):
        start_date = datetime.strptime(start_date, '%Y-%m-%d')
    
    nyse = mcal.get_calendar('NYSE')
    # Schedule trading days for the next window
    schedule = nyse.schedule(start_date=start_date, end_date=start_date + timedelta(days=30))
    business_days = schedule.index[:num_days]
    return business_days

# Fetch stock data
def fetch_stock_data(ticker, start_date, end_date):
    """
    Fetch historical stock data for a given ticker.
    """
    stock = yf.download(ticker, start=start_date, end=end_date)
    stock['Ticker'] = ticker
    return stock

# Add technical indicators
def add_technical_indicators(data):
    """
    Add technical indicators (RSI, MACD, Bollinger Bands, Moving Averages) to the data.
    """
    # RSI
    delta = data['Close'].diff(1)
    gain = delta.where(delta > 0, 0).rolling(window=14).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
    rs = gain / loss
    data['RSI'] = 100 - (100 / (1 + rs))

    # MACD
    data['EMA_12'] = data['Close'].ewm(span=12, adjust=False).mean()
    data['EMA_26'] = data['Close'].ewm(span=26, adjust=False).mean()
    data['MACD'] = data['EMA_12'] - data['EMA_26']

    # Bollinger Bands
    data['MA_20'] = data['Close'].rolling(window=20).mean()
    rolling_std = data['Close'].rolling(window=20).std()

    # Ensure rolling_std is a Series
    if isinstance(rolling_std, pd.DataFrame):
        rolling_std = rolling_std.squeeze()

    data['BB_Upper'] = data['MA_20'] + 2 * rolling_std
    data['BB_Lower'] = data['MA_20'] - 2 * rolling_std

    # Moving Averages
    data['Rolling_5'] = data['Close'].rolling(window=5).mean()
    data['Rolling_10'] = data['Close'].rolling(window=10).mean()

    # Lag Features
    data['Lag_1'] = data['Close'].shift(1)
    data['Lag_5'] = data['Close'].shift(5)

    # Daily Returns
    data['Daily_Return'] = data['Close'].pct_change()

    return data

# Fetch sentiment data
def fetch_sentiment(keyword, start_date, end_date):
    """
    Fetch sentiment score based on Google News headlines between given dates.
    """
    url = f"https://news.google.com/search?q={keyword}+after:{start_date}+before:{end_date}"
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    headlines = [title.text for title in soup.find_all('a', {'class': 'DY5T1d'})]
    sentiment_score = len(headlines) % 100  # Placeholder logic
    return sentiment_score

# Pipeline to preprocess data
def data_pipeline(user_date):
    """
    Complete data pipeline for fetching and preprocessing data.
    """
    user_date = datetime.strptime(user_date, '%Y-%m-%d')
    end_date = user_date.strftime('%Y-%m-%d')
    start_date = (user_date - timedelta(days=365 * 5)).strftime('%Y-%m-%d')

    nvda_data = fetch_stock_data('NVDA', start_date, end_date)
    sp500_data = fetch_stock_data('^GSPC', start_date, end_date)

    nvda_data = add_technical_indicators(nvda_data)
    sp500_data = add_technical_indicators(sp500_data)

    sentiment_start_date = (user_date - timedelta(days=10)).strftime('%Y-%m-%d')
    sentiment_end_date = user_date.strftime('%Y-%m-%d')
    nvda_sentiment = fetch_sentiment('NVIDIA stock', sentiment_start_date, sentiment_end_date)
    sp500_sentiment = fetch_sentiment('S&P 500', sentiment_start_date, sentiment_end_date)

    nvda_data['Sentiment'] = nvda_sentiment
    sp500_data['Sentiment'] = sp500_sentiment

    combined_data = pd.concat([nvda_data, sp500_data[['Close']].rename(columns={'Close': 'SP500_Close'})], axis=1)
    combined_data = combined_data[combined_data.index < end_date]

    print(f"Pipeline completed with {combined_data.shape[0]} rows.")
    return combined_data

# Create target variables
def create_target_variables(data):
    """
    Add target columns for the highest, lowest, and average closing prices
    for the next 5 business days.
    """
    data['High_5'] = data['High'].rolling(window=5).max().shift(-5)
    data['Low_5'] = data['Low'].rolling(window=5).min().shift(-5)
    data['Avg_Close_5'] = data['Close'].rolling(window=5).mean().shift(-5)
    return data

# Prepare data for modeling
def prepare_data(data, target_col):
    """
    Prepare data for training by splitting into features and target.
    """
    # Reset index to avoid multi-index issues
    data = data.reset_index(drop=True)

    features = data.drop(columns=['High_5', 'Low_5', 'Avg_Close_5', 'Ticker'], errors='ignore')
    target = data[target_col]

    clean_data = pd.concat([features, target], axis=1).dropna()
    if clean_data.empty:
        print(f"No valid rows left for {target_col} after dropping NaN values.")
        return None, None, None, None

    features = clean_data.iloc[:, :-1]
    target = clean_data.iloc[:, -1]
    return train_test_split(features, target, test_size=0.2, random_state=42)

# Train a Random Forest model
def train_random_forest(X_train, X_test, y_train, y_test, target_name):
    """
    Train a Random Forest Regressor and evaluate its performance.
    """
    if X_train is None or y_train is None:
        print(f"Skipping training for {target_name} due to insufficient data.")
        return None

    print(f"Training Random Forest model for {target_name}...")
    model = RandomForestRegressor(random_state=42, n_estimators=100)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    print(f"Model for {target_name} trained. MAE: {mae:.2f}")
    return model

# Train models for each target
def train_models(data):
    """
    Train Random Forest models to predict High_5, Low_5, and Avg_Close_5.
    """
    data = create_target_variables(data)

    print("Training High_5 model...")
    X_train, X_test, y_train, y_test = prepare_data(data, 'High_5')
    high_model = train_random_forest(X_train, X_test, y_train, y_test, 'High_5')

    print("Training Low_5 model...")
    X_train, X_test, y_train, y_test = prepare_data(data, 'Low_5')
    low_model = train_random_forest(X_train, X_test, y_train, y_test, 'Low_5')

    print("Training Avg_Close_5 model...")
    X_train, X_test, y_train, y_test = prepare_data(data, 'Avg_Close_5')
    avg_close_model = train_random_forest(X_train, X_test, y_train, y_test, 'Avg_Close_5')

    return high_model, low_model, avg_close_model

def make_predictions(data, high_model, low_model, avg_close_model, user_date):
    """
    Make predictions for the next 5 business days using the trained models
    and return the predictions as arrays.
    """
    # Reset index to avoid multi-index issues
    data = data.reset_index(drop=True)

    # Prepare the last 5 rows of data for predictions
    last_data = data.iloc[-5:].drop(columns=['High_5', 'Low_5', 'Avg_Close_5', 'Ticker']).dropna()
    if len(last_data) == 5:
        future_dates = get_next_business_days(user_date)
        high_pred = high_model.predict(last_data)
        low_pred = low_model.predict(last_data)
        avg_close_pred = avg_close_model.predict(last_data)

        print("\nPredictions for the next 5 business days:")
        for i, future_date in enumerate(future_dates):
            print(f"Day {i+1}, {future_date.date()}:")
            print(f"  Predicted High: {high_pred[i]:.2f}")
            print(f"  Predicted Low: {low_pred[i]:.2f}")
            print(f"  Predicted Average Close: {avg_close_pred[i]:.2f}")

        # Return predictions as arrays for further use
        return high_pred, low_pred, avg_close_pred
    else:
        print("Insufficient data for predictions.")
        return None, None, None

# Example of Using Predictions in Subsequent Functions
def perform_additional_functions(data, high_pred, low_pred, avg_close_pred):
    """
    Perform additional operations using the predictions.
    """
    result = []
    # Save predictions into arrays for later use
    high_predictions = np.array(high_pred)
    low_predictions = np.array(low_pred)
    avg_close_predictions = np.array(avg_close_pred)

    # Example: Calculate overall statistics
    highest_price = np.max(high_predictions)
    lowest_price = np.min(low_predictions)
    avg_closing_price = np.mean(avg_close_predictions)

    print("\nSummary of Predictions:")
    print(f"Highest Predicted Price: {highest_price:.2f}")
    print(f"Lowest Predicted Price: {lowest_price:.2f}")
    print(f"Average Closing Price: {avg_closing_price:.2f}")
    result.append(highest_price)
    result.append(lowest_price) 
    result.append(avg_closing_price)
    return result
    # Example: Use predictions in trading strategy or further calculations
    # Implement your logic here

def perform_predictions(user_date):
    # Your existing model code goes here
    data = data_pipeline(user_date)
    high_model, low_model, avg_close_model = train_models(data)
    high_pred, low_pred, avg_close_pred = make_predictions(data, high_model, low_model, avg_close_model, user_date)
    
    if high_pred is not None and low_pred is not None and avg_close_pred is not None:
        results = perform_additional_functions(data, high_pred, low_pred, avg_close_pred)
        return results
    return None

if __name__ == '__main__':
    app.run(debug=False)
