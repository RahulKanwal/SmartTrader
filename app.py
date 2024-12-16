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
    user_date = None
    if request.method == 'POST':
        user_date = request.form['date']
        # Call your prediction function here with user_date
        data = data_pipeline(user_date)
        high_model, low_model, avg_close_model, open_model = train_models(data)
        high_pred, low_pred, avg_close_pred, open_pred = make_predictions(data, high_model, low_model, avg_close_model, open_model, user_date)
        if high_pred is not None and low_pred is not None and avg_close_pred is not None and open_pred is not None:
            predictions = perform_additional_functions(data, high_pred, low_pred, avg_close_pred, open_pred, user_date)
        return render_template('index.html', predictions=predictions, selected_date = user_date)
    return render_template('index.html')

# Helper function: Get next business days
def get_next_business_days(start_date, num_days=5):
    if isinstance(start_date, str):
        start_date = datetime.strptime(start_date, '%Y-%m-%d')
    nyse = mcal.get_calendar('NYSE')
    schedule = nyse.schedule(start_date=start_date, end_date=start_date + timedelta(days=30))
    business_days = schedule.index[:num_days]
    return business_days

# Fetch stock data
def fetch_stock_data(ticker, start_date, end_date):
    stock = yf.download(ticker, start=start_date, end=end_date)
    stock['Ticker'] = ticker
    return stock

# Add technical indicators
def add_technical_indicators(data):
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
    user_date = datetime.strptime(user_date, '%Y-%m-%d') + timedelta(days=1)  # Start from next day
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
    data['High_5'] = data['High'].rolling(window=5).max().shift(-5)
    data['Low_5'] = data['Low'].rolling(window=5).min().shift(-5)
    data['Avg_Close_5'] = data['Close'].rolling(window=5).mean().shift(-5)
    data['Open_5'] = data['Open'].shift(-5)  # New target variable for Open prices

    return data

# Prepare data for modeling
def prepare_data(data, target_col):
    data = data.reset_index(drop=True)
    features = data.drop(columns=['High_5', 'Low_5', 'Avg_Close_5', 'Open_5', 'Ticker'], errors='ignore')
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
    if X_train is None or y_train is None:
        print(f"Skipping training for {target_name} due to insufficient data.")
        return None
    print(f"Training Random Forest model for {target_name}...")
    model = RandomForestRegressor(random_state=42, n_estimators=200)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    print(f"Model for {target_name} trained. MAE: {mae:.2f}")
    return model

# Train models for each target
def train_models(data):
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
    print("Training Open_5 model...")
    X_train, X_test, y_train, y_test = prepare_data(data, 'Open_5')
    open_model = train_random_forest(X_train, X_test, y_train, y_test, 'Open_5')
    return high_model, low_model, avg_close_model, open_model

def make_predictions(data, high_model, low_model, avg_close_model, open_model, user_date):
    """
    Make predictions for the next 5 business days starting from the next day.
    """
    user_date = datetime.strptime(user_date, '%Y-%m-%d') + timedelta(days=1)
    data = data.reset_index(drop=True)
    last_data = data.iloc[-5:].drop(columns=['High_5', 'Low_5', 'Avg_Close_5', 'Open_5', 'Ticker']).dropna()
    
    if len(last_data) == 5:
        future_dates = get_next_business_days(user_date)
        high_pred = high_model.predict(last_data)
        low_pred = low_model.predict(last_data)
        avg_close_pred = avg_close_model.predict(last_data)
        open_pred = open_model.predict(last_data)

        return high_pred, low_pred, avg_close_pred, open_pred
    else:
        print("Insufficient data for predictions.")
        return None, None, None

def simulate_trading_strategy(data, high_pred, low_pred, avg_close_pred, open_pred, user_date):
    user_date = datetime.strptime(user_date, '%Y-%m-%d') + timedelta(days=1)
    future_dates = get_next_business_days(user_date)
    # Initial portfolio as specified in project guidelines
    nvda_shares = 10000
    nvdq_shares = 100000
    table_data = []

    for i, future_date in enumerate(future_dates):
        nvda_open = open_pred[i]  
        nvda_close_prediction = avg_close_pred[i]

        # Calculate the predicted price movement
        price_change_percentage = ((nvda_close_prediction - nvda_open) / nvda_open) * 100
        
        if price_change_percentage > 0:
            # BULLISH: NVDA expected to rise, convert NVDQ to NVDA
            action = "BULLISH"
            # Calculate NVDQ value and convert to NVDA shares
            nvdq_value = nvdq_shares * nvda_open
            nvda_shares += nvdq_value / nvda_open
            nvdq_shares = 0
        elif price_change_percentage < 0:
            # BEARISH: NVDA expected to fall, convert NVDA to NVDQ
            action = "BEARISH"
            # Calculate NVDA value and convert to NVDQ shares
            nvda_value = nvda_shares * nvda_open
            nvdq_shares += nvda_value / nvda_open
            nvda_shares = 0
        else:
            # IDLE: No significant price movement expected
            action = "IDLE"

        # portfolio_value = nvda_shares * nvda_open + nvdq_shares * nvdq_open
        table_data.append([future_date.date(), action])

    return table_data

def perform_additional_functions(data, high_pred, low_pred, avg_close_pred, open_pred, user_date):
    high_predictions = np.array(high_pred)
    low_predictions = np.array(low_pred)
    open_predictions = np.array(open_pred)
    avg_close_predictions = np.array(avg_close_pred)
    highest_price = np.max(high_predictions)
    lowest_price = np.min(low_predictions)
    avg_closing_price = np.mean(avg_close_predictions)
    avg_opening_price = np.mean(open_predictions)
    print("\nSummary of Predictions:")
    print(f"Highest Predicted Price: {highest_price:.2f}")
    print(f"Lowest Predicted Price: {lowest_price:.2f}")
    print(f"Average Closing Price: {avg_closing_price:.2f}")
    print("\nSimulating Trading Strategy:")
    table_data = simulate_trading_strategy(data, high_pred, low_pred, avg_close_pred, open_pred, user_date)
   
    # Return results as a dictionary
    return {
        "highest_predicted_price": highest_price,
        "lowest_predicted_price": lowest_price,
        "average_closing_price": avg_closing_price,
        "average_opening_price": avg_opening_price,
        "trading_strategy_table": table_data,
    }

if __name__ == '__main__':
    app.run(debug=False)
