import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from tkinter import *
from tkinter import ttk
from tkcalendar import DateEntry
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import streamlit as st
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam

class StockPredictor:
    def __init__(self):
        self.scaler = MinMaxScaler()
        self.model = Sequential([
            LSTM(50, activation='relu', return_sequences=True, input_shape=(None, 17)),
            LSTM(50, activation='relu'),
            Dense(5)
        ])
        self.model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')

    def prepare_features(self, data, window=20):
        data = data.copy()
        data = data.fillna(method='ffill').fillna(method='bfill')

        # Price and volume momentum
        data['Price_Momentum'] = data['Close'].pct_change(window, fill_method=None)
        data['Volume_Momentum'] = data['Volume'].pct_change(window, fill_method=None)
        data['Price_Acceleration'] = data['Price_Momentum'].diff()

        # Moving averages and trends
        data['SMA_5'] = data['Close'].rolling(window=5, min_periods=1).mean()
        data['SMA_20'] = data['Close'].rolling(window=20, min_periods=1).mean()
        data['EMA_12'] = data['Close'].ewm(span=12, adjust=False, min_periods=1).mean()
        data['EMA_26'] = data['Close'].ewm(span=26, adjust=False, min_periods=1).mean()
        data['MACD'] = data['EMA_12'] - data['EMA_26']
        data['Signal_Line'] = data['MACD'].ewm(span=9, adjust=False, min_periods=1).mean()

        # Volatility and momentum indicators
        data['Daily_Range'] = (data['High'] - data['Low']) / data['Open']
        data['ATR'] = data['Daily_Range'].rolling(window=14, min_periods=1).mean()
        data['RSI'] = self.calculate_rsi(data['Close'])
        data['Trend_Strength'] = abs(data['Close'] - data['SMA_20']) / data['SMA_20']

        # Price channels and support/resistance
        data['Upper_Channel'] = data['High'].rolling(window=20, min_periods=1).max()
        data['Lower_Channel'] = data['Low'].rolling(window=20, min_periods=1).min()
        data['Channel_Position'] = (data['Close'] - data['Lower_Channel']) / \
            (data['Upper_Channel'] - data['Lower_Channel']).replace(0, 1)

        features = ['Open', 'High', 'Low', 'Close', 'Volume', 'Price_Momentum', 'Volume_Momentum', 
                    'Price_Acceleration', 'SMA_5', 'SMA_20', 'MACD', 'Signal_Line', 'RSI', 
                    'Daily_Range', 'ATR', 'Trend_Strength', 'Channel_Position']
        return data[features].fillna(method='ffill').fillna(method='bfill')

    def calculate_rsi(self, prices, period=14):
        prices = prices.fillna(method='ffill')
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period, min_periods=1).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period, min_periods=1).mean()
        rs = gain / loss.replace(0, np.inf)
        return 100 - (100 / (1 + rs))

    def train(self, data):
        if len(data) < 50:
            raise ValueError("Insufficient historical data for training")
        
        features = self.prepare_features(data)
        features = features.dropna()

        # Create target variables with different time horizons
        y = np.column_stack([
            data['Close'].shift(-i).loc[features.index] for i in range(1, 6)
        ])
        features = features[:-5]
        y = y[:-5]

        valid_mask = ~(np.isnan(y).any(axis=1) | features.isna().any(axis=1))
        X = features[valid_mask].values
        y = y[valid_mask]

        if len(X) == 0 or len(y) == 0:
            raise ValueError("No valid data after preprocessing")

        self.scaler.fit(X)
        X_scaled = self.scaler.transform(X)

        # Reshape input for LSTM (samples, time steps, features)
        X_reshaped = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))

        self.model.fit(X_reshaped, y, epochs=50, batch_size=32, verbose=0)

    def predict(self, data, days=5):
        features = self.prepare_features(data)
        last_features = features.iloc[-1:].values
        predictions = []
        current_features = last_features.copy()

        # Calculate historical volatility with longer lookback
        historical_volatility = data['Close'].pct_change(30).std() * np.sqrt(252)
        daily_vol = historical_volatility / np.sqrt(252)
        last_close = float(data['Close'].iloc[-1])

        for i in range(days):
            scaled_features = self.scaler.transform(current_features)
            reshaped_features = scaled_features.reshape((1, 1, scaled_features.shape[1]))
            base_pred = float(self.model.predict(reshaped_features)[0][0])

            # Add random walk component
            random_walk = np.random.normal(0, daily_vol * last_close)
            trend_component = base_pred - last_close

            # Combine prediction with random walk
            pred = last_close + trend_component + random_walk
            predictions.append(pred)

            # Update features for next prediction
            current_features[0, 0:4] = pred
            current_features[0, 4] *= np.random.uniform(0.95, 1.05)

        return predictions

class PortfolioManager:
    def __init__(self, nvda_shares=10000, nvdq_shares=100000):
        self.nvda_shares = nvda_shares
        self.nvdq_shares = nvdq_shares
        self.transaction_cost = 0.001
        self.min_return_threshold = 0.002  # Lowered from 0.005

    def determine_action(self, nvda_pred, nvdq_pred):
        if len(nvda_pred) < 2 or len(nvdq_pred) < 2:
            return "IDLE"

        # Calculate short-term and long-term returns
        nvda_short_return = (nvda_pred[1] - nvda_pred[0]) / nvda_pred[0]
        nvda_long_return = (nvda_pred[-1] - nvda_pred[0]) / nvda_pred[0]
        nvdq_short_return = (nvdq_pred[1] - nvdq_pred[0]) / nvdq_pred[0]
        nvdq_long_return = (nvdq_pred[-1] - nvdq_pred[0]) / nvdq_pred[0]

        # Calculate momentum scores
        nvda_momentum = sum(np.diff(nvda_pred) > 0) / (len(nvda_pred) - 1)
        nvdq_momentum = sum(np.diff(nvdq_pred) > 0) / (len(nvdq_pred) - 1)

        # Combined signals
        nvda_signal = (nvda_short_return + nvda_long_return) * nvda_momentum
        nvdq_signal = (nvdq_short_return + nvdq_long_return) * nvdq_momentum

        if nvda_signal > nvdq_signal and nvda_signal > self.min_return_threshold:
            return "BULLISH"
        elif nvdq_signal > nvda_signal and nvdq_signal > self.min_return_threshold:
            return "BEARISH"
        return "IDLE"

    def execute_trade(self, action, nvda_price, nvdq_price):
        if action == "BULLISH" and self.nvdq_shares > 0:
            trade_value = self.nvdq_shares * nvdq_price * (1 - self.transaction_cost)
            self.nvda_shares += trade_value / nvda_price
            self.nvdq_shares = 0
        elif action == "BEARISH" and self.nvda_shares > 0:
            trade_value = self.nvda_shares * nvda_price * (1 - self.transaction_cost)
            self.nvdq_shares += trade_value / nvdq_price
            self.nvda_shares = 0

    def calculate_value(self, nvda_price, nvdq_price):
        return (self.nvda_shares * nvda_price + self.nvdq_shares * nvdq_price)

class SmartTraderApp:
    def __init__(self):
        self.nvda_data = pd.read_csv('nvda_processed.csv', parse_dates=['Date'], dayfirst=True)
        self.nvdq_data = pd.read_csv('nvdq_processed.csv', parse_dates=['Date'], dayfirst=True)
        # Sort data chronologically
        self.nvda_data = self.nvda_data.sort_values('Date')
        self.nvdq_data = self.nvdq_data.sort_values('Date')
        # Initialize predictors
        self.nvda_predictor = StockPredictor()
        self.nvdq_predictor = StockPredictor()

    def run(self):
        st.title("SmartTrader")
        # Date selection
        min_date = max(self.nvda_data['Date'].min(), self.nvdq_data['Date'].min())
        max_date = min(self.nvda_data['Date'].max(), self.nvdq_data['Date'].max())
        selected_date = st.date_input("Select Date", min_value=min_date.date(), max_value=max_date.date(), value=max_date.date())
        if st.button("Predict"):
            self.predict(selected_date)

    def predict(self, selected_date):
        try:
            # Convert selected_date to pandas Timestamp
            selected_timestamp = pd.Timestamp(selected_date)
            # Train models
            self.nvda_predictor.train(self.nvda_data[self.nvda_data['Date'] <= selected_timestamp])
            self.nvdq_predictor.train(self.nvdq_data[self.nvdq_data['Date'] <= selected_timestamp])
            # Make predictions
            nvda_predictions = self.nvda_predictor.predict(
                self.nvda_data[self.nvda_data['Date'] <= selected_timestamp])
            nvdq_predictions = self.nvdq_predictor.predict(
                self.nvdq_data[self.nvdq_data['Date'] <= selected_timestamp])
            # Initialize portfolio manager with default values
            portfolio = PortfolioManager()
            actions = []
            for i in range(5):
                action = portfolio.determine_action(
                    nvda_predictions[i:], nvdq_predictions[i:])
                actions.append(action)
                portfolio.execute_trade(
                    action, nvda_predictions[i], nvdq_predictions[i])
            final_value = portfolio.calculate_value(
                nvda_predictions[-1], nvdq_predictions[-1])
            # Display results
            self.display_results(nvda_predictions, nvdq_predictions, actions, final_value)
            self.plot_predictions(nvda_predictions, nvdq_predictions)
        except Exception as e:
            st.error(f"An unexpected error occurred: {str(e)}")

    def display_results(self, nvda_pred, nvdq_pred, actions, final_value):
        st.subheader("Predictions for the next 5 business days:")
        col1, col2 = st.columns(2)
        with col1:
            st.write("NVDA Predictions:")
            st.write(['${:.2f}'.format(p) for p in nvda_pred])
        with col2:
            st.write("NVDQ Predictions:")
            st.write(['${:.2f}'.format(p) for p in nvdq_pred])
        st.subheader("Recommended Actions:")
        for i, action in enumerate(actions, 1):
            st.write(f"Day {i}: {action}")
        st.subheader("Final Portfolio Value:")
        st.write(f"${final_value:,.2f}")

    def plot_predictions(self, nvda_pred, nvdq_pred):
        fig, ax = plt.subplots()
        days = range(1, len(nvda_pred) + 1)
        ax.plot(days, nvda_pred, 'b-', label='NVDA')
        ax.plot(days, nvdq_pred, 'r-', label='NVDQ')
        ax.set_title('Price Predictions for Next 5 Business Days')
        ax.set_xlabel('Days')
        ax.set_ylabel('Price ($)')
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)

def main():
    app = SmartTraderApp()
    app.run()

if __name__ == "__main__":
    main()