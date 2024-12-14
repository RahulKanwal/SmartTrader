import pandas as pd
import numpy as np
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model

class PortfolioManager:
    def __init__(self, nvda_shares=10000, nvdq_shares=100000):
        self.nvda_shares = nvda_shares
        self.nvdq_shares = nvdq_shares
        self.transaction_cost = 0.001
        self.min_return_threshold = 0.002

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
        # Load datasets
        self.nvda_data = pd.read_csv('nvda_processed.csv', parse_dates=['Date'], dayfirst=True)
        self.nvdq_data = pd.read_csv('nvdq_processed.csv', parse_dates=['Date'], dayfirst=True)
        
        # Sort data chronologically
        self.nvda_data.sort_values('Date', inplace=True)
        self.nvdq_data.sort_values('Date', inplace=True)

        # Load pre-trained models
        self.nvda_model = load_model('nvda_predictor.h5')
        self.nvdq_model = load_model('nvdq_predictor.h5')

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
            selected_timestamp = pd.Timestamp(selected_date)
            
            # Filter data up to the selected date
            nvda_train_data = self.nvda_data[self.nvda_data['Date'] <= selected_timestamp]
            nvdq_train_data = self.nvdq_data[self.nvdq_data['Date'] <= selected_timestamp]
            
            # Prepare features for prediction
            last_nvda_features = self.prepare_features(nvda_train_data).iloc[-1:].values
            last_nvdq_features = self.prepare_features(nvdq_train_data).iloc[-1:].values
            
            # Make predictions using the loaded models
            nvda_predictions = self.make_predictions(self.nvda_model, last_nvda_features)
            nvdq_predictions = self.make_predictions(self.nvdq_model, last_nvdq_features)
            
            # Initialize portfolio manager and execute trades based on predictions
            portfolio_manager = PortfolioManager()
            actions = []
            
            for i in range(5):
                action = portfolio_manager.determine_action(
                    nvda_predictions[i:], nvdq_predictions[i:]
                )
                actions.append(action)
                portfolio_manager.execute_trade(
                    action, nvda_predictions[i], nvdq_predictions[i]
                )
            
            final_value = portfolio_manager.calculate_value(
                nvda_predictions[-1], nvdq_predictions[-1]
            )
            
            # Display results and plot predictions
            self.display_results(nvda_predictions, nvdq_predictions, actions, final_value)
        
        except Exception as e:
            st.error(f"An unexpected error occurred: {str(e)}")
           
           
    def calculate_rsi(self, prices, period=14):
        prices = prices.fillna(method='ffill')
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period, min_periods=1).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period, min_periods=1).mean()
        
        rs = gain / loss.replace(0, np.inf)
        
        return 100 - (100 / (1 + rs))        
    def prepare_features(self, data, window=20):
        """
        Prepares features for prediction similar to the training process.
        Includes momentum, moving averages, volatility, and other indicators.
        """
        data = data.copy()
        data = data.fillna(method='ffill').fillna(method='bfill')  # Fill missing values

        # Price and volume momentum
        data['Price_Momentum'] = data['Close'].pct_change(window).fillna(0)
        data['Volume_Momentum'] = data['Volume'].pct_change(window).fillna(0)
        data['Price_Acceleration'] = data['Price_Momentum'].diff().fillna(0)

        # Moving averages and trends
        data['SMA_5'] = data['Close'].rolling(window=5, min_periods=1).mean()
        data['SMA_20'] = data['Close'].rolling(window=20, min_periods=1).mean()
        data['EMA_12'] = data['Close'].ewm(span=12, adjust=False).mean()
        data['EMA_26'] = data['Close'].ewm(span=26, adjust=False).mean()
        data['MACD'] = data['EMA_12'] - data['EMA_26']
        data['Signal_Line'] = data['MACD'].ewm(span=9, adjust=False).mean()

        # Volatility and momentum indicators
        data['Daily_Range'] = (data['High'] - data['Low']) / data['Open']
        data['ATR'] = data['Daily_Range'].rolling(window=14, min_periods=1).mean()
        data['RSI'] = self.calculate_rsi(data['Close'])
        data['Trend_Strength'] = abs(data['Close'] - data['SMA_20']) / (data['SMA_20']).replace(0, 1)

        # Price channels and support/resistance levels
        data['Upper_Channel'] = data['High'].rolling(window=window, min_periods=1).max()
        data['Lower_Channel'] = data['Low'].rolling(window=window, min_periods=1).min()
        data['Channel_Position'] = (data['Close'] - data['Lower_Channel']) / \
                                (data['Upper_Channel'] - data['Lower_Channel']).replace(0, 1)

        # Select relevant features
        features = ['Open', 'High', 'Low', 'Close', 'Volume', 'Price_Momentum', 
                    'Volume_Momentum', 'Price_Acceleration', 'SMA_5', 'SMA_20', 
                    'MACD', 'Signal_Line', 'RSI', 'Daily_Range', 'ATR', 
                    'Trend_Strength', 'Channel_Position']
        
        return data[features].fillna(method='ffill').fillna(method='bfill')


  

    def make_predictions(self, model, features):
        # Scale features as done during training and reshape for LSTM input
        scaler=MinMaxScaler()
        scaled_features=scaler.fit_transform(features)
        reshaped_features=scaled_features.reshape((1,1,-1))
        
        # Predict the next 5 days prices using the model
        predictions=model.predict(reshaped_features)
        return predictions.flatten()

    def display_results(self, nvdao_preds, nvadqa_preds, actions, final_value):
        st.subheader("Predictions for the next 5 business days:")
        col1,col2=st.columns(2)
        with col1:
            st.write("NVDA Predictions:")
            st.write(['${:.2f}'.format(p)for p in nvdao_preds])
        with col2:
            st.write("NVDQ Predictions:")
            st.write(['${:.2f}'.format(p)for p in nvadqa_preds])
        st.subheader("Recommended Actions:")
        for i,action in enumerate(actions,1):
            st.write(f"Day {i}: {action}")
        st.subheader("Final Portfolio Value:")
        st.write(f"${final_value:,.2f}")

def main():
	app=SmartTraderApp()
	app.run()

if __name__=="__main__":
	main()