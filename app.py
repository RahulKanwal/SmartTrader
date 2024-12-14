import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import joblib

class StockPredictor:
    def __init__(self):
        # Load pre-trained models and scalers
        self.nvda_model = load_model('nvda_model.h5')
        self.nvdq_model = load_model('nvdq_model.h5')
        self.scaler_nvda = joblib.load('scaler_nvda.pkl')
        self.scaler_nvdq = joblib.load('scaler_nvdq.pkl')

    def prepare_features(self, data, window=20):
        data = data.copy()
        data = data.fillna(method='ffill').fillna(method='bfill')

        # Feature engineering
        data['Price_Momentum'] = data['Close'].pct_change(window, fill_method=None)
        data['Volume_Momentum'] = data['Volume'].pct_change(window, fill_method=None)
        data['Price_Acceleration'] = data['Price_Momentum'].diff()
        data['SMA_5'] = data['Close'].rolling(window=5, min_periods=1).mean()
        data['SMA_20'] = data['Close'].rolling(window=20, min_periods=1).mean()
        data['EMA_12'] = data['Close'].ewm(span=12, adjust=False, min_periods=1).mean()
        data['EMA_26'] = data['Close'].ewm(span=26, adjust=False, min_periods=1).mean()
        data['MACD'] = data['EMA_12'] - data['EMA_26']
        data['Signal_Line'] = data['MACD'].ewm(span=9, adjust=False, min_periods=1).mean()
        data['Daily_Range'] = (data['High'] - data['Low']) / data['Open']
        data['ATR'] = data['Daily_Range'].rolling(window=14, min_periods=1).mean()
        data['RSI'] = self.calculate_rsi(data['Close'])
        data['Trend_Strength'] = abs(data['Close'] - data['SMA_20']) / data['SMA_20']
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

    def predict(self, model_name, data, days=5):
        features = self.prepare_features(data)
        last_features = features.iloc[-1:].values
        predictions = []
        
        if model_name == "nvda":
            model = self.nvda_model
            scaler = self.scaler_nvda
        else:
            model = self.nvdq_model
            scaler = self.scaler_nvdq
        
        current_features = last_features.copy()

        # Calculate historical volatility with longer lookback
        historical_volatility = data['Close'].pct_change(30).std() * np.sqrt(252)
        daily_vol = historical_volatility / np.sqrt(252)
        last_close = float(data['Close'].iloc[-1])

        for i in range(days):
            scaled_features = scaler.transform(current_features)
            reshaped_features = scaled_features.reshape((1, 1, scaled_features.shape[1]))
            base_pred = float(model.predict(reshaped_features)[0][0])

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
    def __init__(self):
        self.nvda_shares = 10000
        self.nvdq_shares = 100000
        self.transaction_cost = 0.001
        self.min_return_threshold = 0.002

    def determine_action(self, nvda_pred, nvdq_pred):
        if len(nvda_pred) < 2 or len(nvdq_pred) < 2:
            return "IDLE"

        nvda_short_return=(nvda_pred[1] - nvda_pred[0]) / nvda_pred[0]
        nvda_long_return=(nvda_pred[-1] - nvda_pred[0]) / nvda_pred[0]
        
        nvdq_short_return=(nvdq_pred[1] - nvdq_pred[0]) / nvdq_pred[0]
                
        nvdq_long_return=(nvdq_pred[-1] - nvdq_pred[0]) / nvdq_pred[0]

        nvda_momentum=sum(np.diff(nvda_pred) > 0) / (len(nvda_pred) - 1)
        nvdq_momentum=sum(np.diff(nvdq_pred) > 0) / (len(nvdq_pred) - 1)

        nvda_signal=(nvda_short_return + nvda_long_return) * nvda_momentum
        nvdq_signal=(nvdq_short_return + nvdq_long_return) * nvdq_momentum

        if nvda_signal > nvdq_signal and nvda_signal > self.min_return_threshold:
            return "BULLISH"
        elif nvdq_signal > nvda_signal and nvdq_signal > self.min_return_threshold:
            return "BEARISH"
        return "IDLE"
    
    def execute_trade(self, action,nvda_price,nvdq_price):
        if action == "BULLISH" and self.nvdq_shares > 0:
            trade_value=self.nvdq_shares * nvdq_price * (1 - self.transaction_cost)
            self.nvda_shares += trade_value / nvda_price
            self.nvdq_shares=0
        elif action == "BEARISH" and self.nvda_shares > 0:
            trade_value=self.nvda_shares * nvda_price * (1 - self.transaction_cost)
            self.nvdq_shares += trade_value / nvdq_price
            self.nvda_shares=0

    def calculate_value(self,nvda_price,nvdq_price):
        return (self.nvda_shares * nvda_price + self.nvdq_shares * nvdq_price)

class SmartTraderApp:
    def __init__(self):
       self.nvda_data=pd.read_csv('nvda_processed.csv', parse_dates=['Date'], dayfirst=True)
       self.nvdq_data=pd.read_csv('nvdq_processed.csv', parse_dates=['Date'], dayfirst=True)
       # Sort datasets chronologically.
       self.nvda_data.sort_values('Date', inplace=True)
       self.nvdq_data.sort_values('Date', inplace=True)
       # Initialize predictors.
       self.nvda_predictor=StockPredictor()
       self.nvdq_predictor=StockPredictor()

    def run(self):
       st.title("SmartTrader")
       
       # Unified date selection logic.
       min_date=max(self.nvda_data.Date.min(), self.nvdq_data.Date.min()).date()
       max_date=min(self.nvda_data.Date.max(), self.nvdq_data.Date.max()).date()
       selected_date=st.date_input("Select Date", min_value=min_date, max_value=max_date, value=max_date)
       
       if st.button("Predict"):
           nvda_predictions=self.predict("nvda", selected_date)
           nvdq_predictions=self.predict("nvdq", selected_date)
           st.write("NVDA Predictions:", nvda_predictions)
           st.write("NVDQ Predictions:", nvdq_predictions)
        
           # Calculate statistics for NVDA and NVDQ predictions.
           stats_df=self.calculate_statistics(nvda_predictions, nvdq_predictions)
           st.table(stats_df)
           strategies=self.determine_strategies(nvda_predictions, nvdq_predictions)
           for i, strategy in enumerate(strategies):
               st.write(f"Day {i+1} Strategy: {strategy}")
               
           final_portfolio_value = self.calculate_portfolio_value(nvda_predictions,nvdq_predictions)
           st.write(f"Final Portfolio Value: ${final_portfolio_value:,.2f}")
           

    def predict(self, model_name, selected_date):
      try:
         selected_timestamp=pd.Timestamp(selected_date)

         if model_name == "nvda":
             predictor=self.nvda_predictor
             filtered_data=self.nvda_data[self.nvda_data.Date <= selected_timestamp]
         else:
             predictor=self.nvdq_predictor
             filtered_data=self.nvdq_data[self.nvdq_data.Date <= selected_timestamp]

         predictions=predictor.predict(model_name=model_name,
                                       data=filtered_data)

         return predictions

      except Exception as e:
         st.error(f"An unexpected error occurred: {str(e)}")
         return []

    def calculate_statistics(self, nvda_predictions, nvdq_predictions):
      # Calculate statistics for NVDA.
      nvda_highest=np.max(nvda_predictions)
      nvda_lowest=np.min(nvda_predictions)
      nvda_average=np.mean(nvda_predictions)

      # Calculate statistics for NVDQ.
      nvdq_highest=np.max(nvdq_predictions)
      nvdq_lowest=np.min(nvdq_predictions)
      nvdq_average=np.mean(nvdq_predictions)

      # Create a DataFrame to display the results.
      stats_data={
          'Stock': ['NVDA', 'NVDQ'],
          'Highest': [nvda_highest, nvdq_highest],
          'Lowest': [nvda_lowest, nvdq_lowest],
          'Average': [nvda_average, nvdq_average]
      }

      stats_df=pd.DataFrame(stats_data)
      
      return stats_df
    
    def determine_strategies(self,nvda_predictions,nvdq_predictions):
        portfolio_manager=PortfolioManager()
        strategies=[]
        for i in range(len(nvda_predictions)):
            strategy=portfolio_manager.determine_action(
            nvda_predictions[i:],nvdq_predictions[i:])
            strategies.append(strategy)
        return strategies
    
    def calculate_portfolio_value(self,nvda_predictions,nvdq_predictions):
        portfolio_manager=PortfolioManager()

        for i in range(len(nvda_predictions)):
            action=portfolio_manager.determine_action(
            nvda_predictions[i:],nvdq_predictions[i:])
            portfolio_manager.execute_trade(action,
                                    nvda_predictions[i],nvdq_predictions[i])

        final_value=portfolio_manager.calculate_value(nvda_predictions[-1],nvdq_predictions[-1])
        return final_value
def main():
   app=SmartTraderApp()
   app.run()

if __name__ == "__main__":
   main()
