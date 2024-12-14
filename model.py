import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
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
        self.model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

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

        features = ['Open', 'High', 'Low', 'Close', 'Volume', 'Price_Momentum', 
                    'Volume_Momentum', 'Price_Acceleration', 'SMA_5', 'SMA_20', 
                    'MACD', 'Signal_Line', 'RSI', 'Daily_Range', 'ATR', 
                    'Trend_Strength', 'Channel_Position']
        
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
        features.dropna(inplace=True)

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
        
        X_reshaped = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))
        
        self.model.fit(X_reshaped, y, epochs=50, batch_size=32, verbose=0)

    def save_model(self, filename):
        self.model.save(filename)

# Training and saving models for NVDA and NVDQ
if __name__ == "__main__":
    nvda_data = pd.read_csv('nvda_processed.csv')
    nvdq_data = pd.read_csv('nvdq_processed.csv')
    
    nvda_predictor = StockPredictor()
    nvdq_predictor = StockPredictor()

    nvda_predictor.train(nvda_data)
    nvda_predictor.save_model('nvda_predictor.h5')

    nvdq_predictor.train(nvdq_data)
    nvdq_predictor.save_model('nvdq_predictor.h5')
