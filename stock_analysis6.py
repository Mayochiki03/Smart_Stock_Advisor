import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.model_selection import train_test_split
import time

class StockAnalyzer:
    def __init__(self, stocks):
        self.stocks = stocks

    def fetch_historical_data(self, stock):
        ticker = yf.Ticker(stock)
        data = ticker.history(period="5y")  # 5 ปีย้อนหลัง
        return data['Close']

    def preprocess_data(self, data):
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(data.values.reshape(-1, 1))
        return scaled_data, scaler

    def create_dataset(self, data, time_step=60):
        X, y = [], []
        for i in range(len(data) - time_step - 1):
            X.append(data[i:(i + time_step), 0])
            y.append(data[i + time_step, 0])
        X, y = np.array(X), np.array(y)
        return X, y

    def train_model(self, stock):
        # Fetch data and preprocess
        data = self.fetch_historical_data(stock)
        scaled_data, scaler = self.preprocess_data(data)

        # Prepare the dataset for LSTM
        time_step = 60
        X, y = self.create_dataset(scaled_data, time_step)
        
        # Reshape data to be suitable for LSTM input
        X = X.reshape(X.shape[0], X.shape[1], 1)

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

        # Build LSTM model
        model = Sequential()
        model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
        model.add(LSTM(units=50))
        model.add(Dense(units=1))

        model.compile(optimizer='adam', loss='mean_squared_error')

        # Train the model
        model.fit(X_train, y_train, epochs=10, batch_size=32)

        # Predict and invert scaling
        predicted_stock_price = model.predict(X_test)
        predicted_stock_price = scaler.inverse_transform(predicted_stock_price)
        
        return predicted_stock_price

    def run(self):
        while True:
            for stock in self.stocks:
                predicted_price = self.train_model(stock)
                print(f"\nPredicted price for {stock} in the next day: ${predicted_price[-1][0]:.2f}")
            time.sleep(60)  # Wait for 1 minute before next update

# List of stocks
stocks = ['AAPL', 'JPM', 'KO', 'CVX', 'AMZN', 'MSFT', 'PEP', 'O', 'AMT', 'BRK-B', 'GOOGL','TSLA','RTX']

if __name__ == "__main__":
    analyzer = StockAnalyzer(stocks)
    analyzer.run()
