import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.optimizers import Adam  # เพิ่มบรรทัดนี้
from tensorflow.keras.callbacks import EarlyStopping
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

    def build_model(self, X_train):
        # สร้างโมเดล Sequential โดยใช้ Input layer แทนการใช้ input_shape
        model = Sequential()
        model.add(Input(shape=(X_train.shape[1], 1)))  # กำหนด input shape ที่นี่
        model.add(LSTM(units=50, return_sequences=True))
        model.add(Dropout(0.2))  # เพิ่ม Dropout layer เพื่อลด overfitting
        model.add(LSTM(units=50))
        model.add(Dropout(0.2))  # เพิ่ม Dropout อีกครั้ง
        model.add(Dense(units=1))

        # ใช้ Adam optimizer กับ learning rate ที่ปรับได้
        optimizer = Adam(learning_rate=0.001)  # กำหนด optimizer ที่นี่
        model.compile(optimizer=optimizer, loss='mean_squared_error')
        
        return model

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

        # สร้างโมเดล
        model = self.build_model(X_train)

        # ใช้ EarlyStopping เพื่อลด overfitting
        early_stopping = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)
        
        # Train the model
        model.fit(X_train, y_train, epochs=50, batch_size=32, callbacks=[early_stopping])

        # Predict and invert scaling
        predicted_stock_price = model.predict(X_test)
        predicted_stock_price = scaler.inverse_transform(predicted_stock_price)
        
        return predicted_stock_price

    def run(self):
        while True:
            for stock in self.stocks:
                print(f"Analyzing {stock}...")
                predicted_price = self.train_model(stock)
                print(f"\nPredicted price for {stock} in the next day: ${predicted_price[-1][0]:.2f}")
            time.sleep(60)  # Wait for 1 minute before next update

# List of stocks
stocks = ['AAPL', 'JPM', 'KO', 'CVX', 'AMZN', 'MSFT', 'PEP', 'O', 'AMT', 'BRK-B', 'GOOGL','TSLA','RTX']

if __name__ == "__main__":
    analyzer = StockAnalyzer(stocks)
    analyzer.run()
