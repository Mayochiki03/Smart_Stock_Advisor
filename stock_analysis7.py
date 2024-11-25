import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
import streamlit as st
import plotly.graph_objs as go

class StockAnalyzer:
    def __init__(self, stocks):
        self.stocks = stocks
        self.model = None  # Store the model once created

    def fetch_historical_data(self, stock):
        ticker = yf.Ticker(stock)
        data = ticker.history(period="5y")  # Get the last 5 years of data
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
        if self.model is None:  # Build the model only once
            self.model = Sequential()
            self.model.add(Input(shape=(X_train.shape[1], 1)))
            self.model.add(LSTM(units=50, return_sequences=True))
            self.model.add(Dropout(0.2))
            self.model.add(LSTM(units=50))
            self.model.add(Dropout(0.2))
            self.model.add(Dense(units=1))

            optimizer = Adam(learning_rate=0.001)
            self.model.compile(optimizer=optimizer, loss='mean_squared_error')
        return self.model

    def train_model(self, stock):
        data = self.fetch_historical_data(stock)
        scaled_data, scaler = self.preprocess_data(data)

        time_step = 60
        X, y = self.create_dataset(scaled_data, time_step)
        X = X.reshape(X.shape[0], X.shape[1], 1)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

        model = self.build_model(X_train)
        early_stopping = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)
        model.fit(X_train, y_train, epochs=50, batch_size=32, callbacks=[early_stopping])

        predicted_stock_price = model.predict(X_test, batch_size=32)  # Keep batch size fixed
        predicted_stock_price = scaler.inverse_transform(predicted_stock_price)

        return predicted_stock_price, data, model, scaler

    def plot_graph(self, stock, predicted_price, data):
        fig = go.Figure()

        # Create time axis for historical data (real data)
        time_axis = data.index[-len(predicted_price):]

        # Plot actual stock price (real data)
        fig.add_trace(go.Scatter(x=data.index, y=data.values, mode='lines', name=f'{stock} Actual Prices'))

        # Plot predicted stock price
        fig.add_trace(go.Scatter(x=time_axis, 
                                 y=predicted_price.flatten(), 
                                 mode='lines', 
                                 name=f'{stock} Predicted Prices'))

        fig.update_layout(title=f'{stock} Stock Price Prediction', 
                          xaxis_title='Date', yaxis_title='Price',
                          height=500,  # Increase height for better visibility
                          width=800)   # Increase width
        return fig

    def predict_tomorrow_price(self, model, scaler, data):
        # Use the last 60 days for prediction
        last_data_point = data[-60:].values  # Use the last 60 days
        last_data_point_scaled = scaler.transform(last_data_point.reshape(-1, 1))
        last_data_point_scaled = last_data_point_scaled.reshape(1, -1, 1)

        # Predict tomorrow's price
        predicted_price_scaled = model.predict(last_data_point_scaled)
        tomorrow_price = scaler.inverse_transform(predicted_price_scaled)

        return tomorrow_price.item()  # Return scalar value for tomorrow's predicted price

# List of stocks
stocks = ['AAPL', 'JPM', 'KO', 'CVX', 'AMZN', 'MSFT', 'PEP', 'O', 'AMT', 'BRK-B', 'GOOGL', 'TSLA', 'RTX']

def main():
    st.title('Stock Price Prediction using LSTM')

    analyzer = StockAnalyzer(stocks)

    # Loop through each stock and display the graph and predicted price
    for stock in stocks:
        st.header(stock)

        # Train model and generate graph
        predicted_price, data, model, scaler = analyzer.train_model(stock)
        fig = analyzer.plot_graph(stock, predicted_price, data)

        # Predict the price for tomorrow
        tomorrow_price = analyzer.predict_tomorrow_price(model, scaler, data)

        # Display the predicted price for tomorrow
        st.write(f"📈 **Predicted price for tomorrow: {tomorrow_price:.2f} USD**")

        # Show the graph
        st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
    