import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
import requests
from sklearn.linear_model import LinearRegression
import time
from datetime import datetime, timedelta

class StockAnalyzer:
    def __init__(self, stocks, news_api_key):
        self.stocks = stocks
        self.news_api_key = news_api_key 
        self.model = LinearRegression()
        self.next_sentiment_time = datetime.now()

    def fetch_stock_prices(self):
        prices = {}
        for stock in self.stocks:
            ticker = yf.Ticker(stock)
            last_data = ticker.history(period='1d')
            prices[stock] = last_data['Close'].iloc[-1]  # ดึงราคาปิดล่าสุด
        return prices

    def fetch_news(self, stock):
        url = f"https://newsapi.org/v2/everything?q={stock}&apiKey={self.news_api_key}"
        response = requests.get(url)
        if response.status_code == 200:
            return response.json()['articles']
        else:
            print(f'Error fetching news for {stock}')
            return []

    def fetch_historical_data(self, stock):
        ticker = yf.Ticker(stock)
        historical_data = ticker.history(period='5y')  # ดึงข้อมูลย้อนหลัง 5 ปี
        historical_data.reset_index(inplace=True)
        return historical_data

    def predict_price(self, stock):
        historical_data = self.fetch_historical_data(stock)
        if historical_data['Close'].isnull().any():
            historical_data = historical_data.dropna()  # ลบค่าที่เป็น NaN
        historical_data['Date'] = historical_data['Date'].map(datetime.toordinal)
        X = historical_data[['Date']].values  # ใช้ .values เพื่อให้ไม่มีชื่อฟีเจอร์
        y = historical_data['Close'].values
        if np.any(np.isnan(y)):
            y = y[~np.isnan(y)]  # ลบค่าที่เป็น NaN
            X = X[:len(y)]  # ตัด X ให้มีขนาดเท่ากับ y
        self.model.fit(X, y)
        next_day = np.array([[datetime.now().toordinal() + 1]])  # วันถัดไป
        predicted_price = self.model.predict(next_day)
        return predicted_price[0]

    def calculate_fair_value(self, stock):
        ticker = yf.Ticker(stock)
        info = ticker.info
        eps = info.get('trailingEps', None)
        pe_ratio = info.get('forwardPE', None)
        if eps is not None and pe_ratio is not None:
            fair_value = eps * pe_ratio
            return fair_value
        return None

    def run(self):
        while True:
            prices = self.fetch_stock_prices()
            print("\n### ราคาหุ้นปัจจุบัน ###")
            for stock, price in prices.items():
                print(f"{stock}: ${price:.6f}")

            # คาดการณ์ราคาหุ้นในวันถัดไป
            print("\n### ราคาที่คาดการณ์ในวันถัดไป ###")
            for stock in self.stocks:
                predicted_price = self.predict_price(stock)
                print(f"{stock}: ${predicted_price:.6f}")

            # คำนวณราคายุติธรรม
            print("\n### ราคายุติธรรม (Fair Value) ###")
            for stock in self.stocks:
                fair_value = self.calculate_fair_value(stock)
                if fair_value is not None:
                    print(f"{stock}: ราคายุติธรรม = ${fair_value:.6f}")
                else:
                    print(f"{stock}: ไม่สามารถคำนวณราคายุติธรรมได้")

                print("\n### สรุปผลการวิเคราะห์ Sentiment ของข่าวสาร ###")


            # รอ 1 นาทีก่อนทำการอัพเดตข้อมูลอีกครั้ง
            time.sleep(60)
            


# รายชื่อหุ้นที่ต้องการ
stocks = ['AAPL', 'JPM', 'KO', 'CVX', 'AMZN', 'MSFT', 'PEP', 'O', 'AMT', 'BRK-B', 'GOOGL','TSLA','RTX']
news_api_key = '1516f93d42d9467bb86e0e0fa1b819cc'  # แทนที่ด้วย API Key ของคุณ

if __name__ == "__main__":
    analyzer = StockAnalyzer(stocks, news_api_key)
    analyzer.run()