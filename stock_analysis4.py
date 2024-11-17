import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
import requests
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.linear_model import LinearRegression
import time
from datetime import datetime, timedelta

# สร้างคลาสสำหรับการวิเคราะห์ราคาหุ้น
class StockAnalyzer:
    def __init__(self, stocks, news_api_key):
        self.stocks = stocks
        self.news_api_key = news_api_key
        self.analyzer = SentimentIntensityAnalyzer()
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

    def analyze_sentiment(self, articles):
        sentiments = []
        for article in articles:
            sentiment = self.analyzer.polarity_scores(article.get('content', ''))
            sentiments.append(sentiment)
        return sentiments

    def fetch_historical_data(self, stock):
        ticker = yf.Ticker(stock)
        historical_data = ticker.history(period='5y')  # ดึงข้อมูลย้อนหลัง 5 ปี
        historical_data.reset_index(inplace=True)
        return historical_data

    def predict_price(self, stock):
        historical_data = self.fetch_historical_data(stock)

        # ตรวจสอบข้อมูลที่เป็น NaN
        if historical_data['Close'].isnull().any():
            historical_data = historical_data.dropna()  # ลบค่าที่เป็น NaN

        # แปลงวันเป็นตัวเลข
        historical_data['Date'] = historical_data['Date'].map(datetime.toordinal)
        X = historical_data[['Date']].values  # ใช้ .values เพื่อให้ไม่มีชื่อฟีเจอร์
        y = historical_data['Close'].values

        # ตรวจสอบว่าข้อมูล y ไม่มี NaN
        if np.any(np.isnan(y)):
            y = y[~np.isnan(y)]  # ลบค่าที่เป็น NaN
            X = X[:len(y)]  # ตัด X ให้มีขนาดเท่ากับ y

        # ฝึกโมเดล
        self.model.fit(X, y)

        # คาดการณ์ราคาหุ้นในวันถัดไป
        next_day = np.array([[datetime.now().toordinal() + 1]])  # วันถัดไป
        predicted_price = self.model.predict(next_day)

        return predicted_price[0]

    def run(self):
        while True:
            # ดึงราคาหุ้น
            prices = self.fetch_stock_prices()
            print("\n### ราคาหุ้นปัจจุบัน ###")
            for stock, price in prices.items():
                print(f"{stock}: ${price:.6f}")

            # อัพเดตการวิเคราะห์อารมณ์ทุกๆ 24 ชั่วโมง
            if datetime.now() >= self.next_sentiment_time:
                all_sentiments = []
                for stock in self.stocks:
                    news_articles = self.fetch_news(stock)
                    sentiments = self.analyze_sentiment(news_articles)
                    avg_positive_sentiment = sum([s['pos'] for s in sentiments]) / len(sentiments) if sentiments else 0
                    all_sentiments.append({'Stock': stock, 'Positive Sentiment': avg_positive_sentiment})

                sentiment_df = pd.DataFrame(all_sentiments)
                print("\n### สรุปผลการวิเคราะห์ Sentiment ของข่าวสาร ###")
                for index, row in sentiment_df.iterrows():
                    print(f"{row['Stock']}: Positive Sentiment = {row['Positive Sentiment']:.6f}")

                # ตั้งเวลาใหม่สำหรับการวิเคราะห์อารมณ์ครั้งถัดไป
                self.next_sentiment_time = datetime.now() + timedelta(hours=24)

            # คาดการณ์ราคาหุ้นในวันถัดไป
            print("\n### ราคาที่คาดการณ์ในวันถัดไป ###")
            for stock in self.stocks:
                predicted_price = self.predict_price(stock)
                print(f"{stock}: ${predicted_price:.6f}")

            # รอ 1 นาทีก่อนทำการอัพเดตข้อมูลอีกครั้ง
            time.sleep(60)


# รายชื่อหุ้นที่ต้องการ
stocks = ['AAPL', 'JPM', 'KO', 'CVX', 'AMZN', 'MSFT', 'PEP', 'O', 'AMT', 'BRK-B', 'GOOGL','TSLA','RTX']
news_api_key = '1516f93d42d9467bb86e0e0fa1b819cc'  # แทนที่ด้วย API Key ของคุณ

if __name__ == "__main__":
    analyzer = StockAnalyzer(stocks, news_api_key)
    analyzer.run()
