import pandas as pd
import requests
import yfinance as yf
import time
from datetime import datetime, timedelta
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


class StockPriceFetcher:
    def __init__(self, stocks):
        self.stocks = stocks

    def fetch_stock_prices(self):
        prices = {}
        for stock in self.stocks:
            ticker = yf.Ticker(stock)
            last_data = ticker.history(period='1d')
            prices[stock] = last_data['Close'].iloc[-1]  # ดึงราคาปิดล่าสุด
        return prices


class NewsFetcher:
    def __init__(self, api_key):
        self.api_key = api_key

    def fetch_news(self, stock):
        url = f"https://newsapi.org/v2/everything?q={stock}&apiKey={self.api_key}"
        response = requests.get(url)
        if response.status_code == 200:
            return response.json()['articles']
        else:
            print(f'Error fetching news for {stock}')
            return []


class SentimentAnalyzer:
    def __init__(self):
        self.analyzer = SentimentIntensityAnalyzer()

    def analyze_sentiment(self, articles):
        sentiments = []
        for article in articles:
            sentiment = self.analyzer.polarity_scores(article.get('content', ''))
            sentiments.append(sentiment)
        return sentiments


class StockSentimentApp:
    def __init__(self, stocks, news_api_key):
        self.stock_fetcher = StockPriceFetcher(stocks)
        self.news_fetcher = NewsFetcher(news_api_key)
        self.sentiment_analyzer = SentimentAnalyzer()
        self.next_sentiment_time = datetime.now()

    def run(self):
        while True:
            # ดึงราคาหุ้น
            prices = self.stock_fetcher.fetch_stock_prices()
            self.display_prices(prices)

            # อัพเดตการวิเคราะห์อารมณ์ทุกๆ 24 ชั่วโมง
            if datetime.now() >= self.next_sentiment_time:
                self.analyze_and_display_sentiment(prices.keys())
                self.next_sentiment_time = datetime.now() + timedelta(hours=24)

            # รอ 10 วินาทีก่อนทำการอัพเดตข้อมูลอีกครั้ง
            time.sleep(10)

    def display_prices(self, prices):
        print("\n### ราคาหุ้นปัจจุบัน ###")
        for stock, price in prices.items():
            print(f"{stock}: ${price:.6f}")

    def analyze_and_display_sentiment(self, stocks):
        all_sentiments = []
        for stock in stocks:
            news_articles = self.news_fetcher.fetch_news(stock)
            sentiments = self.sentiment_analyzer.analyze_sentiment(news_articles)
            avg_positive_sentiment = sum([s['pos'] for s in sentiments]) / len(sentiments) if sentiments else 0
            all_sentiments.append({'Stock': stock, 'Positive Sentiment': avg_positive_sentiment})

        sentiment_df = pd.DataFrame(all_sentiments)

        # แสดงผลการวิเคราะห์อารมณ์
        print("\n### สรุปผลการวิเคราะห์ Sentiment ของข่าวสาร ###")
        for index, row in sentiment_df.iterrows():
            print(f"{row['Stock']}: Positive Sentiment = {row['Positive Sentiment']:.6f}")


if __name__ == "__main__":
    stocks = ['AAPL', 'JPM', 'KO', 'JNJ', 'XOM', 'NEE', 'MSFT', 'PEP', 'O', 'AMT', 'BRK-B', 'GOOGL', 'HON' , 'TSLA']
    news_api_key = "27b1609a1d73443ebb7ce3e6e962ac1d"  # API Key ของคุณ
    app = StockSentimentApp(stocks, news_api_key)
    app.run()
