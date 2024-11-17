import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
import requests
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# สร้างวัตถุวิเคราะห์อารมณ์
analyzer = SentimentIntensityAnalyzer()

# ฟังก์ชันในการดึงข้อมูลราคาหุ้น
def fetch_stock_prices(stocks):
    prices = {}
    for stock in stocks:
        ticker = yf.Ticker(stock)
        last_data = ticker.history(period='1d')
        prices[stock] = last_data['Close'].iloc[-1]  # ดึงราคาปิดล่าสุด
    return prices

# ฟังก์ชันในการดึงข่าว
def fetch_news(stock):
    #url = f"https://newsapi.org/v2/everything?q={stock}&apiKey=8ff90c3729c04582990521080fdc29c9"
    url = f"https://newsapi.org/v2/everything?q={stock}&apiKey=27b1609a1d73443ebb7ce3e6e962ac1d"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()['articles']
    else:
        print(f'Error fetching news for {stock}')
        return []

# ฟังก์ชันในการวิเคราะห์อารมณ์
def analyze_sentiment(articles):
    sentiments = []
    for article in articles:
        sentiment = {'title': article['title'], 'sentiment': analyze(article.get('content', ''))}
        sentiments.append(sentiment)
    return sentiments

# ฟังก์ชันในการวิเคราะห์อารมณ์ (ใช้งาน VADER)
def analyze(content):
    if content:
        sentiment = analyzer.polarity_scores(content)
        return sentiment
    else:
        return {'neg': 0.0, 'neu': 1.0, 'pos': 0.0, 'compound': 0.0}  # หากไม่มีเนื้อหา

# รายชื่อหุ้นที่ต้องการ
stocks = ['AAPL', 'JPM', 'KO', 'JNJ', 'XOM', 'NEE', 'MSFT', 'PEP', 'O', 'AMT', 'BRK-B', 'GOOGL', 'HON' , 'TSLA']

# ดึงราคาหุ้น
prices = fetch_stock_prices(stocks)

# สร้าง DataFrame สำหรับราคาหุ้น
price_df = pd.DataFrame(prices.items(), columns=['Stock', 'Price'])

# สร้าง DataFrame สำหรับการวิเคราะห์อารมณ์ข่าว
all_sentiments = []
for stock in stocks:
    news_articles = fetch_news(stock)
    sentiments = analyze_sentiment(news_articles)
    for sentiment in sentiments:
        all_sentiments.append({'Stock': stock, 'Title': sentiment['title'], 'Sentiment': sentiment['sentiment']})

sentiment_df = pd.DataFrame(all_sentiments)

# บันทึก DataFrame ลงในไฟล์ CSV
price_df.to_csv('stock_prices.csv', index=False)
sentiment_df.to_csv('news_sentiments.csv', index=False)

# วิเคราะห์ sentiment
sentiment_df['Positive'] = sentiment_df['Sentiment'].apply(lambda x: x['pos'])

# วิเคราะห์และสรุป sentiment
sentiment_summary = sentiment_df.groupby('Stock')['Positive'].mean()

# แสดงผล
print("### สรุปผลการวิเคราะห์ Sentiment ของข่าวสาร ###")
for stock, sentiment in sentiment_summary.items():
    print(f"{stock}: Positive Sentiment = {sentiment:.4f}")

# สร้างกราฟด้วย Seaborn
plt.figure(figsize=(12, 10))

# กราฟ Sentiment
plt.subplot(2, 1, 1)  # 2 แถว 1 คอลัมน์ กราฟที่ 1
sns.barplot(x=sentiment_summary.index, y=sentiment_summary.values, palette='Blues_d')
plt.title('Mean Positive Sentiment by Stock', fontsize=16)
plt.xlabel('Stock', fontsize=12)
plt.ylabel('Mean Positive Sentiment', fontsize=12)
plt.xticks(rotation=45)
plt.grid(axis='y')

# กราฟราคาหุ้น
plt.subplot(2, 1, 2)  # 2 แถว 1 คอลัมน์ กราฟที่ 2
sns.barplot(x=price_df['Stock'], y=price_df['Price'], palette='Greens_d')
plt.title('Current Stock Prices', fontsize=16)
plt.xlabel('Stock', fontsize=12)
plt.ylabel('Price', fontsize=12)
plt.xticks(rotation=45)

# เพิ่มราคาบนกราฟ
for i in range(len(price_df)):
    plt.text(i, price_df['Price'][i], f"{price_df['Price'][i]:.2f}", ha='center', va='bottom', fontsize=10)

# ปรับความละเอียดของแกน Y
plt.yticks(fontsize=10)  # ปรับขนาดตัวอักษรของ Y-axis
plt.grid(axis='y')  # เพิ่มตารางบนแกน Y

plt.tight_layout()
plt.show()
