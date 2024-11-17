import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import requests

# ฟังก์ชันในการดึงข้อมูลราคาหุ้น
def fetch_stock_prices(stocks):
    prices = {}
    for stock in stocks:
        ticker = yf.Ticker(stock)
        prices[stock] = ticker.history(period='1d')['Close'].iloc[-1]  # ดึงราคาปิดล่าสุด
    return prices

# ฟังก์ชันในการดึงข่าว
def fetch_news(stock):
    url = f"https://newsapi.org/v2/everything?q={stock}&apiKey=8ff90c3729c04582990521080fdc29c9"
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
        # ใช้โมเดลการวิเคราะห์อารมณ์ เช่น VADER หรือ TextBlob ที่นี่
        sentiment = {'title': article['title'], 'sentiment': analyze(article.get('content', ''))}
        sentiments.append(sentiment)
    return sentiments

# ฟังก์ชันในการวิเคราะห์อารมณ์ (ตัวอย่าง)
def analyze(content):
    # แทนที่ด้วยการวิเคราะห์อารมณ์จริง
    return {'neg': 0.0, 'neu': 1.0, 'pos': 0.0, 'compound': 0.0}  # แค่ตัวอย่าง

# รายชื่อหุ้นที่ต้องการ
stocks = ['AAPL', 'JPM', 'KO', 'JNJ', 'XOM', 'NEE', 'MSFT', 'PEP', 'O', 'AMT', 'GOOGL', 'HON', 'BRK-B','TSLA']

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
# ไม่จำเป็นต้องใช้ ast.literal_eval เนื่องจาก Sentiment เป็น dictionary อยู่แล้ว
sentiment_df['Positive'] = sentiment_df['Sentiment'].apply(lambda x: x['pos'])

# วิเคราะห์และสรุป sentiment
sentiment_summary = sentiment_df.groupby('Stock')['Positive'].mean()

# แสดงผล
print("### สรุปผลการวิเคราะห์ Sentiment ของข่าวสาร ###")
for stock, sentiment in sentiment_summary.items():
    print(f"{stock}: Positive Sentiment = {sentiment:.4f}")

# สร้างกราฟ
plt.figure(figsize=(10, 5))
sentiment_summary.plot(kind='bar', color='skyblue')
plt.title('Mean Positive Sentiment by Stock')
plt.xlabel('Stock')
plt.ylabel('Mean Positive Sentiment')
plt.xticks(rotation=45)
plt.grid(axis='y')
plt.tight_layout()
plt.show()
