import pandas as pd
import requests
from dotenv import load_dotenv
import os

# Load environment variables for API token
load_dotenv()
API_TOKEN = os.getenv('HUGGINGFACE_API_TOKEN')

API_URL = "https://api-inference.huggingface.co/models/cardiffnlp/twitter-roberta-base-sentiment"
headers = {"Authorization": f"Bearer {API_TOKEN}"}

# Sentiment mapping
label_mapping = {
    "LABEL_0": "Negative",
    "LABEL_1": "Neutral",
    "LABEL_2": "Positive"
}
sentiment_score_mapping = {
    "Negative": -1,
    "Neutral": 0,
    "Positive": 1
}

# Sentiment analysis function
def analyze_sentiment(review_text):
    data = {"inputs": review_text}
    response = requests.post(API_URL, headers=headers, json=data)
    
    if response.status_code == 200:
        sentiment_result = response.json()
        if isinstance(sentiment_result, list) and len(sentiment_result) > 0:
            label_scores = sentiment_result[0]
            top_label = max(label_scores, key=lambda x: x['score'])
            return label_mapping.get(top_label['label'], 'Unknown')
    return "Unknown"

# Load reviews data from an Excel file
df = pd.read_excel("C:/Users/vgeddam/GenAI_files/reviews.xlsx")

# Apply sentiment analysis to the reviews
df['sentiment'] = df['review_text'].apply(analyze_sentiment)

# Map sentiments to numerical scores
df['sentiment_score'] = df['sentiment'].map(sentiment_score_mapping)

# Calculate overall sentiment score (mean of individual scores)
overall_sentiment_score = df['sentiment_score'].mean()

# Generate sentiment distribution
sentiment_counts = df['sentiment'].value_counts(normalize=True) * 100  # Percentage distribution

# Generate consolidated summary
if overall_sentiment_score > 0:
    overall_sentiment = "Positive"
elif overall_sentiment_score < 0:
    overall_sentiment = "Negative"
else:
    overall_sentiment = "Neutral"

consolidated_summary = {
    "Overall Sentiment Score": overall_sentiment_score,
    "Overall Sentiment": overall_sentiment,
    "Sentiment Distribution": sentiment_counts.to_dict(),   
}

print(f"Consolidated Sentiment Summary: {consolidated_summary}")

# Optionally, save results to CSV
df.to_csv('reviews_with_sentiments_and_scores.csv', index=False)
