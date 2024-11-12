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
LABEL_MAPPING = {
    "LABEL_0": "Negative",
    "LABEL_1": "Neutral",
    "LABEL_2": "Positive"
}

SENTIMENT_SCORE_MAPPING = {
    "Negative": -1,
    "Neutral": 0,
    "Positive": 1
}


def analyze_sentiment(review_text):
    """Analyze sentiment of a given review text."""
    data = {"inputs": review_text}
    response = requests.post(API_URL, headers=headers, json=data)
    
    if response.status_code == 200:
        sentiment_result = response.json()
        if isinstance(sentiment_result, list) and len(sentiment_result) > 0:
            label_scores = sentiment_result[0]
            top_label = max(label_scores, key=lambda x: x['score'])
            return LABEL_MAPPING.get(top_label['label'], 'Unknown')
    return "Unknown"


def load_reviews(file_path):
    """Load product reviews data from an Excel file."""
    return pd.read_excel(file_path)


def process_reviews(df):
    """Apply sentiment analysis and calculate sentiment scores."""
    df['sentiment'] = df['review_text'].apply(analyze_sentiment)
    df['sentiment_score'] = df['sentiment'].map(SENTIMENT_SCORE_MAPPING)
    return df


def calculate_overall_sentiment_score(df):
    """Calculate the overall sentiment score based on individual review scores."""
    return df['sentiment_score'].mean()


def generate_sentiment_distribution(df):
    """Generate percentage distribution of sentiments."""
    return df['sentiment'].value_counts(normalize=True) * 100


def generate_consolidated_summary(overall_sentiment_score, sentiment_counts):
    """Generate the consolidated sentiment summary."""
    if overall_sentiment_score > 0:
        overall_sentiment = "Positive"
    elif overall_sentiment_score < 0:
        overall_sentiment = "Negative"
    else:
        overall_sentiment = "Neutral"

    summary = {
        "Overall Sentiment": overall_sentiment,
        "Overall Sentiment Score": overall_sentiment_score,
        "Sentiment Distribution": {
            "Negative": sentiment_counts.get('Negative', 0),
            "Neutral": sentiment_counts.get('Neutral', 0),
            "Positive": sentiment_counts.get('Positive', 0)
        }
    }
    
    return summary


def print_summary(summary):
    """Print the consolidated sentiment summary in a readable format."""
    print("Consolidated Summary:")
    print(f"  Overall Sentiment: {summary['Overall Sentiment']}")
    print(f"  Overall Sentiment Score: {summary['Overall Sentiment Score']:.2f}")
    print("  Sentiment Distribution:")
    print(f"    Negative: {summary['Sentiment Distribution']['Negative']:.2f}%")
    print(f"    Neutral: {summary['Sentiment Distribution']['Neutral']:.2f}%")
    print(f"    Positive: {summary['Sentiment Distribution']['Positive']:.2f}%")


def save_to_csv(df, output_file):
    """Save the reviews with sentiment scores to a CSV file."""
    df.to_csv(output_file, index=False)


def main():
    # Load reviews data
    file_path = "C:/Users/vgeddam/GenAI_files/reviews.xlsx"
    reviews_df = load_reviews(file_path)

    # Process reviews and perform sentiment analysis
    reviews_df = process_reviews(reviews_df)

    # Calculate overall sentiment score
    overall_sentiment_score = calculate_overall_sentiment_score(reviews_df)

    # Generate sentiment distribution
    sentiment_counts = generate_sentiment_distribution(reviews_df)

    # Generate the consolidated summary
    summary = generate_consolidated_summary(overall_sentiment_score, sentiment_counts)

    # Print the summary
    print_summary(summary)

    # Optionally, save results to CSV
    save_to_csv(reviews_df, 'reviews_with_sentiments_and_scores.csv')


main()
