import requests
import time
import pandas as pd


API_TOKEN = "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" 

headers = {
    "Authorization": f"Bearer {API_TOKEN}"
}

# Model URL for Twitter sentiment analysis
API_URL = "https://api-inference.huggingface.co/models/cardiffnlp/twitter-roberta-base-sentiment"

# Label mapping for user-friendly output
label_mapping = {
    "LABEL_0": "NEGATIVE",
    "LABEL_1": "NEUTRAL",
    "LABEL_2": "POSITIVE"
}

def query(review):
    response = requests.post(API_URL, headers=headers, json={"inputs": review})
    return response.json()

def classify_sentiment(review):
    for attempt in range(5):  # Try up to 5 times
        result = query(review)
        
        # Handle loading state
        if isinstance(result, dict) and 'error' in result:
            if "loading" in result['error']:
                time.sleep(attempt + 1)  # Wait longer with each attempt
                continue

        # Process valid response
        if isinstance(result, list) and isinstance(result[0], list):
            sentiment_label = result[0][0]['label']
            confidence = result[0][0]['score']
            sentiment = label_mapping.get(sentiment_label, sentiment_label)
            
            # Adjust for low-confidence positives
            if sentiment == 'POSITIVE' and confidence < 0.7:
                sentiment = 'NEUTRAL'
            return review, sentiment, confidence
        
    return review, "ERROR", "Max retries exceeded"


reviews = [
    "It’s decent, not great but not terrible either.",
    "It’s fine for the price, but I've seen better.",
    "Not happy with it, I wouldn’t buy this again.",
    "Terrible experience, this product is not worth it at all.",
    # "I recently purchased this product and couldn't be happier with my experience. Right from the start, everything exceeded my expectations. The packaging was immaculate, and the product arrived exactly when promised. Setting it up was a breeze, with clear instructions that guided me through every step. Once I started using it, I was blown away by the quality and attention to detail. It's evident that a lot of thought went into the design – it's sleek, modern, and looks fantastic in my space. Functionality-wise, it performs flawlessly, delivering on every promise made by the manufacturer. I also had to reach out to customer service with a few questions, and they were incredibly responsive and helpful, resolving my issues quickly. This product has genuinely made a difference in my daily life, and I would highly recommend it to anyone looking for something reliable, high-quality, and easy to use!"
]

# Run sentiment analysis on each review
results = []
sentiment_counts = {"POSITIVE": 0, "NEUTRAL": 0, "NEGATIVE": 0}

for review in reviews:
    review_text, sentiment, confidence = classify_sentiment(review)
    
    if sentiment == "ERROR":
        print(f"Error for review: '{review_text}' - Response: {confidence}\n")
    else:
        print(f"Review: '{review_text}'\nSentiment: {sentiment}, Confidence: {confidence:.2f}\n")
        results.append([review_text, sentiment, confidence])
        
        # Update sentiment counts
        if sentiment in sentiment_counts:
            sentiment_counts[sentiment] += 1

# Calculate overall sentiment summary
total_reviews = sum(sentiment_counts.values())
overall_sentiment = max(sentiment_counts, key=sentiment_counts.get)  # Majority sentiment

print("\nOverall Sentiment Summary:")
print(f"Total Reviews: {total_reviews}")
print(f"Positive: {sentiment_counts['POSITIVE']}, Neutral: {sentiment_counts['NEUTRAL']}, Negative: {sentiment_counts['NEGATIVE']}")
print(f"Majority Sentiment: {overall_sentiment}")


# df_results = pd.DataFrame(results, columns=["Review", "Sentiment", "Confidence"])
# print("\nDetailed Results:")
# print(df_results)
