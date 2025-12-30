# sentiment_labeling.py
# Task 1: Improved Sentiment Labeling for professional emails

from transformers import pipeline
from tqdm import tqdm

print("Loading improved sentiment analysis model...")
# Better model for formal/professional text
sentiment_pipeline = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english",
    return_all_scores=True  # Get both scores to detect Neutral
)

def add_sentiment_column(df):
    """
    Adds 'sentiment' column: Positive, Negative, or Neutral
    Uses confidence threshold to detect Neutral
    """
    print("\n" + "="*70)
    print("TASK 1: IMPROVED SENTIMENT LABELING")
    print("="*70)

    # Safely convert to string
    subjects = df['subject'].fillna("").astype(str)
    bodies = df['body'].fillna("").astype(str)
    texts = (subjects + " " + bodies).str.strip()

    print(f"Analyzing {len(texts)} messages...")

    sentiments = []
    for text in tqdm(texts, desc="Sentiment Analysis", unit="msg"):
        if not text.strip():
            sentiments.append("Neutral")
            continue

        # Truncate long texts
        text = text[:512]

        try:
            results = sentiment_pipeline(text)[0]  # List of [{'label': 'POSITIVE', 'score': 0.99}, ...]
            pos_score = next(r['score'] for r in results if r['label'] == 'POSITIVE')
            neg_score = next(r['score'] for r in results if r['label'] == 'NEGATIVE')

            if pos_score > 0.7:
                sentiments.append("Positive")
            elif neg_score > 0.7:
                sentiments.append("Negative")
            else:
                sentiments.append("Neutral")  # Low confidence = Neutral
        except:
            sentiments.append("Neutral")

    df['sentiment'] = sentiments

    print("\nSentiment labeling completed!")
    print("\nFinal Sentiment Distribution:")
    print(df['sentiment'].value_counts())
    print("\nPercentage:")
    print((df['sentiment'].value_counts(normalize=True) * 100).round(2))

    return df