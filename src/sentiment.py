from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas as pd


def score_reviews(df, text_col='review'):
    """Compute sentiment compound score for each review using VADER."""
    analyzer = SentimentIntensityAnalyzer()
    scores = df[text_col].fillna('').astype(str).apply(lambda t: analyzer.polarity_scores(t)['compound'])
    # normalize to 0..1
    norm = (scores + 1) / 2
    df = df.copy()
    df['sentiment'] = norm
    return df


def agg_item_sentiment(df, item_col='item_id'):
    g = df.groupby(item_col).agg(
        mean_sentiment=('sentiment', 'mean'),
        review_count=( 'sentiment', 'count')
    ).reset_index()
    return g
