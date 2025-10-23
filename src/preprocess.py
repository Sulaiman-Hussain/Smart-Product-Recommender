import re
import pandas as pd
from nltk import download
from nltk.tokenize import word_tokenize

download('punkt')


def load_reviews(path):
    """Load reviews.txt which appears to be label-prefixed Amazon reviews.

    The file uses markers like '__label__1' and '__label__2' at the start of each review.
    We'll split on those markers and create a DataFrame with 'label' and 'review'.
    """
    with open(path, 'r', encoding='utf-8', errors='replace') as f:
        text = f.read()

    # Split keeping delimiter
    parts = re.split(r'(__label__\d+)', text)
    rows = []
    for i in range(1, len(parts), 2):
        label = parts[i].strip()
        review = parts[i+1].strip()
        # Some entries might be empty
        if review:
            rows.append({'label': label, 'review': review})

    df = pd.DataFrame(rows)
    # Normalize label to int
    df['label'] = df['label'].str.replace('__label__', '').astype(int)
    # Create a simple id field for items (we don't have product ids in the dataset). We'll cluster by review content.
    df['review_len'] = df['review'].str.len()
    return df


def basic_clean(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", ' ', text)
    text = re.sub(r"\s+", ' ', text).strip()
    return text


def extract_user_item_df(df, n_items=1000, seed=42):
    """Create synthetic user_id, item_id, rating dataset from reviews.

    Because the dataset doesn't include explicit user/item ids, we'll create items by
    clustering reviews into n_items groups using simple hashing and then simulate users
    by sampling.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.cluster import MiniBatchKMeans
    import numpy as np

    vect = TfidfVectorizer(max_features=5000, stop_words='english')
    X = vect.fit_transform(df['review'])

    kmeans = MiniBatchKMeans(n_clusters=min(n_items, max(2, X.shape[0]//5)), random_state=seed)
    labels = kmeans.fit_predict(X)
    df = df.copy()
    df['item_id'] = labels

    # Assign sentiment placeholder
    df['sentiment'] = None

    # Build user-item matrix by sampling
    np.random.seed(seed)
    n_users = 500
    users = [f'user_{i}' for i in range(n_users)]

    rows = []
    for u in users:
        # each user reviews between 5 and 40 items
        k = np.random.randint(5, 40)
        sampled = df.sample(n=min(k, len(df)))
        for _, r in sampled.iterrows():
            # Derive a rating from label: label 2 -> positive (4-5), label1 -> negative (1-2)
            if r['label'] == 2:
                rating = np.random.choice([4,5], p=[0.4,0.6])
            else:
                rating = np.random.choice([1,2], p=[0.7,0.3])
            rows.append({'user_id': u, 'item_id': int(r['item_id']), 'rating': rating, 'review': r['review']})

    ui = pd.DataFrame(rows)
    return df, ui
