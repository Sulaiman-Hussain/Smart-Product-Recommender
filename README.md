# Sentiment-weighted Product Recommender

This project demonstrates a small pipeline that:

- Loads Amazon-style reviews from `reviews.txt` (labelled with `__label__1`/`__label__2`).
- Preprocesses and clusters reviews into synthetic items.
- Creates a synthetic user-item-rating dataset by sampling reviews per user.
- Computes sentiment scores using VADER and aggregates per-item sentiment.
- Trains a Surprise SVD model for collaborative filtering.
- Exposes a Streamlit dashboard (`app.py`) to query a user and get top-5 recommendations weighted by sentiment.

How to run

1. Create a virtual environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Run Streamlit:

```bash
streamlit run app.py
```

Notes

- The included dataset `reviews.txt` is large; the code samples the file and synthesizes user/item ids because the original dataset doesn't include explicit user/item identifiers.
- VADER is used for speed and simplicity. 

# Smart-Product-Recommender
