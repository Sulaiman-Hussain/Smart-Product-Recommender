import numpy as np
import pandas as pd
from collections import defaultdict


class FunkSVD:
    """Simple FunkSVD implementation (stochastic gradient descent)"""
    def __init__(self, n_factors=20, lr=0.005, reg=0.02, n_epochs=20, random_state=42):
        self.n_factors = n_factors
        self.lr = lr
        self.reg = reg
        self.n_epochs = n_epochs
        self.random_state = random_state

    def fit(self, ratings_df):
        users = ratings_df['user_id'].unique().tolist()
        items = ratings_df['item_id'].unique().tolist()
        self.user_map = {u:i for i,u in enumerate(users)}
        self.item_map = {i:j for j,i in enumerate(items)}
        n_users = len(users)
        n_items = len(items)
        rng = np.random.default_rng(self.random_state)
        self.P = rng.normal(0, 0.1, size=(n_users, self.n_factors))
        self.Q = rng.normal(0, 0.1, size=(n_items, self.n_factors))

        # build training triples
        triples = []
        for _, r in ratings_df.iterrows():
            ui = self.user_map[r['user_id']]
            ii = self.item_map[int(r['item_id'])]
            triples.append((ui, ii, float(r['rating'])))

        for epoch in range(self.n_epochs):
            for (ui, ii, rating) in triples:
                pred = self.P[ui].dot(self.Q[ii])
                err = rating - pred
                # update
                self.P[ui] += self.lr * (err * self.Q[ii] - self.reg * self.P[ui])
                self.Q[ii] += self.lr * (err * self.P[ui] - self.reg * self.Q[ii])

    def predict(self, user_id, item_id):
        if user_id not in self.user_map or int(item_id) not in self.item_map:
            # cold start
            return 3.0
        ui = self.user_map[user_id]
        ii = self.item_map[int(item_id)]
        return float(self.P[ui].dot(self.Q[ii]))


def train_svd(ratings_df, n_factors=50):
    model = FunkSVD(n_factors=n_factors)
    model.fit(ratings_df)
    return model


def recommend_for_user(algo, user_id, ratings_df, item_sentiment_df, top_n=5):
    all_items = ratings_df['item_id'].unique()
    seen = set(ratings_df[ratings_df['user_id']==user_id]['item_id'].unique())
    candidates = [i for i in all_items if i not in seen]

    preds = []
    for item in candidates:
        est = algo.predict(user_id, item)
        preds.append((int(item), est))

    preds_df = pd.DataFrame(preds, columns=['item_id', 'est'])
    preds_df = preds_df.merge(item_sentiment_df, on='item_id', how='left')
    preds_df['mean_sentiment'] = preds_df['mean_sentiment'].fillna(0.5)
    preds_df['est_norm'] = (preds_df['est'] - 1) / 4.0
    preds_df['final_score'] = 0.7 * preds_df['est_norm'] + 0.3 * preds_df['mean_sentiment']
    out = preds_df.sort_values('final_score', ascending=False).head(top_n)
    return out
