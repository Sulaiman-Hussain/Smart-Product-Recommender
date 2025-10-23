import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from src.preprocess import load_reviews, extract_user_item_df, basic_clean
from src.sentiment import score_reviews, agg_item_sentiment
from src.recommender import train_svd, recommend_for_user

def plot_sentiment_dist(item_sent_df):
    """Plot distribution of sentiment scores across items."""
    fig = px.histogram(
        item_sent_df, 
        x='mean_sentiment',
        nbins=30,
        title='Distribution of Item Sentiment Scores',
        labels={'mean_sentiment': 'Average Sentiment Score', 'count': 'Number of Items'}
    )
    return fig

def plot_rating_dist(ratings_df):
    """Plot distribution of ratings."""
    ratings_count = ratings_df['rating'].value_counts().sort_index()
    fig = go.Figure(data=[
        go.Bar(x=ratings_count.index, y=ratings_count.values)
    ])
    fig.update_layout(
        title='Distribution of Ratings',
        xaxis_title='Rating',
        yaxis_title='Count'
    )
    return fig

def plot_sentiment_vs_reviews(item_sent_df):
    """Scatter plot of sentiment vs number of reviews."""
    fig = px.scatter(
        item_sent_df,
        x='review_count',
        y='mean_sentiment',
        title='Sentiment Score vs Number of Reviews',
        labels={
            'review_count': 'Number of Reviews',
            'mean_sentiment': 'Average Sentiment'
        }
    )
    return fig

@st.cache_data
def prepare(path):
    df = load_reviews(path)
    df['clean'] = df['review'].apply(basic_clean)
    items_df, ratings_df = extract_user_item_df(df)
    
    # Extract most common words for item titles
    from collections import Counter
    import re
    words = Counter()
    for review in df['clean']:
        words.update(w for w in review.split() if len(w) > 3)
    common_words = {i: f"Product {i} ({words.most_common(1000)[i][0]})" 
                   for i in range(1000)}
    
    # Score sentiments on items' representative reviews
    scored = score_reviews(ratings_df, text_col='review')
    item_sent = agg_item_sentiment(scored, item_col='item_id')
    
    # Add synthetic titles
    item_sent['title'] = item_sent['item_id'].map(lambda x: common_words.get(x, f"Product {x}"))
    
    algo = train_svd(ratings_df)
    return df, ratings_df, item_sent, algo, common_words


def main():
    st.title('Sentiment-weighted Product Recommender')
    st.write('Analysis of Product Reviews and Recommendations')
    st.write('This app analyzes product reviews, computes sentiment scores, and provides personalized recommendations.')
    st.write('Upload your reviews.txt file or use the default to get started.')

    # Sidebar for controls
    with st.sidebar:
        st.header('Controls')
        path = st.text_input('Path to reviews.txt', value='reviews.txt')
        if st.button('Prepare data', key='prepare'):
            with st.spinner('Loading and analyzing reviews...'):
                df, ratings_df, item_sent, algo, titles = prepare(path)
                st.session_state.update({
                    'df': df,
                    'ratings_df': ratings_df,
                    'item_sent': item_sent,
                    'algo': algo,
                    'titles': titles
                })
            st.success('Ready!')

        st.markdown('---')
        user_id = st.text_input('Enter user id (e.g. user_0)', key='user_id')
        n_recs = st.slider('Number of recommendations', 5, 20, 5)
        
    # Main content area
    if 'item_sent' not in st.session_state:
        st.info('👈 Start by clicking "Prepare data" in the sidebar')
        return

    # Tabs for different views
    tab1, tab2, tab3 = st.tabs(['Recommendations', 'Sentiment Analysis', 'Rating Patterns'])
    
    with tab1:
        if st.button('Get Recommendations', key='get_recs'):
            user = user_id.strip() or 'user_0'
            out = recommend_for_user(
                st.session_state['algo'], 
                user, 
                st.session_state['ratings_df'],
                st.session_state['item_sent'],
                top_n=n_recs
            )
            
            # Add titles and format
            out['title'] = out['item_id'].map(lambda x: st.session_state['titles'].get(x, f"Product {x}"))
            out['mean_sentiment'] = out['mean_sentiment'].round(3)
            out['final_score'] = out['final_score'].round(3)
            
            st.subheader(f'Top {n_recs} recommendations for {user}')
            
            # Display each recommendation with more detail
            for i, row in out.iterrows():
                with st.expander(f"{row['title']} (Score: {row['final_score']:.2f})"):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric('Predicted Rating', f"{row['est']:.1f}/5.0")
                    with col2:
                        st.metric('Sentiment Score', f"{row['mean_sentiment']:.2f}")
                    
                    # Show sample reviews for this item
                    reviews = st.session_state['ratings_df'][
                        st.session_state['ratings_df']['item_id'] == row['item_id']
                    ]['review'].head(3)
                    if not reviews.empty:
                        st.write('Sample Reviews:')
                        for r in reviews:
                            st.write(f"• {r[:200]}...")
    
    with tab2:
        st.subheader('Sentiment Analysis')
        col1, col2 = st.columns(2)
        
        with col1:
            # Distribution of sentiment scores
            st.plotly_chart(
                plot_sentiment_dist(st.session_state['item_sent']),
                use_container_width=True
            )
        
        with col2:
            # Sentiment vs review count
            st.plotly_chart(
                plot_sentiment_vs_reviews(st.session_state['item_sent']),
                use_container_width=True
            )
        
        # Top and bottom products by sentiment
        col3, col4 = st.columns(2)
        with col3:
            st.markdown('##### Highest Sentiment Scores')
            top = st.session_state['item_sent'].nlargest(10, 'mean_sentiment')
            top['mean_sentiment'] = top['mean_sentiment'].round(3)
            st.dataframe(
                top[['title', 'mean_sentiment', 'review_count']],
                hide_index=True
            )
        
        with col4:
            st.markdown('##### Lowest Sentiment Scores')
            bottom = st.session_state['item_sent'].nsmallest(10, 'mean_sentiment')
            bottom['mean_sentiment'] = bottom['mean_sentiment'].round(3)
            st.dataframe(
                bottom[['title', 'mean_sentiment', 'review_count']],
                hide_index=True
            )
    
    with tab3:
        st.subheader('Rating Distribution')
        st.plotly_chart(
            plot_rating_dist(st.session_state['ratings_df']),
            use_container_width=True
        )


if __name__ == '__main__':
    main()
