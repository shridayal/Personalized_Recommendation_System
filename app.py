# app.py - Integrated Movie Recommendation System

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
from sklearn.decomposition import NMF
from sklearn.metrics import mean_squared_error, mean_absolute_error
import pickle
import os
import time
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Movie Recommendation System - Complete Pipeline", layout="wide")

# Initialize session state
if 'current_week' not in st.session_state:
    st.session_state.current_week = 1
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'models_trained' not in st.session_state:
    st.session_state.models_trained = False

# Title and description
st.title("🎬 Movie Recommendation System - Complete Pipeline")
st.markdown("### 4-Week Project Implementation")
st.markdown("---")

# Sidebar for navigation
with st.sidebar:
    st.header("Navigation")
    week_selection = st.radio(
        "Select Week",
        ["Week 1: Data & EDA", "Week 2: Collaborative Filtering", 
         "Week 3: Content-Based & Hybrid", "Week 4: Evaluation & Interface"]
    )
    
    st.markdown("---")
    st.markdown("### Project Status")
    progress = st.session_state.current_week / 4
    st.progress(progress)
    st.markdown(f"Progress: {progress*100:.0f}%")

# Helper functions from all weeks
@st.cache_data
def load_movielens_data():
    """Load MovieLens dataset"""
    try:
        ratings = pd.read_csv('ratings.csv')
        movies = pd.read_csv('movies.csv')
        tags = pd.read_csv('tags.csv')
        links = pd.read_csv('links.csv')
        return ratings, movies, tags, links
    except:
        st.error("Please ensure MovieLens CSV files are in the same directory")
        return None, None, None, None

def preprocess_data(ratings, movies):
    """Week 1: Data preprocessing"""
    ratings['timestamp'] = pd.to_datetime(ratings['timestamp'], unit='s')
    movies['year'] = movies['title'].str.extract(r'$$(\d{4})$$')
    movies['title_clean'] = movies['title'].str.replace(r'\s*$$\d{4}$$\s*$', '', regex=True)
    movies['genres_list'] = movies['genres'].str.split('|')
    
    ratings = ratings.drop_duplicates(['userId', 'movieId'])
    
    return ratings, movies

def create_user_item_matrix(ratings):
    """Create user-item interaction matrix"""
    return ratings.pivot_table(index='userId', columns='movieId', values='rating')

def user_based_cf(train_matrix, n_neighbors=50):
    """Week 2: User-based collaborative filtering"""
    filled_matrix = train_matrix.fillna(0)
    user_similarity = cosine_similarity(filled_matrix)
    user_similarity_df = pd.DataFrame(user_similarity, 
                                      index=train_matrix.index, 
                                      columns=train_matrix.index)
    
    predictions = pd.DataFrame(index=train_matrix.index, columns=train_matrix.columns)
    
    progress_bar = st.progress(0)
    total_users = len(train_matrix.index)
    
    for i, user in enumerate(train_matrix.index[:100]):  # Limit for demo
        similar_users = user_similarity_df[user].sort_values(ascending=False)[1:n_neighbors+1]
        
        for movie in train_matrix.columns:
            if pd.isna(train_matrix.loc[user, movie]):
                similar_users_ratings = train_matrix.loc[similar_users.index, movie]
                valid_ratings = similar_users_ratings.dropna()
                
                if len(valid_ratings) > 0:
                    weights = similar_users[valid_ratings.index]
                    weighted_sum = np.sum(weights * valid_ratings)
                    sum_weights = np.sum(weights)
                    
                    if sum_weights > 0:
                        predictions.loc[user, movie] = weighted_sum / sum_weights
        
        progress_bar.progress((i + 1) / min(100, total_users))
    
    progress_bar.empty()
    return predictions

def matrix_factorization_svd(train_matrix, n_factors=50):
    """Week 2: SVD for collaborative filtering"""
    filled_matrix = train_matrix.fillna(0)
    sparse_matrix = csr_matrix(filled_matrix.values)
    
    U, sigma, Vt = svds(sparse_matrix, k=n_factors)
    sigma = np.diag(sigma)
    
    predicted_ratings = np.dot(np.dot(U, sigma), Vt)
    predictions_df = pd.DataFrame(predicted_ratings, 
                                  index=train_matrix.index,
                                  columns=train_matrix.columns)
    
    return predictions_df

def create_content_features(movies, tags):
    """Week 3: Create content-based features"""
    movies['content'] = movies['title_clean'].fillna('') + ' ' + movies['genres'].fillna('')
    
    if 'year' in movies.columns:
        movies['decade'] = (movies['year'].astype(float) // 10 * 10).fillna(0).astype(int)
        movies['content'] = movies['content'] + ' decade' + movies['decade'].astype(str)
    
    if tags is not None:
        tags_grouped = tags.groupby('movieId')['tag'].apply(lambda x: ' '.join(x.dropna().astype(str))).reset_index()
        movies = movies.merge(tags_grouped, on='movieId', how='left')
        movies['content'] = movies['content'] + ' ' + movies['tag'].fillna('')
    
    return movies

def content_based_similarity(movies):
    """Week 3: Calculate content-based similarity"""
    tfidf = TfidfVectorizer(max_features=1000, stop_words='english', ngram_range=(1, 2))
    tfidf_matrix = tfidf.fit_transform(movies['content'])
    
    content_similarity = cosine_similarity(tfidf_matrix)
    content_similarity_df = pd.DataFrame(content_similarity, 
                                         index=movies['movieId'], 
                                         columns=movies['movieId'])
    
    return content_similarity_df, tfidf

def hybrid_recommendations(user_id, cf_predictions, content_similarity, user_ratings, movies_df, 
                          cf_weight=0.7, content_weight=0.3, n_recommendations=10):
    """Week 3: Hybrid recommendation system"""
    recommendations = pd.DataFrame()
    
    # Get CF scores
    if user_id in cf_predictions.index:
        cf_scores = cf_predictions.loc[user_id].dropna()
        if len(cf_scores) > 0:
            cf_scores = (cf_scores - cf_scores.min()) / (cf_scores.max() - cf_scores.min())
            recommendations['cf_score'] = cf_scores
    
    # Get content-based scores
    user_movies = user_ratings[user_ratings['userId'] == user_id]
    if len(user_movies) > 0:
        all_content_scores = pd.Series(dtype=float)
        
        for _, row in user_movies.sort_values('rating', ascending=False).head(5).iterrows():
            movie_id = row['movieId']
            if movie_id in content_similarity.index:
                sim_scores = content_similarity[movie_id] * (row['rating'] / 5.0)
                all_content_scores = all_content_scores.add(sim_scores, fill_value=0)
        
        if len(all_content_scores) > 0:
            all_content_scores = (all_content_scores - all_content_scores.min()) / (all_content_scores.max() - all_content_scores.min())
            for movie_id in all_content_scores.index:
                if movie_id not in recommendations.index:
                    recommendations.loc[movie_id, 'cf_score'] = 0
                recommendations.loc[movie_id, 'content_score'] = all_content_scores[movie_id]
    
    # Calculate hybrid score
    recommendations['content_score'] = recommendations['content_score'].fillna(0)
    recommendations['cf_score'] = recommendations['cf_score'].fillna(0)
    recommendations['hybrid_score'] = (cf_weight * recommendations['cf_score'] + 
                                       content_weight * recommendations['content_score'])
    
    # Filter out already watched movies
    user_watched = user_ratings[user_ratings['userId'] == user_id]['movieId'].values
    recommendations = recommendations[~recommendations.index.isin(user_watched)]
    
    # Get top recommendations
    top_recommendations = recommendations.sort_values('hybrid_score', ascending=False).head(n_recommendations)
    
    # Merge with movie details
    final_recs = movies_df[movies_df['movieId'].isin(top_recommendations.index)].copy()
    final_recs = final_recs.merge(top_recommendations, left_on='movieId', right_index=True)
    
    return final_recs[['movieId', 'title', 'genres', 'hybrid_score', 'cf_score', 'content_score']].sort_values('hybrid_score', ascending=False)

# Main content based on week selection
if "Week 1" in week_selection:
    st.header("Week 1: Data Collection & Exploratory Data Analysis")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if st.button("Load MovieLens Data", type="primary"):
            with st.spinner("Loading data..."):
                ratings, movies, tags, links = load_movielens_data()
                
                if ratings is not None:
                    st.session_state.ratings_raw = ratings
                    st.session_state.movies_raw = movies
                    st.session_state.tags = tags
                    st.session_state.links = links
                    st.session_state.data_loaded = True
                    st.success("Data loaded successfully!")
    
    if st.session_state.data_loaded:
        ratings = st.session_state.ratings_raw
        movies = st.session_state.movies_raw
        
        # Display basic statistics
        st.subheader("📊 Dataset Overview")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Ratings", f"{len(ratings):,}")
        with col2:
            st.metric("Unique Users", f"{ratings['userId'].nunique():,}")
        with col3:
            st.metric("Unique Movies", f"{ratings['movieId'].nunique():,}")
        with col4:
            sparsity = 1 - (len(ratings) / (ratings['userId'].nunique() * ratings['movieId'].nunique()))
            st.metric("Sparsity", f"{sparsity:.2%}")
        
        # Data preprocessing
        if st.button("Preprocess Data"):
            with st.spinner("Preprocessing..."):
                ratings_clean, movies_clean = preprocess_data(ratings, movies)
                st.session_state.ratings = ratings_clean
                st.session_state.movies = movies_clean
                st.success("Data preprocessing completed!")
                st.session_state.current_week = 1
        
        # Visualizations
        st.subheader("📈 Data Visualizations")
        
        tab1, tab2, tab3 = st.tabs(["Rating Distribution", "User Activity", "Genre Analysis"])
        
        with tab1:
            fig = px.histogram(ratings, x='rating', title='Distribution of Ratings',
                              labels={'rating': 'Rating', 'count': 'Count'})
            st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            ratings_per_user = ratings.groupby('userId').size().reset_index(name='count')
            fig = px.histogram(ratings_per_user, x='count', title='Ratings per User Distribution',
                              labels={'count': 'Number of Ratings'})
            fig.update_yaxis(type='log')
            st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            if 'movies' in st.session_state:
                movies_clean = st.session_state.movies
                all_genres = []
                for genres_list in movies_clean['genres_list'].dropna():
                    all_genres.extend(genres_list)
                genre_counts = pd.Series(all_genres).value_counts().head(15)
                
                fig = px.bar(x=genre_counts.values, y=genre_counts.index, orientation='h',
                            title='Top 15 Movie Genres', labels={'x': 'Count', 'y': 'Genre'})
                st.plotly_chart(fig, use_container_width=True)

elif "Week 2" in week_selection:
    st.header("Week 2: Collaborative Filtering Models")
    
    if not st.session_state.data_loaded:
        st.warning("Please load and preprocess data in Week 1 first!")
    else:
        if 'ratings' in st.session_state:
            ratings = st.session_state.ratings
            
            # Train-test split
            if st.button("Create Train-Test Split"):
                train_data, test_data = train_test_split(ratings, test_size=0.2, random_state=42)
                st.session_state.train_data = train_data
                st.session_state.test_data = test_data
                st.success(f"Train size: {len(train_data):,}, Test size: {len(test_data):,}")
            
            if 'train_data' in st.session_state:
                train_matrix = create_user_item_matrix(st.session_state.train_data)
                
                st.subheader("Select Model to Train")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    if st.button("Train User-Based CF", type="primary"):
                        with st.spinner("Training User-Based CF..."):
                            start_time = time.time()
                            user_based_pred = user_based_cf(train_matrix, n_neighbors=50)
                            st.session_state.user_based_predictions = user_based_pred
                
