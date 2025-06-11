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

def item_based_cf(train_matrix, n_neighbors=20):
    """Week 2: Item-based collaborative filtering"""
    filled_matrix = train_matrix.fillna(0)
    item_similarity = cosine_similarity(filled_matrix.T)
    item_similarity_df = pd.DataFrame(item_similarity,
                                      index=train_matrix.columns,
                                      columns=train_matrix.columns)
    
    predictions = pd.DataFrame(index=train_matrix.index, columns=train_matrix.columns)
    
    progress_bar = st.progress(0)
    total_users = len(train_matrix.index)
    
    for i, user in enumerate(train_matrix.index[:100]):  # Limit for demo
        user_ratings = train_matrix.loc[user].dropna()
        
        for movie in train_matrix.columns:
            if pd.isna(train_matrix.loc[user, movie]):
                similar_movies = item_similarity_df[movie].sort_values(ascending=False)[1:n_neighbors+1]
                similar_movies = similar_movies[similar_movies.index.isin(user_ratings.index)]
                
                if len(similar_movies) > 0:
                    weights = similar_movies.values
                    ratings = user_ratings[similar_movies.index].values
                    
                    weighted_sum = np.sum(weights * ratings)
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
    
    return predictions_df, U, sigma, Vt

def matrix_factorization_nmf(train_matrix, n_factors=50):
    """Week 2: NMF for collaborative filtering"""
    filled_matrix = train_matrix.fillna(0)
    
    model = NMF(n_components=n_factors, init='random', random_state=42, max_iter=200)
    W = model.fit_transform(filled_matrix)
    H = model.components_
    
    predicted_ratings = np.dot(W, H)
    predictions_df = pd.DataFrame(predicted_ratings,
                                  index=train_matrix.index,
                                  columns=train_matrix.columns)
    
    return predictions_df, model, W, H

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

def content_based_recommendations(movie_id, similarity_matrix, movies_df, n_recommendations=10):
    """Get content-based recommendations for a single movie"""
    if movie_id not in similarity_matrix.index:
        return None
    
    sim_scores = similarity_matrix[movie_id].sort_values(ascending=False)
    sim_scores = sim_scores[sim_scores.index != movie_id]
    
    top_movies = sim_scores.head(n_recommendations).index
    recommendations = movies_df[movies_df['movieId'].isin(top_movies)].copy()
    recommendations['similarity_score'] = sim_scores[top_movies].values
    
    return recommendations[['movieId', 'title', 'genres', 'similarity_score']].sort_values('similarity_score', ascending=False)

def content_based_user_recommendations(user_id, user_ratings, similarity_matrix, movies_df, n_recommendations=10):
    """Get content-based recommendations for a user"""
    user_movies = user_ratings[user_ratings['userId'] == user_id]
    if len(user_movies) == 0:
        return None
    
    user_movies = user_movies.sort_values('rating', ascending=False)
    
    all_recommendations = pd.DataFrame()
    
    for _, row in user_movies.head(5).iterrows():
        movie_id = row['movieId']
        movie_rating = row['rating']
        
        similar_movies = content_based_recommendations(movie_id, similarity_matrix, movies_df, n_recommendations=20)
        if similar_movies is not None:
            similar_movies['weighted_score'] = similar_movies['similarity_score'] * (movie_rating / 5.0)
            all_recommendations = pd.concat([all_recommendations, similar_movies])
    
    all_recommendations = all_recommendations[~all_recommendations['movieId'].isin(user_movies['movieId'])]
    
    final_recommendations = all_recommendations.groupby(['movieId', 'title', 'genres']).agg({
        'weighted_score': 'sum'
    }).reset_index()
    
    return final_recommendations.sort_values('weighted_score', ascending=False).head(n_recommendations)

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
    content_recs = content_based_user_recommendations(user_id, user_ratings, content_similarity, movies_df, n_recommendations=50)
    if content_recs is not None:
        content_scores = content_recs.set_index('movieId')['weighted_score']
        if len(content_scores) > 0:
            content_scores = (content_scores - content_scores.min()) / (content_scores.max() - content_scores.min())
            
            for movie_id in content_scores.index:
                if movie_id not in recommendations.index:
                    recommendations.loc[movie_id, 'cf_score'] = 0
                recommendations.loc[movie_id, 'content_score'] = content_scores[movie_id]
    
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

def evaluate_model(predictions, test_data, model_name):
    """Week 4: Evaluate model performance"""
    test_predictions = []
    test_actuals = []
    
    for _, row in test_data.iterrows():
        user_id = row['userId']
        movie_id = row['movieId']
        actual_rating = row['rating']
        
        if user_id in predictions.index and movie_id in predictions.columns:
            predicted_rating = predictions.loc[user_id, movie_id]
            if not pd.isna(predicted_rating):
                test_predictions.append(predicted_rating)
                test_actuals.append(actual_rating)
    
    if len(test_predictions) > 0:
        rmse = np.sqrt(mean_squared_error(test_actuals, test_predictions))
        mae = mean_absolute_error(test_actuals, test_predictions)
        coverage = len(test_predictions) / len(test_data)
        
        return {
            'Model': model_name,
            'RMSE': rmse,
            'MAE': mae,
            'Coverage': coverage,
            'Predictions': len(test_predictions)
        }
    else:
        return None

# Main content based on week selection
if "Week 1" in week_selection:
    st.header("Week 1: Data Collection & Exploratory Data Analysis")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if st.button("Load MovieLens Data", type="primary"):
            with st.spinner("Loading data..."):
                ratings, movies, tags,
