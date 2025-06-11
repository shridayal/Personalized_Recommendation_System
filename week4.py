import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.metrics import mean_squared_error, mean_absolute_error
import streamlit as st
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go

ratings = pd.read_csv('ratings_cleaned.csv')
movies = pd.read_csv('movies_cleaned.csv')
test_data = pd.read_csv('test_data.csv') if 'test_data.csv' in os.listdir() else ratings.sample(frac=0.2, random_state=42)

with open('user_based_cf_predictions.pkl', 'rb') as f:
    user_based_predictions = pickle.load(f)
with open('item_based_cf_predictions.pkl', 'rb') as f:
    item_based_predictions = pickle.load(f)
with open('svd_model.pkl', 'rb') as f:
    svd_data = pickle.load(f)
    svd_predictions = svd_data['predictions']
with open('nmf_model.pkl', 'rb') as f:
    nmf_data = pickle.load(f)
    nmf_predictions = nmf_data['predictions']
with open('content_similarity_matrix.pkl', 'rb') as f:
    content_similarity_df = pickle.load(f)

def comprehensive_evaluation(predictions, test_data, model_name):
    results = {'model': model_name}
    
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
        results['rmse'] = np.sqrt(mean_squared_error(test_actuals, test_predictions))
        results['mae'] = mean_absolute_error(test_actuals, test_predictions)
        results['predictions_made'] = len(test_predictions)
        results['coverage'] = len(test_predictions) / len(test_data)
        
        precision_scores = []
        recall_scores = []
        for threshold in [3.5, 4.0, 4.5]:
            tp = sum((p >= threshold and a >= threshold) for p, a in zip(test_predictions, test_actuals))
            fp = sum((p >= threshold and a < threshold) for p, a in zip(test_predictions, test_actuals))
            fn = sum((p < threshold and a >= threshold) for p, a in zip(test_predictions, test_actuals))
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            
            results[f'precision@{threshold}'] = precision
            results[f'recall@{threshold}'] = recall
    
    return results

all_models = {
    'User-Based CF': user_based_predictions,
    'Item-Based CF': item_based_predictions,
    'SVD': svd_predictions,
    'NMF': nmf_predictions
}

evaluation_results = []
for model_name, predictions in all_models.items():
    results = comprehensive_evaluation(predictions, test_data, model_name)
    evaluation_results.append(results)

evaluation_df = pd.DataFrame(evaluation_results)
evaluation_df.to_csv('final_model_comparison.csv', index=False)

print("="*70)
print("FINAL MODEL COMPARISON RESULTS")
print("="*70)
print(evaluation_df.to_string(index=False))

fig, axes = plt.subplots(2, 2, figsize=(15, 12))

axes[0, 0].bar(evaluation_df['model'], evaluation_df['rmse'])
axes[0, 0].set_title('RMSE Comparison')
axes[0, 0].set_ylabel('RMSE')
axes[0, 0].tick_params(axis='x', rotation=45)

axes[0, 1].bar(evaluation_df['model'], evaluation_df['mae'])
axes[0, 1].set_title('MAE Comparison')
axes[0, 1].set_ylabel('MAE')
axes[0, 1].tick_params(axis='x', rotation=45)

axes[1, 0].bar(evaluation_df['model'], evaluation_df['coverage'])
axes[1, 0].set_title('Coverage Comparison')
axes[1, 0].set_ylabel('Coverage')
axes[1, 0].tick_params(axis='x', rotation=45)

precision_cols = [col for col in evaluation_df.columns if 'precision' in col]
precision_data = evaluation_df[['model'] + precision_cols].melt(id_vars='model', var_name='threshold', value_name='precision')
axes[1, 1].set_title('Precision at Different Thresholds')
for model in evaluation_df['model']:
    model_data = precision_data[precision_data['model'] == model]
    axes[1, 1].plot([3.5, 4.0, 4.5], model_data['precision'], marker='o', label=model)
axes[1, 1].set_xlabel('Threshold')
axes[1, 1].set_ylabel('Precision')
axes[1, 1].legend()

plt.tight_layout()
plt.savefig('model_comparison_charts.png', dpi=300, bbox_inches='tight')
plt.show()

def hybrid_recommendations_with_scores(user_id, cf_predictions, content_similarity, user_ratings, movies_df, 
                                       cf_weight=0.7, content_weight=0.3, n_recommendations=10):
    recommendations = pd.DataFrame()
    
    if user_id in cf_predictions.index:
        cf_scores = cf_predictions.loc[user_id].dropna()
        cf_scores = (cf_scores - cf_scores.min()) / (cf_scores.max() - cf_scores.min()) if len(cf_scores) > 0 else cf_scores
        recommendations['cf_score'] = cf_scores
    
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
    
    recommendations['content_score'] = recommendations['content_score'].fillna(0)
    recommendations['cf_score'] = recommendations['cf_score'].fillna(0)
    recommendations['hybrid_score'] = (cf_weight * recommendations['cf_score'] + content_weight * recommendations['content_score'])
    
    user_watched = user_ratings[user_ratings['userId'] == user_id]['movieId'].values
    recommendations = recommendations[~recommendations.index.isin(user_watched)]
    
    top_recommendations = recommendations.sort_values('hybrid_score', ascending=False).head(n_recommendations)
    
    final_recs = movies_df[movies_df['movieId'].isin(top_recommendations.index)].copy()
    final_recs = final_recs.merge(top_recommendations, left_on='movieId', right_index=True)
    
    return final_recs[['movieId', 'title', 'genres', 'hybrid_score', 'cf_score', 'content_score']].sort_values('hybrid_score', ascending=False)

hybrid_results = []
for cf_weight in np.arange(0, 1.1, 0.1):
    predictions = []
    actuals = []
    
    sample_users = test_data['userId'].unique()[:50]
    for user in sample_users:
        hybrid_recs = hybrid_recommendations_with_scores(user, svd_predictions, content_similarity_df, 
                                                        ratings, movies, cf_weight=cf_weight, 
                                                        content_weight=1-cf_weight, n_recommendations=10)
        
        if hybrid_recs is not None and len(hybrid_recs) > 0:
            user_test = test_data[test_data['userId'] == user]
            for movie_id in hybrid_recs['movieId']:
                test_rating = user_test[user_test['movieId'] == movie_id]['rating'].values
                if len(test_rating) > 0:
                    predictions.append(hybrid_recs[hybrid_recs['movieId'] == movie_id]['hybrid_score'].values[0] * 5)
                    actuals.append(test_rating[0])
    
    if len(predictions) > 0:
        rmse = np.sqrt(mean_squared_error(actuals, predictions))
        mae = mean_absolute_error(actuals, predictions)
        hybrid_results.append({'cf_weight': cf_weight, 'content_weight': 1-cf_weight, 'rmse': rmse, 'mae': mae})

hybrid_df = pd.DataFrame(hybrid_results)
best_weight = hybrid_df.loc[hybrid_df['rmse'].idxmin()]
print(f"\nOptimal Hybrid Weights: CF={best_weight['cf_weight']:.2f}, Content={best_weight['content_weight']:.2f}")

with open('streamlit_app.py', 'w') as f:
    f.write('''
import streamlit as st
import pandas as pd
import pickle
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Movie Recommendation System", layout="wide")

@st.cache_data
def load_data():
    ratings = pd.read_csv('ratings_cleaned.csv')
    movies = pd.read_csv('movies_cleaned.csv')
    
    with open('svd_model.pkl', 'rb') as f:
        svd_data = pickle.load(f)
        svd_predictions = svd_data['predictions']
    
    with open('content_similarity_matrix.pkl', 'rb') as f:
        content_similarity_df = pickle.load(f)
    
    return ratings, movies, svd_predictions, content_similarity_df

ratings, movies, svd_predictions, content_similarity_df = load_data()

st.title("🎬 Personalized Movie Recommendation System")
st.markdown("---")

tab1, tab2, tab3, tab4 = st.tabs(["Get Recommendations", "Model Performance", "Data Insights", "About"])

with tab1:
    st.header("Get Your Personalized Movie Recommendations")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        user_id = st.number_input("Enter User ID", min_value=1, max_value=ratings['userId'].max(), value=1)
        
        recommendation_type = st.selectbox(
            "Select Recommendation Type",
            ["Hybrid (Best Performance)", "Collaborative Filtering", "Content-Based"]
        )
        
        if recommendation_type == "Hybrid (Best Performance)":
            cf_weight = st.slider("Collaborative Filtering Weight", 0.0, 1.0, 0.7)
            content_weight = 1 - cf_weight
        
        n_recommendations = st.slider("Number of Recommendations", 5, 20, 10)
        
        if st.button("Get Recommendations", type="primary"):
            with st.spinner("Generating recommendations..."):
                if recommendation_type == "Hybrid (Best Performance)":
                    from week4_functions import hybrid_recommendations_with_scores
                    recommendations = hybrid_recommendations_with_scores(
                        user_id, svd_predictions, content_similarity_df, ratings, movies,
                        cf_weight=cf_weight, content_weight=content_weight, n_recommendations=n_recommendations
                    )
                
                with col2:
                    st.subheader(f"Top {n_recommendations} Recommendations for User {user_id}")
                    
                    if recommendations is not None and len(recommendations) > 0:
                        for idx, row in recommendations.iterrows():
                            with st.container():
                                st.markdown(f"**{idx+1}. {row['title']}**")
                                st.markdown(f"*Genres:* {row['genres']}")
                                st.markdown(f"*Score:* {row['hybrid_score']:.3f}")
                                st.progress(row['hybrid_score'])
                                st.markdown("---")
                    else:
                        st.warning("No recommendations available for this user.")

with tab2:
    st.header("Model Performance Comparison")
    
    evaluation_df = pd.read_csv('final_model_comparison.csv')
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig_rmse = px.bar(evaluation_df, x='model', y='rmse', title='RMSE by Model')
        st.plotly_chart(fig_rmse, use_container_width=True)
    
    with col2:
        fig_mae = px.bar(evaluation_df, x='model', y='mae', title='MAE by Model')
        st.plotly_chart(fig_mae, use_container_width=True)
    
    st.subheader("Detailed Performance Metrics")
    st.dataframe(evaluation_df, use_container_width=True)

with tab3:
    st.header("Dataset Insights")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Users", f"{ratings['userId'].nunique():,}")
        st.metric("Total Movies", f"{ratings['movieId'].nunique():,}")
    
    with col2:
        st.metric("Total Ratings", f"{len(ratings):,}")
        st.metric("Avg Rating", f"{ratings['rating'].mean():.2f}")
    
    with col3:
        sparsity = 1 - (len(ratings) / (ratings['userId'].nunique() * ratings['movieId'].nunique()))
        st.metric("Sparsity", f"{sparsity:.2%}")
        st.metric("Ratings per User", f"{len(ratings) / ratings['userId'].nunique():.1f}")
    
    st.subheader("Rating Distribution")
    fig_ratings = px.histogram(ratings, x='rating', title='Distribution of Ratings')
    st.plotly_chart(fig_ratings, use_container_width=True)

with tab4:
    st.header("About This Recommendation System")
    st.markdown("""
    ### Project Overview
    This movie recommendation system uses multiple approaches to provide personalized recommendations:
    
    - **Collaborative Filtering**: Finds patterns in user behavior
    - **Content-Based Filtering**: Recommends based on movie features
    - **Hybrid Approach**: Combines both methods for best results
    
    ### Technologies Used
    - Python, Pandas, NumPy, Scikit-learn
    - Matrix Factorization (SVD, NMF)
    - TF-IDF for content analysis
    - Streamlit for web interface
    
    ### Dataset
    MovieLens dataset with user ratings and movie metadata
    """)
''')

final_report = f"""
# Personalized Movie Recommendation System - Final Report

## Executive Summary
This project successfully implemented a comprehensive recommendation system using
