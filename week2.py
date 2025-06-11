import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
from sklearn.decomposition import NMF
from sklearn.metrics import mean_squared_error, mean_absolute_error
import pickle
import time

ratings = pd.read_csv('ratings_cleaned.csv')
movies = pd.read_csv('movies_cleaned.csv')

train_data, test_data = train_test_split(ratings, test_size=0.2, random_state=42)
print(f"Train size: {len(train_data)}, Test size: {len(test_data)}")

train_matrix = train_data.pivot_table(index='userId', columns='movieId', values='rating')
test_matrix = test_data.pivot_table(index='userId', columns='movieId', values='rating')

def user_based_cf(train_matrix, n_neighbors=50):
    filled_matrix = train_matrix.fillna(0)
    user_similarity = cosine_similarity(filled_matrix)
    user_similarity_df = pd.DataFrame(user_similarity, 
                                      index=train_matrix.index, 
                                      columns=train_matrix.index)
    
    predictions = pd.DataFrame(index=train_matrix.index, columns=train_matrix.columns)
    
    for user in train_matrix.index:
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
    
    return predictions

print("Building User-Based Collaborative Filtering...")
start_time = time.time()
user_based_predictions = user_based_cf(train_matrix, n_neighbors=50)
print(f"User-Based CF completed in {time.time() - start_time:.2f} seconds")

def item_based_cf(train_matrix, n_neighbors=20):
    filled_matrix = train_matrix.fillna(0)
    item_similarity = cosine_similarity(filled_matrix.T)
    item_similarity_df = pd.DataFrame(item_similarity,
                                      index=train_matrix.columns,
                                      columns=train_matrix.columns)
    
    predictions = pd.DataFrame(index=train_matrix.index, columns=train_matrix.columns)
    
    for user in train_matrix.index:
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
    
    return predictions

print("\nBuilding Item-Based Collaborative Filtering...")
start_time = time.time()
item_based_predictions = item_based_cf(train_matrix, n_neighbors=20)
print(f"Item-Based CF completed in {time.time() - start_time:.2f} seconds")

def matrix_factorization_svd(train_matrix, n_factors=50):
    filled_matrix = train_matrix.fillna(0)
    
    sparse_matrix = csr_matrix(filled_matrix.values)
    
    U, sigma, Vt = svds(sparse_matrix, k=n_factors)
    
    sigma = np.diag(sigma)
    
    predicted_ratings = np.dot(np.dot(U, sigma), Vt)
    predictions_df = pd.DataFrame(predicted_ratings, 
                                  index=train_matrix.index,
                                  columns=train_matrix.columns)
    
    return predictions_df, U, sigma, Vt

print("\nBuilding SVD Model...")
start_time = time.time()
svd_predictions, U, sigma, Vt = matrix_factorization_svd(train_matrix, n_factors=50)
print(f"SVD completed in {time.time() - start_time:.2f} seconds")

def matrix_factorization_nmf(train_matrix, n_factors=50):
    filled_matrix = train_matrix.fillna(0)
    
    model = NMF(n_components=n_factors, init='random', random_state=42, max_iter=200)
    W = model.fit_transform(filled_matrix)
    H = model.components_
    
    predicted_ratings = np.dot(W, H)
    predictions_df = pd.DataFrame(predicted_ratings,
                                  index=train_matrix.index,
                                  columns=train_matrix.columns)
    
    return predictions_df, model, W, H

print("\nBuilding NMF Model...")
start_time = time.time()
nmf_predictions, nmf_model, W, H = matrix_factorization_nmf(train_matrix, n_factors=50)
print(f"NMF completed in {time.time() - start_time:.2f} seconds")

def evaluate_predictions(predictions, test_data):
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
        return rmse, mae, len(test_predictions)
    else:
        return None, None, 0

def precision_at_k(predictions, test_data, k=10, threshold=4):
    precisions = []
    
    users = test_data['userId'].unique()
    
    for user in users:
        if user not in predictions.index:
            continue
            
        user_predictions = predictions.loc[user].dropna().sort_values(ascending=False)
        top_k_items = user_predictions.head(k).index
        
        user_test = test_data[test_data['userId'] == user]
        relevant_items = user_test[user_test['rating'] >= threshold]['movieId'].values
        
        if len(relevant_items) > 0:
            hits = len(set(top_k_items) & set(relevant_items))
            precision = hits / k
            precisions.append(precision)
    
    return np.mean(precisions) if precisions else 0

print("\n" + "="*50)
print("EVALUATION RESULTS")
print("="*50)

models = {
    'User-Based CF': user_based_predictions,
    'Item-Based CF': item_based_predictions,
    'SVD': svd_predictions,
    'NMF': nmf_predictions
}

results = []
for name, predictions in models.items():
    rmse, mae, n_predictions = evaluate_predictions(predictions, test_data)
    precision = precision_at_k(predictions, test_data, k=10)
    
    results.append({
        'Model': name,
        'RMSE': rmse,
        'MAE': mae,
        'Precision@10': precision,
        'Predictions': n_predictions
    })
    
    print(f"\n{name}:")
    print(f"  RMSE: {rmse:.4f}" if rmse else "  RMSE: N/A")
    print(f"  MAE: {mae:.4f}" if mae else "  MAE: N/A")
    print(f"  Precision@10: {precision:.4f}")
    print(f"  Test predictions made: {n_predictions}")

results_df = pd.DataFrame(results)
results_df.to_csv('model_evaluation_results.csv', index=False)

print("\nSaving models...")
with open('user_based_cf_predictions.pkl', 'wb') as f:
    pickle.dump(user_based_predictions, f)

with open('item_based_cf_predictions.pkl', 'wb') as f:
    pickle.dump(item_based_predictions, f)

with open('svd_model.pkl', 'wb') as f:
    pickle.dump({'predictions': svd_predictions, 'U': U, 'sigma': sigma, 'Vt': Vt}, f)

with open('nmf_model.pkl', 'wb') as f:
    pickle.dump({'predictions': nmf_predictions, 'model': nmf_model, 'W': W, 'H': H}, f)

def get_recommendations(user_id, predictions, movies_df, n_recommendations=10):
    if user_id not in predictions.index:
        return None
    
    user_predictions = predictions.loc[user_id].dropna().sort_values(ascending=False)
    
    train_movies = train_data[train_data['userId'] == user_id]['movieId'].values
    recommendations = user_predictions[~user_predictions.index.isin(train_movies)].head(n_recommendations)
    
    recommended_movies = movies_df[movies_df['movieId'].isin(recommendations.index)]
    recommended_movies = recommended_movies.merge(
        recommendations.to_frame('predicted_rating'),
        left_on='movieId',
        right_index=True
    )
    
    return recommended_movies[['movieId', 'title', 'genres', 'predicted_rating']].sort_values('predicted_rating', ascending=False)

sample_user = train_data['userId'].iloc[0]
print(f"\n\nSample Recommendations for User {sample_user}:")
print("\nSVD Recommendations:")
svd_recs = get_recommendations(sample_user, svd_predictions, movies, n_recommendations=10)
print(svd_recs)

print("\nWeek 2 Complete! All models trained and evaluated.")
