import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
import pickle
import warnings
warnings.filterwarnings('ignore')

ratings = pd.read_csv('ratings_cleaned.csv')
movies = pd.read_csv('movies_cleaned.csv')
train_data = pd.read_csv('train_data.csv') if 'train_data.csv' in os.listdir() else ratings.sample(frac=0.8, random_state=42)
test_data = pd.read_csv('test_data.csv') if 'test_data.csv' in os.listdir() else ratings.drop(train_data.index)

with open('svd_model.pkl', 'rb') as f:
    svd_data = pickle.load(f)
    svd_predictions = svd_data['predictions']

movies['content'] = movies['title_clean'].fillna('') + ' ' + movies['genres'].fillna('')
movies['content'] = movies['content'].str.lower()

if 'year' in movies.columns:
    movies['decade'] = (movies['year'].astype(float) // 10 * 10).fillna(0).astype(int)
    movies['content'] = movies['content'] + ' decade' + movies['decade'].astype(str)

tags_grouped = pd.read_csv('tags.csv').groupby('movieId')['tag'].apply(lambda x: ' '.join(x.dropna().astype(str))).reset_index()
movies = movies.merge(tags_grouped, on='movieId', how='left')
movies['content'] = movies['content'] + ' ' + movies['tag'].fillna('')

tfidf = TfidfVectorizer(max_features=1000, stop_words='english', ngram_range=(1, 2))
tfidf_matrix = tfidf.fit_transform(movies['content'])

content_similarity = cosine_similarity(tfidf_matrix)
content_similarity_df = pd.DataFrame(content_similarity, 
                                     index=movies['movieId'], 
                                     columns=movies['movieId'])

def content_based_recommendations(movie_id, similarity_matrix, movies_df, n_recommendations=10):
    if movie_id not in similarity_matrix.index:
        return None
    
    sim_scores = similarity_matrix[movie_id].sort_values(ascending=False)
    sim_scores = sim_scores[sim_scores.index != movie_id]
    
    top_movies = sim_scores.head(n_recommendations).index
    recommendations = movies_df[movies_df['movieId'].isin(top_movies)].copy()
    recommendations['similarity_score'] = sim_scores[top_movies].values
    
    return recommendations[['movieId', 'title', 'genres', 'similarity_score']].sort_values('similarity_score', ascending=False)

def content_based_user_recommendations(user_id, user_ratings, similarity_matrix, movies_df, n_recommendations=10):
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

sample_user = train_data['userId'].iloc[0]
content_recs = content_based_user_recommendations(sample_user, train_data, content_similarity_df, movies)
print(f"Content-Based Recommendations for User {sample_user}:")
print(content_recs)

def hybrid_recommendations(user_id, cf_predictions, content_similarity, user_ratings, movies_df, 
                          cf_weight=0.7, content_weight=0.3, n_recommendations=10):
    
    recommendations = pd.DataFrame()
    
    if user_id in cf_predictions.index:
        cf_scores = cf_predictions.loc[user_id].dropna()
        cf_scores = (cf_scores - cf_scores.min()) / (cf_scores.max() - cf_scores.min())
        recommendations['cf_score'] = cf_scores
    
    content_recs = content_based_user_recommendations(user_id, user_ratings, content_similarity, movies_df, n_recommendations=50)
    if content_recs is not None:
        content_scores = content_recs.set_index('movieId')['weighted_score']
        content_scores = (content_scores - content_scores.min()) / (content_scores.max() - content_scores.min())
        
        for movie_id in content_scores.index:
            if movie_id not in recommendations.index:
                recommendations.loc[movie_id, 'cf_score'] = 0
            recommendations.loc[movie_id, 'content_score'] = content_scores[movie_id]
    
    recommendations['content_score'] = recommendations['content_score'].fillna(0)
    recommendations['cf_score'] = recommendations['cf_score'].fillna(0)
    
    recommendations['hybrid_score'] = (cf_weight * recommendations['cf_score'] + 
                                       content_weight * recommendations['content_score'])
    
    user_watched = user_ratings[user_ratings['userId'] == user_id]['movieId'].values
    recommendations = recommendations[~recommendations.index.isin(user_watched)]
    
    top_recommendations = recommendations.sort_values('hybrid_score', ascending=False).head(n_recommendations)
    
    final_recs = movies_df[movies_df['movieId'].isin(top_recommendations.index)].copy()
    final_recs = final_recs.merge(top_recommendations[['hybrid_score', 'cf_score', 'content_score']], 
                                  left_on='movieId', right_index=True)
    
    return final_recs[['movieId', 'title', 'genres', 'hybrid_score', 'cf_score', 'content_score']].sort_values('hybrid_score', ascending=False)

print(f"\nHybrid Recommendations for User {sample_user}:")
hybrid_recs = hybrid_recommendations(sample_user, svd_predictions, content_similarity_df, train_data, movies)
print(hybrid_recs)

class AdaptiveRecommendationSystem:
    def __init__(self, initial_cf_weight=0.7):
        self.cf_weight = initial_cf_weight
        self.content_weight = 1 - initial_cf_weight
        self.user_preferences = {}
        self.feedback_history = []
    
    def get_recommendations(self, user_id, cf_predictions, content_similarity, user_ratings, movies_df, n_recommendations=10):
        if user_id in self.user_preferences:
            cf_weight = self.user_preferences[user_id]['cf_weight']
            content_weight = self.user_preferences[user_id]['content_weight']
        else:
            cf_weight = self.cf_weight
            content_weight = self.content_weight
        
        return hybrid_recommendations(user_id, cf_predictions, content_similarity, 
                                     user_ratings, movies_df, cf_weight, content_weight, n_recommendations)
    
    def update_weights(self, user_id, feedback):
        if user_id not in self.user_preferences:
            self.user_preferences[user_id] = {
                'cf_weight': self.cf_weight,
                'content_weight': self.content_weight,
                'feedback_count': 0
            }
        
        cf_performance = np.mean([f['cf_score'] for f in feedback if f['liked']])
        content_performance = np.mean([f['content_score'] for f in feedback if f['liked']])
        
        total_performance = cf_performance + content_performance
        if total_performance > 0:
            new_cf_weight = cf_performance / total_performance
            new_content_weight = content_performance / total_performance
            
            self.user_preferences[user_id]['cf_weight'] = 0.8 * self.user_preferences[user_id]['cf_weight'] + 0.2 * new_cf_weight
            self.user_preferences[user_id]['content_weight'] = 1 - self.user_preferences[user_id]['cf_weight']
            self.user_preferences[user_id]['feedback_count'] += len(feedback)
        
        self.feedback_history.extend(feedback)
    
    def simulate_feedback(self, recommendations, user_ratings):
        feedback = []
        for _, rec in recommendations.iterrows():
            actual_rating = user_ratings[user_ratings['movieId'] == rec['movieId']]['rating'].values
            if len(actual_rating) > 0:
                liked = actual_rating[0] >= 4
            else:
                liked = np.random.random() > 0.5
            
            feedback.append({
                'movie_id': rec['movieId'],
                'cf_score': rec['cf_score'],
                'content_score': rec['content_score'],
                'liked': liked
            })
        return feedback

adaptive_system = AdaptiveRecommendationSystem(initial_cf_weight=0.7)

print("\nTesting Adaptive System with User Feedback Simulation...")
for i in range(3):
    print(f"\nIteration {i+1}:")
    
    recs = adaptive_system.get_recommendations(sample_user, svd_predictions, content_similarity_df, train_data, movies)
    
    print(f"Current weights - CF: {adaptive_system.user_preferences.get(sample_user, {'cf_weight': 0.7})['cf_weight']:.3f}, "
          f"Content: {adaptive_system.user_preferences.get(sample_user, {'content_weight': 0.3})['content_weight']:.3f}")
    
    feedback = adaptive_system.simulate_feedback(recs.head(5), test_data)
    adaptive_system.update_weights(sample_user, feedback)

def evaluate_content_based(test_data, train_data, content_similarity, movies_df):
    precisions = []
    users = test_data['userId'].unique()[:100]
    
    for user in users:
        recs = content_based_user_recommendations(user, train_data, content_similarity, movies_df, n_recommendations=10)
        if recs is not None:
            test_movies = test_data[(test_data['userId'] == user) & (test_data['rating'] >= 4)]['movieId'].values
            if len(test_movies) > 0:
                hits = len(set(recs['movieId'].values) & set(test_movies))
                precision = hits / len(recs)
                precisions.append(precision)
    
    return np.mean(precisions) if precisions else 0

def evaluate_hybrid(test_data, train_data, cf_predictions, content_similarity, movies_df, cf_weight=0.7):
    precisions = []
    users = test_data['userId'].unique()[:100]
    
    for user in users:
        recs = hybrid_recommendations(user, cf_predictions, content_similarity, train_data, movies_df, 
                                     cf_weight=cf_weight, content_weight=1-cf_weight)
        if recs is not None and len(recs) > 0:
            test_movies = test_data[(test_data['userId'] == user) & (test_data['rating'] >= 4)]['movieId'].values
            if len(test_movies) > 0:
                hits = len(set(recs['movieId'].values) & set(test_movies))
                precision = hits / len(recs)
                precisions.append(precision)
    
    return np.mean(precisions) if precisions else 0

print("\n" + "="*50)
print("HYBRID SYSTEM EVALUATION")
print("="*50)

content_precision = evaluate_content_based(test_data, train_data, content_similarity_df, movies)
print(f"Content-Based Precision@10: {content_precision:.4f}")

weight_results = []
for cf_weight in [0.3, 0.5, 0.7, 0.9]:
    precision = evaluate_hybrid(test_data, train_data, svd_predictions, content_similarity_df, movies, cf_weight)
    weight_results.append({'cf_weight': cf_weight, 'content_weight': 1-cf_weight, 'precision': precision})
    print(f"Hybrid (CF: {cf_weight}, Content: {1-cf_weight}) Precision@10: {precision:.4f}")

weight_results_df = pd.DataFrame(weight_results)
weight_results_df.to_csv('hybrid_weight_evaluation.csv', index=False)

with open('content_similarity_matrix.pkl', 'wb') as f:
    pickle.dump(content_similarity_df, f)

with open('tfidf_vectorizer.pkl', 'wb') as f:
    pickle.dump(tfidf, f)

with open('adaptive_system.pkl', 'wb') as f:
    pickle.dump(adaptive_system, f)

print("\nWeek 3 Complete! Content-based and hybrid systems implemented.")
