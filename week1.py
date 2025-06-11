import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

ratings = pd.read_csv('ratings.csv')
movies = pd.read_csv('movies.csv')
tags = pd.read_csv('tags.csv')
links = pd.read_csv('links.csv')

print("Dataset Shapes:")
print(f"Ratings: {ratings.shape}")
print(f"Movies: {movies.shape}")
print(f"Tags: {tags.shape}")
print(f"Links: {links.shape}")

print("\nRatings Data:")
print(ratings.head())
print("\nMovies Data:")
print(movies.head())

print("\nData Types:")
print(ratings.dtypes)
print("\nMissing Values:")
print(ratings.isnull().sum())
print(movies.isnull().sum())

ratings['timestamp'] = pd.to_datetime(ratings['timestamp'], unit='s')
movies['year'] = movies['title'].str.extract(r'$$(\d{4})$$')
movies['title_clean'] = movies['title'].str.replace(r'\s*$$\d{4}$$\s*$', '', regex=True)
movies['genres_list'] = movies['genres'].str.split('|')

print("\nDuplicate Ratings:")
print(ratings.duplicated(['userId', 'movieId']).sum())

ratings = ratings.drop_duplicates(['userId', 'movieId'])

print("\nRating Statistics:")
print(ratings['rating'].describe())

print(f"\nNumber of Unique Users: {ratings['userId'].nunique()}")
print(f"Number of Unique Movies: {ratings['movieId'].nunique()}")
print(f"Total Number of Ratings: {len(ratings)}")
print(f"Average Ratings per User: {len(ratings) / ratings['userId'].nunique():.2f}")
print(f"Average Ratings per Movie: {len(ratings) / ratings['movieId'].nunique():.2f}")

total_possible = ratings['userId'].nunique() * ratings['movieId'].nunique()
sparsity = 1 - (len(ratings) / total_possible)
print(f"Sparsity: {sparsity:.4%}")

plt.figure(figsize=(10, 6))
ratings['rating'].value_counts().sort_index().plot(kind='bar')
plt.title('Distribution of Ratings')
plt.xlabel('Rating')
plt.ylabel('Count')
plt.tight_layout()
plt.show()

ratings_per_user = ratings.groupby('userId').size()
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.hist(ratings_per_user, bins=50, edgecolor='black')
plt.title('Distribution of Ratings per User')
plt.xlabel('Number of Ratings')
plt.ylabel('Number of Users')
plt.yscale('log')

ratings_per_movie = ratings.groupby('movieId').size()
plt.subplot(1, 2, 2)
plt.hist(ratings_per_movie, bins=50, edgecolor='black')
plt.title('Distribution of Ratings per Movie')
plt.xlabel('Number of Ratings')
plt.ylabel('Number of Movies')
plt.yscale('log')
plt.tight_layout()
plt.show()

avg_rating_by_user = ratings.groupby('userId')['rating'].mean()
plt.figure(figsize=(10, 6))
plt.hist(avg_rating_by_user, bins=50, edgecolor='black')
plt.title('Distribution of Average Ratings by User')
plt.xlabel('Average Rating')
plt.ylabel('Number of Users')
plt.tight_layout()
plt.show()

avg_rating_by_movie = ratings.groupby('movieId')['rating'].mean()
rating_count_by_movie = ratings.groupby('movieId').size()
movie_stats = pd.DataFrame({
    'avg_rating': avg_rating_by_movie,
    'rating_count': rating_count_by_movie
})

plt.figure(figsize=(10, 6))
plt.scatter(movie_stats['rating_count'], movie_stats['avg_rating'], alpha=0.5)
plt.xscale('log')
plt.xlabel('Number of Ratings (log scale)')
plt.ylabel('Average Rating')
plt.title('Average Rating vs Number of Ratings per Movie')
plt.tight_layout()
plt.show()

top_rated_movies = ratings.groupby('movieId').agg({
    'rating': ['mean', 'count']
}).reset_index()
top_rated_movies.columns = ['movieId', 'avg_rating', 'rating_count']
top_rated_movies = top_rated_movies[top_rated_movies['rating_count'] >= 50]
top_rated_movies = top_rated_movies.sort_values('avg_rating', ascending=False).head(20)
top_rated_movies = top_rated_movies.merge(movies[['movieId', 'title']], on='movieId')
print("\nTop 20 Highest Rated Movies (min 50 ratings):")
print(top_rated_movies[['title', 'avg_rating', 'rating_count']])

most_rated_movies = ratings.groupby('movieId').size().reset_index(name='rating_count')
most_rated_movies = most_rated_movies.sort_values('rating_count', ascending=False).head(20)
most_rated_movies = most_rated_movies.merge(movies[['movieId', 'title']], on='movieId')
print("\nTop 20 Most Rated Movies:")
print(most_rated_movies[['title', 'rating_count']])

ratings['year'] = ratings['timestamp'].dt.year
ratings['month'] = ratings['timestamp'].dt.month
ratings['hour'] = ratings['timestamp'].dt.hour

plt.figure(figsize=(12, 6))
ratings_by_year = ratings.groupby('year').size()
plt.plot(ratings_by_year.index, ratings_by_year.values)
plt.title('Number of Ratings Over Years')
plt.xlabel('Year')
plt.ylabel('Number of Ratings')
plt.tight_layout()
plt.show()

all_genres = []
for genres in movies['genres_list']:
    if isinstance(genres, list):
        all_genres.extend(genres)
genre_counts = pd.Series(all_genres).value_counts()

plt.figure(figsize=(12, 8))
genre_counts.head(20).plot(kind='barh')
plt.title('Top 20 Movie Genres')
plt.xlabel('Number of Movies')
plt.tight_layout()
plt.show()

genre_dummies = movies['genres'].str.get_dummies('|')
genre_ratings = pd.concat([movies[['movieId']], genre_dummies], axis=1)
genre_ratings = genre_ratings.merge(ratings[['movieId', 'rating']], on='movieId')

genre_avg_ratings = {}
for genre in genre_dummies.columns:
    if genre != '(no genres listed)':
        avg_rating = genre_ratings[genre_ratings[genre] == 1]['rating'].mean()
        genre_avg_ratings[genre] = avg_rating

genre_avg_df = pd.DataFrame(list(genre_avg_ratings.items()), 
                           columns=['Genre', 'Average Rating'])
genre_avg_df = genre_avg_df.sort_values('Average Rating', ascending=False)

plt.figure(figsize=(12, 8))
plt.barh(genre_avg_df['Genre'], genre_avg_df['Average Rating'])
plt.xlabel('Average Rating')
plt.title('Average Rating by Genre')
plt.tight_layout()
plt.show()

user_stats = ratings.groupby('userId').agg({
    'rating': ['count', 'mean'],
    'movieId': 'nunique'
}).reset_index()
user_stats.columns = ['userId', 'rating_count', 'avg_rating', 'unique_movies']

movie_stats = ratings.groupby('movieId').agg({
    'rating': ['count', 'mean', 'std'],
    'userId': 'nunique'
}).reset_index()
movie_stats.columns = ['movieId', 'rating_count', 'avg_rating', 'rating_std', 'unique_users']

print("\nUser Statistics Summary:")
print(user_stats.describe())
print("\nMovie Statistics Summary:")
print(movie_stats.describe())

ratings_cleaned = ratings.copy()
movies_cleaned = movies.copy()

ratings_cleaned.to_csv('ratings_cleaned.csv', index=False)
movies_cleaned.to_csv('movies_cleaned.csv', index=False)
user_stats.to_csv('user_stats.csv', index=False)
movie_stats.to_csv('movie_stats.csv', index=False)

print("\nWeek 1 Data Preprocessing and EDA Complete!")
print(f"Cleaned ratings saved: {ratings_cleaned.shape}")
print(f"Cleaned movies saved: {movies_cleaned.shape}")
