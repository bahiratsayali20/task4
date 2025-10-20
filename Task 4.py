import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

class MovieRecommender:
    def __init__(self):
        # Sample data
        self.ratings_data = {
            'User': ['A', 'A', 'A', 'B', 'B', 'C', 'C', 'D'],
            'Movie': ['Inception', 'Avatar', 'Titanic', 'Avatar', 'Titanic', 'Inception', 'Avengers', 'Avengers'],
            'Rating': [5, 4, 3, 5, 4, 4, 5, 3]
        }
        
        self.movies_data = {
            'Title': ['Inception', 'Avatar', 'Titanic', 'Avengers', 'Interstellar'],
            'Description': [
                'dream within a dream sci-fi thriller',
                'fantasy world adventure with aliens',
                'romantic tragedy on a sinking ship',
                'superheroes save the world together',
                'space travel and black holes exploration'
            ]
        }
        
        self.df = pd.DataFrame(self.ratings_data)
        self.movies_df = pd.DataFrame(self.movies_data)
        self._prepare_matrices()

    def _prepare_matrices(self):
        # Collaborative filtering matrices
        self.user_movie_matrix = self.df.pivot_table(
            index='User', 
            columns='Movie', 
            values='Rating'
        ).fillna(0)
        
        self.user_similarity = cosine_similarity(self.user_movie_matrix)
        self.user_similarity_df = pd.DataFrame(
            self.user_similarity,
            index=self.user_movie_matrix.index,
            columns=self.user_movie_matrix.index
        )

        # Content-based matrices
        self.tfidf = TfidfVectorizer(stop_words='english')
        self.tfidf_matrix = self.tfidf.fit_transform(self.movies_df['Description'])
        self.movie_similarity = cosine_similarity(self.tfidf_matrix, self.tfidf_matrix)

    def get_collaborative_recommendations(self, user_name, n_recommendations=3):
        """Get movie recommendations based on similar users' ratings"""
        if user_name not in self.user_similarity_df.index:
            return "User not found"
            
        # Find similar users
        similar_users = self.user_similarity_df[user_name].sort_values(ascending=False)
        similar_users = similar_users[similar_users.index != user_name]
        
        # Get top similar user
        top_user = similar_users.index[0]
        
        # Find movies rated by similar user but not by target user
        user_movies = set(self.df[self.df['User'] == user_name]['Movie'])
        top_user_movies = self.df[self.df['User'] == top_user]
        recommendations = top_user_movies[~top_user_movies['Movie'].isin(user_movies)]
        
        return recommendations[['Movie', 'Rating']].sort_values(
            by='Rating', 
            ascending=False
        ).head(n_recommendations)

    def get_content_recommendations(self, movie_title, n_recommendations=3):
        """Get movie recommendations based on movie content similarity"""
        if movie_title not in self.movies_df['Title'].values:
            return "Movie not found"
            
        idx = self.movies_df[self.movies_df['Title'] == movie_title].index[0]
        sim_scores = list(enumerate(self.movie_similarity[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:n_recommendations+1]
        
        movie_indices = [i[0] for i in sim_scores]
        return self.movies_df['Title'].iloc[movie_indices]

def main():
    recommender = MovieRecommender()
    
    # Test collaborative filtering
    print("\n🎬 Collaborative Filtering Recommendations")
    print("Recommendations for User A:")
    print(recommender.get_collaborative_recommendations('A'))
    
    # Test content-based filtering
    print("\n🎬 Content-Based Filtering Recommendations")
    print("Movies similar to 'Inception':")
    print(recommender.get_content_recommendations('Inception'))

if __name__ == "__main__":
    main()