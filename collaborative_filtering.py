import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Example user-item rating matrix
ratings = np.array([
    [4, 0, 0, 5, 1],
    [5, 5, 4, 0, 0],
    [0, 0, 0, 2, 4],
    [2, 4, 5, 0, 0]
])

# Compute cosine similarity between users
user_similarity = cosine_similarity(ratings)

# Predict ratings for a specific user (user 0)
user_id = 0
similar_users = user_similarity[user_id]

# Weighted sum of ratings by similar users
predicted_ratings = np.dot(similar_users, ratings) / np.array([np.abs(similar_users).sum()])

print(f"Predicted ratings for user {user_id}: {predicted_ratings}")
