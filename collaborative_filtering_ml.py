import numpy as np 
class CollaborativeFiltering:
    def __init__(self,ratings,userIds,movieIds,num_features=10,iterations=10000,lr=0.001):
        self.lr = lr
        self.ratings = ratings 
        self.userIds = userIds
        self.movieIds = movieIds
        self.num_features = num_features
        self.iterations = iterations
        self.num_users = len(userIds)
        self.num_movies = len(movieIds)
        self.weights = np.random.rand(self.num_users,self.num_features)
        self.bias = np.random.rand(self.num_users,1)
        self.features = np.random.rand(self.num_movies,self.num_features)
        #regularization 
        self.lambda_ =0.0001
        #mean normalization  of the ratings 
        

    def predict(self):
        return np.dot(self.weights,self.features.T) + self.bias
    def cost(self,preds):
        mask = self.ratings != -1 
        return np.sum((self.ratings - preds)**2*mask)
    def fit(self):
        for iteration in range(self.iterations): 
               
            preds = self.predict() + self.lambda_*np.sum(self.weights**2)
            mask = self.ratings != -1
            error = preds - self.ratings
            if iteration % 100 == 0: 
                print(f'iteration: {iteration}   , Cost: {self.cost(preds)}')
            w_grads = np.dot((error*mask),self.features)
            w_features = np.dot((error*mask).T,self.weights)
            b_grads = np.sum(error*mask,axis=1,keepdims=True)
            self.weights -= self.lr*w_grads 
            self.bias -= self.lr*b_grads
            self.features -= self.lr*w_features
        print(np.round(np.abs(self.predict()),2))
            

    
# Example usage


userIds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
movieIds = [0, 1, 3,4]
ratings = [
    [0, 5, 2,3],
    [2.8, -1, 5,2],
    [4, 3, 5,5],
    [1, -1, 3,1]
]  # Ratings for all the three movies

# Convert ratings to a full matrix with -1 for unrated movies
full_ratings = np.full((len(userIds), len(movieIds)), -1, dtype=float)
for i, rating in enumerate(ratings):
    full_ratings[i, :len(rating)] = rating

cf = CollaborativeFiltering(full_ratings,userIds,movieIds)

cf.fit()














