import numpy as np

class PMF:
    def __init__(self, n_users, n_items, n_factors=10, lr=0.01, lambda_reg=0.1):
        """
        n_factors: Number of latent factors (K)
        lr: Learning rate
        lambda_reg: Regularization parameter to avoid overfitting
        """
        self.n_users = n_users
        self.n_items = n_items
        self.n_factors = n_factors
        self.lr = lr
        self.lambda_reg = lambda_reg
        
        # Generate random latent factor matrices U and V
        self.U = np.random.normal(0, 0.1, (n_users, n_factors))
        self.V = np.random.normal(0, 0.1, (n_items, n_factors))

    def predict(self, u, i):
        """
        Predict the rating for user u and item i
        u: User matrix index
        i: Item matrix index
        Returns the predicted rating as a float.
        """
        return np.dot(self.U[u], self.V[i])
    
    def fit(self, train_data, epochs=20):
        """
        train_data: Tuples list (user_id, item_id, rating)
        """
        history = []
        for epoch in range(epochs):
            total_error = 0
            for u, i, rating in train_data:
                # Calculate the error
                prediction = self.predict(u, i)
                error = rating - prediction
                total_error += error**2
                
                # Gradient update (Stochastic Gradient Descent)
                # Update U using the current value of V, and vice versa
                u_gradient = error * self.V[i] - self.lambda_reg * self.U[u]
                v_gradient = error * self.U[u] - self.lambda_reg * self.V[i]
                
                self.U[u] += self.lr * u_gradient
                self.V[i] += self.lr * v_gradient
            
            rmse = np.sqrt(total_error / len(train_data))
            history.append(rmse)
            print(f"Epoch {epoch+1}/{epochs} - RMSE: {rmse:.4f}")
        
        return history