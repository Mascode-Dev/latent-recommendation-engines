import numpy as np

class RBM:
    def __init__(self, n_visible, n_hidden, lr=0.1):
        """
        n_visible: Movie count (visible units)
        n_hidden: Number of latent factors (hidden units)
        lr: Learning rate
        """
        self.lr = lr
        # Initialize weights (W), visible biases (a), and hidden biases (b)
        self.W = np.random.normal(0, 0.01, (n_visible, n_hidden))
        self.a = np.zeros(n_visible)  # Movie biases
        self.b = np.zeros(n_hidden)   # Latent factor biases

    def sigmoid(self, x): # Non linear activation function
        return 1 / (1 + np.exp(-x))

    def sample_h(self, v):
        """
        Sampling the hidden layer given the visible layer P(h|v)
        v: Matrix (batch_size x n_visible)
        Returns the probabilities and sampled activations of the hidden layer.
        """
        prob_h = self.sigmoid(np.dot(v, self.W) + self.b)
        
        # Bernoulli sampling to determine if the neuron activates
        return prob_h, np.random.binomial(1, prob_h)

    def sample_v(self, h):
        """
        Sampling the visible layer given the hidden layer P(v|h)
        h: Matrix (batch_size x n_hidden)
        Returns the probabilities and sampled activations of the visible layer.
        """
        prob_v = self.sigmoid(np.dot(h, self.W.T) + self.a)
        return prob_v, np.random.binomial(1, prob_v)

    def train(self, data_tensor, epochs=10, batch_size=100):
        """
        data_tensor: Matrix (users x movies) filled with 0 and 1
        """
        n_users = data_tensor.shape[0]
        
        for epoch in range(epochs):
            train_loss = 0
            # Shuffle data at each epoch
            np.random.shuffle(data_tensor)
            
            for i in range(0, n_users, batch_size): # Mini-batch processing
                v0 = data_tensor[i:i+batch_size] # Matrix of size (batch_size x n_visible)
                
                # Contrastive Divergence with 1 step (CD-1)
                prob_h0, h0 = self.sample_h(v0) # Forward
                
                prob_v1, v1 = self.sample_v(h0) # Backward
                
                prob_h1, h1 = self.sample_h(v1) # Forward
                
                # Weight update
                positive_grad = np.dot(v0.T, prob_h0) # Association between real data and latent factors
                negative_grad = np.dot(v1.T, prob_h1) # Association between reconstructed data and latent factors
                
                self.W += self.lr * (positive_grad - negative_grad) / batch_size
                self.a += self.lr * np.mean(v0 - v1, axis=0)
                self.b += self.lr * np.mean(prob_h0 - prob_h1, axis=0)
                
                train_loss += np.mean(np.abs(v0 - v1))
            
            print(f"Epoch {epoch+1}/{epochs} - Loss: {train_loss:.4f}")