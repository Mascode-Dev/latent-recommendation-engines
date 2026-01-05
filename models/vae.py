import torch
import torch.nn as nn
import torch.nn.functional as F

class VAE(nn.Module):
    def __init__(self, n_items, n_hidden=200, n_latent=64):
        super(VAE, self).__init__()
        
        # Encode the grades into latent space
        self.fc1 = nn.Linear(n_items, n_hidden) # Input layer
        self.fc_mu = nn.Linear(n_hidden, n_latent) # Mean
        self.fc_logvar = nn.Linear(n_hidden, n_latent) # Variance (log)
        
        # Decode the latent representation back to grades
        self.fc3 = nn.Linear(n_latent, n_hidden) # Hidden layer
        self.fc4 = nn.Linear(n_hidden, n_items) # Output layer

    def encode(self, x): # Encoder
        h = F.relu(self.fc1(x))
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        # Make the sampling process differentiable for backpropagation calculation
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std) # Random normal noise
        return mu + eps * std

    def decode(self, z): # Decoder
        h = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h))

    def forward(self, x): # Full forward pass
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar
    
def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
    # Math formula for KL Divergence between the learned latent distribution and standard normal distribution
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD