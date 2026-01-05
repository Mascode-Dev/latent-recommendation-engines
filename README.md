# Comparative Study of Latent Representation Models for Recommendation Systems

## Introduction

This report presents a comparative analysis of three collaborative filtering models based on latent variables. The objective is to evaluate how the evolution of architectures from linear to stochastic and generative-impacts recommendation accuracy using the MovieLens 100k dataset.

---

## Part I: Model presentation

### 1. Probabilistic Matrix Factorization (PMF)

PMF models user-item interactions as a dot product within a $D$-dimensional latent space.

The observed rating follows a normal distribution centered on the product of latent vectors:
- Each user $i$ is represented in a vector $U_i$ of dimension $D$.
- Each item $j$ is represented in a vector $V_j$ of dimension $D$.
- The predicted grade is :

<p align="center">
  <img src="assets/pmf_value.png" width="200">
</p>

- To prevent overfitting, I setup a L2-regularization with the objective to minimize this cost function : 

<p align="center">
  <img src="assets/reg.png" width="500">
</p>

### 2. Restricted Boltzmann Machine (RBM)

The RBM is a stochastic neural nertwoks composed of two layers : 
- Visible layer ($v$) : Items ratings (movies in the case of a recommendation algorithm)
- Hidden layer ($h$) : Latent characteristics
The **restricted** term in RBM means that there is no links between two neurons of the same layer.
This model is stochastic because each neuron will take a decision (0 or 1) according to a **probability**.

<p align="center">
  <img src="assets/rbm.png" width="500">
</p>

The learning process : Contrastive Divergence (CD)
We can't optimize our weights with a classic gradient descent, we need to use the CD-k algorithm.

*Example for a CD-1* :\
i) Injection of data in $v_0$ layer and we are computing $h_0$\
ii) Use of $h_0$ to regenerate another version of $v$ ($v_1$)\
iii) Computation of $h_1$ from data stored in $v_1$\
iv) Updating of weights by applying this formula
- $\eta$ : Learning rate
- $w_{ij}$ : The weight from visible layer element $i$ to hidden layer element $j$.

<p align="center">
  <img src="assets/rbm_weights_update.png" width="500">
</p>

### 3. Variational Auto-Encoder (VAE)

The VAE is an evolution of traditionnal encoder, instead of just compress the input into a single fixed point in the latent space, the VAE is bringing a **probability distribution** into it.

<p align="center">
  <img src="assets/VAE_Basic.png" width="500">
</p>

*Contextual information :* In the context of a recommendation algorithm for movies, user rates only a little percentage of movies he watched. So we have a small sample of data to determine the preference of the user.

A traditionnal encoder would have determine a precise category for the user despite the small sample.\
A VAE is expressing a uncertainty, smaller is our dataset, bigger the variance $\sigma$ will be. The user will be locate in a *zone* instead of a precise single point.

To maximize the probability of our data, we want to maximize the ELBO (Evidence Lower Bound) function.

<p align="center">
  <img src="assets/ELBO.png" width="500">
</p>

This function is divided into 2 parts : 
- The Log-Likelihood (the capacity of the encoder to re-create $x$ from $z$)
- The KL-Divergence (A distance metrics between the encoder's produced distribution and a reference distribution (most likely $\mathcal{N}(0,1)$ ).

By maximizing this function, we are minimizing the Loss function which is $LOSS_{VAE} = - ELBO$.

---
