# Comparative Study of Latent Representation Models for Recommendation Systems

## Introduction

This report presents a comparative analysis of three collaborative filtering models based on latent variables. The objective is to evaluate how the evolution of architectures—from linear to stochastic and generative—impacts recommendation accuracy using the MovieLens 100k dataset.

---

## Part I: Mathematical Foundations

### 1. Probabilistic Matrix Factorization (PMF)

PMF models user-item interactions as a dot product within a -dimensional latent space.

- The observed rating follows a normal distribution centered on the product of latent vectors:

<p align="center">
  <img src="assets/pmf_value.png" width="600">
</p>

- To prevent overfitting, I setup a L2-regularization with the objective to minimize this cost function : 

<p align="center">
  <img src="assets/reg.png" width="600">
</p>

---