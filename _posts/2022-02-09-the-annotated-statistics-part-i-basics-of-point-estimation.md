---
layout: post
title: 'The Annotated Statistics. Part I: Basics of Point Estimation'
date: 2022-02-09 03:13 +0800
categories: [Statistics]
tags: [statistics, point-estimation, exponential-family, cramer-rao-inequality, fisher-information]
math: true
---

> This series of posts is a guidance to statistics for those who already have knowledge in probability theory and would like to become familiar with mathematical statistics. Part I focuses on point estimation of parameters for the most frequently used probability distributions.


## Intro

Imagine that you are a pharmaceutical company about to introduce a new drug into production. Prior to launch you need to carry out experiments to assess its quality depending on the dosage. Say you give this medicine to an animal, after which the animal is examined and checked whether it has recovered or not by taking a dose of $X$. You can think of the result as random variable $Y$ following Bernoulli distribution:

$$ Y \sim \operatorname{Bin}(1, p(X)), $$

where $p(X)$ is a probability of healing given dose $X$. 

Typically, several independent experiments $Y_1, \dots, Y_n$ with different doses $X_1, \dots, X_n$ are made, such that

$$ Y_i \sim \operatorname{Bin}(1, p(X_i)). $$ 
	
Our goal is to estimate function $p: [0, \infty) \rightarrow [0, 1]$. For example, we can simplify to parametric model

$$ p(x) = 1 - e^{-\vartheta x}, \quad \vartheta > 0. $$

Then estimating function $p(x)$ is equal to estimating parameter $\vartheta $.

![Drug experiment]({{ '/assets/img/drug-experiment.png' | relative_url }})
*Fig. 1. Visualization of statistical experiment. The question arises: how do we estimate the value of $\vartheta$ based on our observations?*

### Notations

Here is a list of notations to help you read through equations in this post.

| Symbol(s) | Meaning |
| ----------------------------- | ------------- |
| $$(\Omega, \mathcal{A}, \mathbb{P})$$ | **Probability space**: triplet, containing <br> $\cdot$ set of all possible outcomes $\Omega$, <br> $\cdot$ $\sigma$-algebra (event space) $\mathcal{A}$, <br> $\cdot$ probability measure $\mathbb{P}$. |
| $$ (\mathcal{X}, \mathcal{B}) $$ | Measurable space, defined by set $\mathcal{X}$ and $\sigma$-algebra $\mathcal{B}$. If we define measure <br> $$P(B) = \mathbb{P}(X^{-1}(B)),\ B \in \mathcal{B},$$ <br> then $(\mathcal{X}, \mathcal{B}, P)$ is also a probability space and $\mathcal{X}$  is called **sample space**.|
| $$ X: (\Omega, \mathcal{A}, \mathbb{P}) \rightarrow (\mathcal{X}, \mathcal{B}) $$ | Random variable: mapping from set of possible outcomes $\Omega$ to sample space $\mathcal {X}$.  |
| $$ x = X(\omega) $$ | Sample, element of $\mathcal {X}$. |
| $$ \Theta $$ | **Parametric space**, $\vert \Theta \vert \geq 2$. |
| $$ \mathcal{P} = \{ P_\vartheta \mid \vartheta \in \Theta \} $$ | Family of probability measures on $(\mathcal{X}, \mathcal{B})$, where $P_\vartheta \neq P_{\vartheta'} \ \forall \vartheta \neq \vartheta'$. |

We are interested in the true distribution $P \in \mathcal{P}$ of random variable $X: \Omega \rightarrow \mathcal{X}$. On the basis of $x=X(\omega)$ we make a decision about the unknown $P$. By identifying family $\mathcal{P}$ with the parameter space $\Theta$, a decision for $P$ is equivalent to a decision for $\vartheta$. In our example above:

$$ Y_i \sim \operatorname{Bin}(1, 1 - e^{-\vartheta X_i}) = P_i^\vartheta. $$ 

Formally,

$$ \mathcal{X}=\{ 0, 1 \}^n, \quad \mathcal{B}=\mathcal{P(X)}, \quad \mathcal{P}=\{\otimes_{i=1}^nP_i^{\vartheta} \mid \vartheta>0 \}, \quad \Theta=\left[0, \infty\right). $$


### Uniformly best estimator
Now we are ready to construct formal definition of parameter estimation. Let's define measurable space $(\Gamma, \mathcal{A}_\Gamma)$ and mapping $\gamma: \Theta \rightarrow \Gamma$. Then measurable function 

$$ g: (\mathcal{X}, \mathcal{B}) \rightarrow \Gamma(\mathcal{A}_\Gamma) $$

is called **(point) estimation** of $\gamma(\vartheta)$.

Mandatory parameter estimation example that should be in every statistics handbook is mean and variance estimation for Normal distribution. Let $X_1, \dots, X_n$ i.i.d. $\sim \mathcal{N}(\mu, \sigma^2) = P_{\mu, \sigma^2}$, then

$$\mathcal{X}=\mathbb{R}^n, \quad \mathcal{B}=\mathcal{B}^n, \quad \mathcal{P}=\{\otimes_{i=1}^nP_{\mu, \sigma^2} \mid \mu \in \mathbb{R}, \sigma^2>0 \}, \quad \Theta=\mathbb{R} \times \left[0, \infty\right).$$

The typical estimation for $\gamma(\vartheta) = \vartheta = (\mu, \sigma^2)$ (which you already might have seen somewhere) is

$$ g(x) = \begin{pmatrix} \overline{x}_n \\ \hat{s}_n^2 \end{pmatrix} = \begin{pmatrix} \frac{1}{n} \sum_{i=1}^n x_i \\ \frac{1}{n} \sum_{i=1}^n (x_i-\overline{x}_n)^2 \end{pmatrix}. $$

```python
import numpy as np

def g1(x):
	return np.mean(x), np.std(x)
```
		
Let's think of another example: $X_1, \dots, X_n$ i.i.d. $\sim F$, where $F(x) = \mathbb{P}(X \leq x)$ is unknown distribution function. Here $\Theta$ is an infinite-dimensional family of distribution functions. Say we are interested in value of this function at one point $k$: 

$$\gamma(F) = F(k).$$ 

Then a point estimator could be $g(x) = \frac{1}{n} \sum_{i=1}^n \mathbf{1}_{\{X_i \leq k\}}$.

```python
def g2(x):
	return np.sum([1 for xx in x if xx <= k]) / len(x)
```

We see from the last example why mapping $\gamma: \Theta \rightarrow \Gamma$ was introduced, as we are not always interested in $\vartheta$ itself, but in an appropriate functional. 

## UMVU estimator 

* Let $g$ be an estimation of $\gamma$, then

  $$ B_\theta(g) = \mathbb{E}_\theta[g(X)] - \gamma(\theta) $$

  is called **bias** of $g$. Estimation $g$ is called **unbiased** if 
  
  $$ B_\theta(g) = 0 \quad \forall \theta \in \Theta.$$

* Estimator $\tilde{g}$ is called **uniformly minimum variance unbiased (UMVU)** if

  $$ \tilde{g} \in \mathcal{E}_\gamma = \{g| B_\theta(g) = 0 \} $$

and

  $$\operatorname{Var}_\theta(\tilde{g}(X)) = \mathbb{E}_\theta[(\tilde{g}(X) - \gamma(\theta))^2] = \inf_{g \in \mathcal{E}_\gamma} \operatorname{Var}(g(X)).$$


## Efficient estimator

...

