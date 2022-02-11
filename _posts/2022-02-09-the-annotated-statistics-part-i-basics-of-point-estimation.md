---
layout: post
title: 'The Annotated Statistics. Part I: Basics of Point Estimation'
date: 2022-02-09 03:13 +0800
categories: [Statistics]
tags: [statistics, parameter-estimation, exponential-family, cramer-rao-inequality, fisher-information]
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

<details>
  <summary> Code </summary>
  
  Python code here:
  
  ```python
  import numpy as np
  import matplotlib.pyplot as plt

  plt.rcParams['text.usetex'] = True
  plt.rcParams["font.size"] = "14"

  def plot_drug_experiment(theta):
      plt.rcParams["figure.figsize"] = (10, 1)
      sx = plt.subplot(1, 1, 1)
      x = {True: [], False: []}
      for i in np.arange(1, 10):
          p = 1 - np.exp(-theta * i)
          y = np.random.binomial(1, p)
          x[y > 0].append(i)
      plt.scatter(x[True], [1] * len(x[True]), marker="P", color="#21ba0d", \
                  alpha=0.6, s=100, edgecolor="k", linewidth=1)
      plt.scatter(x[False], [1] * len(x[False]), marker="X", color="#db4444", \
                  alpha=0.6, s=100, edgecolor="k", linewidth=1)
      plt.ylim([0.5, 1.5])
      plt.yticks([1], labels=["Healed"])
      plt.xlabel("Dose $X_i$")

  plot_drug_experiment(???) # place your parameter here
  ```
  
</details>

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
| $$ \Theta $$ | **Parameter space**, $\vert \Theta \vert \geq 2$. |
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

The typical estimation for $\gamma(\vartheta) = \vartheta = (\mu, \sigma^2)$ would be

$$ g(x) = \begin{pmatrix} \overline{x}_n \\ \hat{s}_n^2 \end{pmatrix} = \begin{pmatrix} \frac{1}{n} \sum_{i=1}^n x_i \\ \frac{1}{n} \sum_{i=1}^n (x_i-\overline{x}_n)^2 \end{pmatrix}. $$
		
Let's think of another example: $X_1, \dots, X_n$ i.i.d. $\sim F$, where $F(x) = \mathbb{P}(X \leq x)$ is unknown distribution function. Here $\Theta$ is an infinite-dimensional family of distribution functions. Say we are interested in value of this function at point $k$: 

$$\gamma(F) = F(k).$$ 

Then a point estimator could be $g(x) = \frac{1}{n} \sum_{i=1}^n \mathbf{1}_{\{X_i \leq k\}}$. We see from this example why mapping $\gamma: \Theta \rightarrow \Gamma$ was introduced, as we are not always interested in $\vartheta$ itself, but in an appropriate functional. 

Now you might want to ask, how to choose point estimator and how to measure its goodness? Let's define non-negative function $L: \Gamma \times \Gamma \rightarrow [0, \infty)$, we will call it **loss function**, and for estimator $g$ function

$$ R(\vartheta, g) = \mathbb{E}[L(\gamma(\vartheta), g(X))] = \int_\mathcal{X} L(\gamma(\vartheta), g(X)) P_\vartheta(dx)$$ 

will be the **risk of $g$ under $L$**.

If $\vartheta$ is the true parameter and $g(x)$ is an estimation, then $L(\gamma(\vartheta), g(x))$ measures the corresponding loss. If $\Gamma$ is a metric space, then loss functions typically depend on the distance between $\gamma(\vartheta)$ and $g(x)$, like the quadratic loss $L(x, y)=(x-y)^2$ for $\Gamma = \mathbb{R}$. The risk then is the expected loss.

Now say we have a set of all possible estimators $g$ called $\mathcal{K}$, then $\tilde{g} \in \mathcal{K}$ is called **uniformly best estimator** if

$$ R(\vartheta, \tilde{g}) = \inf_{g \in \mathcal{K}} R(\vartheta, g), \quad \forall \vartheta \in \Theta. $$

In general, neither uniformly best estimators exist nor is one estimator uniformly better than another. For example, let's take normal random variable with unit variance and estimate its mean $\gamma(\mu) = \mu$ with quadratic loss. Pick the trivial constant estimator $g_\nu(x)=\nu$. Then

$$ R(\mu, g_\nu) = \mathbb{E}[(\mu - \nu)^2] = (\mu - \nu)^2. $$

In particular, $R(\nu, g_\nu)=0$. Thus no $g_\nu$ is uniformly better than some $g_\mu$. Also, in order to obtain a uniformly better estimator $\tilde{g}$, 

$$ \mathbb{E}[(\tilde{g}(X)-\mu)^2]=0 \quad \forall \mu \in \mathbb{R} \quad \Longleftrightarrow \quad \tilde{g}(x)=\mu \ P_\mu\text{-a.s.} \quad \forall \mu \in \mathbb{R}$$

has to hold, which is contradictory.

Therefore in order to still get *optimal* estimators we have to choose other criteria than a uniformly smaller risk.

## UMVU estimator 

* Let $g$ be an estimation of $\gamma$, then

  $$ B_\theta(g) = \mathbb{E}_\theta[g(X)] - \gamma(\theta) $$

  is called **bias** of $g$. Estimation $g$ is called **unbiased** if 
  
  $$ B_\theta(g) = 0 \quad \forall \theta \in \Theta.$$

* Estimator $\tilde{g}$ is called **uniformly minimum variance unbiased (UMVU)** if

  $$ \tilde{g} \in \mathcal{E}_\gamma = \{g| B_\theta(g) = 0 \} $$

  and

  $$\operatorname{Var}_\theta(\tilde{g}(X)) = \mathbb{E}_\theta[(\tilde{g}(X) - \gamma(\theta))^2] = \inf_{g \in \mathcal{E}_\gamma} \operatorname{Var}(g(X)).$$

* If we choose $L(x, y) = (x - y)^2$, then

  $$ MSE_\vartheta(g) = R(\vartheta, g)=\mathbb{E}[(g(X)-\gamma(\vartheta))^2]=\operatorname{Var}_\vartheta(g(X))+B_\vartheta^2(g)$$

  is called the **mean squared error**.

## Efficient estimator

...

