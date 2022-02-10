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

### Notations

Here is a list of notations to help you read through equations in the post easily.

| Symbol | Meaning |
| ----------------------------- | ------------- |
| $$(\Omega, \mathcal{A}, \mathbb{P})$$ | **Probability space**: triplet of <br> $\cdot$ set of all possible outcomes $\Omega$, <br> $\cdot$ $\sigma$-algebra (event space) $\mathcal{A}$, <br> $\cdot$ probability measure $\mathbb{P}$. |
| $$ (\mathcal{X}, \mathcal{B}) $$ | Measurable space, defined by set $\mathcal{X}$ and $\sigma$-algebra $\mathcal{B}$. |
| $$ X: (\Omega, \mathcal{A}, \mathbb{P}) \rightarrow (\mathcal{X}, \mathcal{B}) $$ | Random variable. Recall that random variable is a function defined on a set of possible outcomes $\Omega$ and it maps to measurable space $\mathcal {X}$. <br> If we define measure $P(B) = \mathbb{P}(X^{-1}(B))$, $B \in \mathcal{B}$, then triplet $(\mathcal{X}, \mathcal{B}, P)$ is also a probability space and $\mathcal{X}$  is called **sample space**. |
| $$ x = X(\omega) $$ | Sample, element of $\mathcal {X}$. |
| $$ \Theta $$ | **Parametric space**, $\mid \Theta \mid \geq 2$. |
| $$ \mathcal{P} = \{ P_\vartheta \mid \vartheta \in \Theta \} $$ | Family of probability measures on $(\mathcal{X}, \mathcal{B})$, where $P_\vartheta \neq P_{\vartheta'} \ \forall \vartheta \neq \vartheta'$. |

The idea is that we are interested in the true distribution $P \in \mathcal{P}$ of random variable $X: \Omega \rightarrow \mathcal{X}$. On the basis of $x=X(\omega)$ we make a decision about the unknown $P$. By identifying family $\mathcal{P}$ with the parameter space $\Theta$, a decision for $P$ is equivalent to a decision for $\vartheta$. In our example above:

$$ Y_i \sim \operatorname{Bin}(1, 1 - e^{-\vartheta X_i}) = P_i^\vartheta. $$ 

Formally,

$$ \mathcal{X}=\{ 0, 1 \}^n, \quad \mathcal{B}=\mathcal{P(X)}, \quad \mathcal{P}=\{\otimes_{i=1}^nP_i^{\vartheta} \mid \vartheta>0 \}, \quad \Theta=\left[0, \infty\right). $$


### Uniformly best estimator



## UMVU estimator 

* Let $g$ be an estimation of $\gamma$, then

  $$ B_\theta(g) = \mathbb{E}_\theta[g(X)] - \gamma(\theta) $$

  is called **bias** of $g$. Estimation $g$ is called **unbiased** if 
  
  $$ B_\theta(g) = 0 \quad \forall \theta \in \Theta.$$

* Estimator $\tilde{g}$ is called **uniformly minimum variance unbiased (UMVU)** if

  $$ \tilde{g} \in \mathcal{E}_\gamma = \{g| B_\theta(g) \} $$

and

  $$\operatorname{Var}_\theta(\tilde{g}(X)) = \mathbb{E}_\theta[(\tilde{g}(X) - \gamma(\theta))^2] = \inf_{g \in \mathcal{E}_\gamma} \operatorname{Var}(g(X)).$$


## Efficient estimator

...

