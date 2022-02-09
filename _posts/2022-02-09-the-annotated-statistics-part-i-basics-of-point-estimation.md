---
layout: post
title: 'The Annotated Statistics. Part I: Basics of Point Estimation'
date: 2022-02-09 03:13 +0800
categories: [Statistics]
tags: [statistics, point-estimation, exponential-family, cramer-rao-inequality, fisher-information]
math: true
---

> This series of posts is a guidance to statistics for those who already have knowledge in probability theory and would like to become familiar with mathematical statistics. Part I focuses on point estimation of parameters for the most frequently used probability distributions.

### Uniformly best estimator

Imagine that you are a pharmaceutical company about to introduce a new drug into production. Prior to launch you need to carry out experiments to assess its quality depending on the dosage. Say you give this medicine to an animal, after which the animal is examined and checked whether it has recovered or not by taking a dose of $X$. You can think of the result as random variable $Y$, following Bernoulli distribution:

$$ Y \sim \operatorname{Bin}(1, p(X)), $$

where $p(X)$ is a probability of healing given dose $X$. 

Typically, several independent experiments $Y_1, \dots, Y_n$ with different doses $X_1, \dots, X_n$ are made, such that

$$ Y_i \sim \operatorname{Bin}(1, p(X_i)). $$ 

Our goal is to estimate function $p: [0, \infty) \rightarrow [0, 1]$. For example, we can simplify to parametric model

$$ p(x) = 1 - e^{-\beta x}, \quad \beta > 0. $$

Then estimating function $p(x)$ is equal to estimating parameter $\beta$.


### UMVU estimator 

* Let $g$ be an estimation of $\gamma$, then

  $$ B_\theta(g) = \mathbb{E}_\theta[g(X)] - \gamma(\theta) $$

  is called **bias** of $g$. Estimation $g$ is called *unbiased* if 
  
  $$ B(\theta) = 0 \quad \forall \theta \in \Theta.$$

* Estimator $\tilde{g}$ is called *uniformly minimum variance unbiased (UMVU)* if

  $$ \tilde{g} \in \Varepsilon_\gamma = \{g|g \text{ is unbiased}\} $$

and

  $$\operatorname{Var}_\theta(\tilde{g}(X)) = \mathbb{E}_\theta[(\tilde{g}(X) - \gamma(\theta))^2] = \inf_{g \in \Varepsilon_\gamma} \operatorname{Var}(g(X)).$$


### Efficient estimator

...

