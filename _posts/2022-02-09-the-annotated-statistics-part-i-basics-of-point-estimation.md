---
layout: post
title: 'The Annotated Statistics. Part I: Basics of Point Estimation'
date: 2022-02-09 03:13 +0800
categories: [Statistics]
tags: [statistics, point-estimation, exponential-family]
math: true
---

> This series of posts is a guidance to statistics for those who already have knowledge in probability theory and would like to become familiar with mathematical statistics. Part I focuses on point estimation of parameters for the most frequently used probability distributions.

### Definition

* Let $g$ be an estimation of $\gamma$, then

  $$ B_\theta(g) = \mathbb{E}_\theta[g(X)] - \gamma(\theta) $$

  is called **bias** of $g$. Estimation $g$ is called *unbiased* if 
  
  $$ B(\theta) = 0 \quad \forall \theta \in \Theta.$$

* Estimator $\tilde{g}$ is called *uniformly minimum variance unbiased (UMVU)* if

  $$\operatorname{Var}_\theta(\tilde{g}(X)) = \mathbb{E}_\theta[(\tilde{g}(X) - \gamma(\theta))^2] = \inf_{g \in \{g|g \text{ is unbiased}\} } \operatorname{Var}(g(X)).$$



