
---
title: The Annotated Statistics. Part II
author: Aleksandr Samarin
date: 2021-02-08
categories: [Statistics]
tags: [statistics]
math: true
---

### Definition

* Let $g$ be an estimation of $\gamma$, then

  $$ B_\theta(g) = \mathbb{E}_\theta[g(X)] - \gamma(\theta) $$

  is called **bias** of $\g$. Estimation $g$ is called *unbiased* if $B(\theta) = 0 \quad \forall \theta \in \Theta$.

* Estimator $g*$ is called *uniformly minimum variance unbiased (UMVU)* if

  $$\operatorname{Var}_\theta(g*(X)) = \mathbb{E}_\theta[(g*(X) - \gamma(\theta))**2] = \inf_{g \in \{g|g \text{is unbiased}\} } \operatorname{Var}(g(X)).$$
