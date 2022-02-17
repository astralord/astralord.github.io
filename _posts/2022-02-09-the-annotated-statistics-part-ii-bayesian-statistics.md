---
layout: post
title: 'The Annotated Statistics. Part II: Bayesian Statistics'
date: 2022-02-09 03:22 +0800
categories: [Statistics]
tags: [statistics, parameter-estimation, bayesian-statistics, bayes-estimator, minimax-estimator]
math: true
enable_d3: true
---
<script src="//code.jquery.com/jquery.js"></script>
<style>

.node {
  stroke: #fff;
  stroke-width: 1.5px;
}

.link {
  stroke: #999;
  stroke-opacity: .6;
}

</style>


> Part II introduces different approach to parameters estimation called Bayesian interpretation.

We noted in the previous part that it is extremely unlikely to get a uniformly best estimator. An alternative way to compare risk functions is to integrate or calculate the maximum.

In Bayes interpretation parameter $\vartheta$ is random, namely realization of random variable $\theta: \Omega \rightarrow \Theta$ with distribution $\pi$. We call $\pi$ - **a prior distribution** for $\vartheta$. For an estimator $g \in \mathcal{K}$ and its risk $R(\cdot, g)$

$$ R(\pi, g) = \int_{\Theta} R(\theta, g) \pi(d \vartheta) $$

is called the **Bayes risk of $g$ with respect to $\pi$**. An estimator $\tilde{g} \in \mathcal{K}$ is called a **Bayes estimator** if it minimizes the Bayes risk over all estimators, that is

$$ R(\pi, \tilde{g}) = \inf_{g \in \mathcal{K}} R(\pi, g). $$

The right hand side of the equation above is call the **Bayes risk**. The function $R(\pi, g)$ plays the role of the average value over all risk functions, where the possible values ​​of $\theta$ are weighted according to their probabilities. Distribution $\pi$ can interpreted as prior knowledge of statistician about unknown parameter.

In the following we will denote conditional distribution of $X$ (under condition $\theta = \vartheta$) as 

$$ P^\vartheta = Q^{X|\theta=\vartheta} $$

and joint distribution of $(X, \theta)$ as $Q^{X, \theta}$: 

$$ Q^{X, \theta}(A) = \int_\Theta \int_\mathcal{X} 1_A(x,\vartheta) P_\vartheta (dx) \pi(d \vartheta). $$ 

Before experiment we have $\pi = Q^\theta$, marginal distribution of $\theta$ under $Q^{X, \theta}$, assumed distribution of parameter $\vartheta$. After observation $X(\omega)=x$ the information about $\theta$ changes from $\pi$ to $Q^{\theta | X=x}$, which we will call a **posterior distribution**  of random variable $\theta$ under condition $X=x$.