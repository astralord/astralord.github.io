---
layout: post
title: 'The Annotated Statistics. Part I: Basics of Point Estimation'
date: 2022-02-09 03:13 +0800
categories: [Statistics]
tags: [statistics, parameter-estimation, exponential-family, cramer-rao-inequality, fisher-information]
math: true
---

> This series of posts is a guidance to statistics for those who already have knowledge in probability theory and would like to become familiar with mathematical statistics. Part I focuses on point estimation of parameters for the most frequently used probability distributions.


### Intro
Imagine that you are a pharmaceutical company about to introduce a new drug into production. Prior to launch you need to carry out experiments to assess its quality depending on the dosage. Say you give this medicine to an animal, after which the animal is examined and checked whether it has recovered or not by taking a dose of $X$. You can think of the result as random variable $Y$ following Bernoulli distribution:

$$ Y \sim \operatorname{Bin}(1, p(X)), $$

where $p(X)$ is a probability of healing given dose $X$. 

Typically, several independent experiments $Y_1, \dots, Y_n$ with different doses $X_1, \dots, X_n$ are made, such that

$$ Y_i \sim \operatorname{Bin}(1, p(X_i)). $$ 
	
Our goal is to estimate function $p: [0, \infty) \rightarrow [0, 1]$. For example, we can simplify to parametric model

$$ p(x) = 1 - e^{-\vartheta x}, \quad \vartheta > 0. $$

Then estimating $p(x)$ is equal to estimating parameter $\vartheta $.

![Drug experiment]({{'/assets/img/drug-efficiency.gif'|relative_url}})
*Fig. 1. Visualization of statistical experiments. The question arises: how do we estimate the value of $\vartheta$ based on our observations?*

Formally, we can define **parameter space** $\Theta$ with $\vert \Theta \vert \geq 2$ and family of probability measures $\mathcal{P} = \{ P_\vartheta \mid \vartheta \in \Theta \}$, where $P_\vartheta \neq P_{\vartheta'} \ \forall \vartheta \neq \vartheta'$. Then we are interested in the true distribution $P \in \mathcal{P}$ of random variable $X$. 

Recall from probability theory that random variable $X$ is a mapping from set of all possible outcomes $\Omega$ to a **sample space** $\mathcal{X}$. On the basis of given sample $x = X(\omega)$, $\omega \in \Omega$ we make a decision about the unknown $P$. By identifying family $\mathcal{P}$ with the parameter space $\Theta$, a decision for $P$ is equivalent to a decision for $\vartheta$. In our example above

$$ Y_i \sim \operatorname{Bin}(1, 1 - e^{-\vartheta X_i}) = P_i^\vartheta $$ 

and

$$ \mathcal{X} = \{0, 1\}^n, \quad \Theta=\left[0, \infty\right), \quad \mathcal{P}=\{\otimes_{i=1}^nP_i^{\vartheta} \mid \vartheta>0 \}. $$


### Uniformly best estimator

Mandatory parameter estimation example which can be found in every statistics handbook is mean and variance estimation for Normal distribution. Let $X_1, \dots, X_n$ i.i.d. $\sim \mathcal{N}(\mu, \sigma^2) = P_{\mu, \sigma^2}$. The typical estimation for $\vartheta = (\mu, \sigma^2)$ would be

$$ g(x) = \begin{pmatrix} \overline{x}_n \\ \hat{s}_n^2 \end{pmatrix} = \begin{pmatrix} \frac{1}{n} \sum_{i=1}^n x_i \\ \frac{1}{n} \sum_{i=1}^n (x_i-\overline{x}_n)^2 \end{pmatrix}. $$

We will get back to characteristics of this estimation later. But now it is worth noting that we are not always interested in $\vartheta$ itself, but in an appropriate functional $\gamma(\vartheta)$. We can see it in another example.

Let $X_1, \dots, X_n$ i.i.d. $\sim F$, where $F(x) = \mathbb{P}(X \leq x)$ is unknown distribution function. Here $\Theta$ is an infinite-dimensional family of distribution functions. Say we are interested in value of this function at point $k$: 

$$\gamma(F) = F(k).$$ 

Then a point estimator could be $g(x) = \frac{1}{n} \sum_{i=1}^n \mathbf{1}_{\{X_i \leq k\}}$.

Now we are ready to construct formal definition of parameter estimation. Let's define measurable space $\Gamma$ and mapping $\gamma: \Theta \rightarrow \Gamma$. Then measurable function $ g: \mathcal{X} \rightarrow \Gamma $ is called **(point) estimation** of $\gamma(\vartheta)$. 

But how do we choose point estimator and how we can measure its goodness? Let's define a criteria, non-negative function $L: \Gamma \times \Gamma \rightarrow [0, \infty)$, which we will call **loss function**, and for estimator $g$ function

$$ R(\vartheta, g) = \mathbb{E}[L(\gamma(\vartheta), g(X))] = \int_\mathcal{X} L(\gamma(\vartheta), g(X)) P_\vartheta(dx)$$ 

we will call the **risk of $g$ under $L$**.

If $\vartheta$ is the true parameter and $g(x)$ is an estimation, then $L(\gamma(\vartheta), g(x))$ measures the corresponding loss. If $\Gamma$ is a metric space, then loss functions typically depend on the distance between $\gamma(\vartheta)$ and $g(x)$, like the quadratic loss $L(x, y)=(x-y)^2$ for $\Gamma = \mathbb{R}$. The risk then is the expected loss.

Suppose we have a set of all possible estimators $g$ called $\mathcal{K}$. Then it is natural to search for an estimator, which mimimizes our risk, namely $\tilde{g} \in \mathcal{K}$, such that

$$ R(\vartheta, \tilde{g}) = \inf_{g \in \mathcal{K}} R(\vartheta, g), \quad \forall \vartheta \in \Theta. $$ 

Let's call $\tilde{g}$ an **uniformly best estimator**.

Sadly, in general, neither uniformly best estimators exist nor is one estimator uniformly better than another. For example, let's take normal random variable with unit variance and estimate its mean $\gamma(\mu) = \mu$ with quadratic loss. Pick the trivial constant estimator $g_\nu(x)=\nu$. Then

$$ R(\mu, g_\nu) = \mathbb{E}[(\mu - \nu)^2] = (\mu - \nu)^2. $$

In particular, $R(\nu, g_\nu)=0$. Thus no $g_\nu$ is uniformly better than some $g_\mu$. Also, in order to obtain a uniformly better estimator $\tilde{g}$, 

$$ \mathbb{E}[(\tilde{g}(X)-\mu)^2]=0 \quad \forall \mu \in \mathbb{R} $$

has to hold, which basically means that $\tilde{g}(x) = \mu$ with probability $1$ for every $\mu \in \mathbb{R}$, which of course is impossible.


### UMVU estimator

In order to still get *optimal* estimators we have to choose other criteria than a uniformly smaller risk. What should be our objective properties of $g$?

Let's think of difference between this estimator's expected value and the true value of $\gamma$ being estimated:

$$ B_\vartheta(g) = \mathbb{E}[g(X)] - \gamma(\vartheta). $$

This value in is called **bias** of $g$ and estimator $g$ is called **unbiased** if 
  
$$ B_\vartheta(g) = 0 \quad \forall \vartheta \in \Theta.$$
 
It is reasonable (at least at the start) to put constraint on unbiasedness for $g$ and search only in $\mathcal{E}_\gamma = \{ g \in \mathcal{K} \mid B_\vartheta(g) = 0 \}.$ Surely there can be infinite number of unbiased estimators, and we not only interested in expected value of $g$, but also in how $g$ can vary from it. Our metric for goodness then might be the variance of $g$. We call estimator $\tilde{g}$ **uniformly minimum variance unbiased (UMVU)** if

  $$\operatorname{Var}(\tilde{g}(X)) = \mathbb{E}[(\tilde{g}(X) - \gamma(\theta))^2] = \inf_{g \in \mathcal{E}_\gamma} \operatorname{Var}(g(X)).$$

In general, if we choose $L(x, y) = (x - y)^2$, then

$$ MSE_\vartheta(g) = R(\vartheta, g)=\mathbb{E}[(g(X)-\gamma(\vartheta))^2]=\operatorname{Var}_\vartheta(g(X))+B_\vartheta^2(g)$$

is called the **mean squared error**. Note, that in some cases biased estimators have lower MSE because they have a smaller variance than does any unbiased estimator.

### Chi-squared and t-distributions

Remember we talked about $\overline{x}_n$ and $\hat{s}_n^2$ being typical estimators for mean and standard deviation of normally distributed random variable? Now we are ready to talk about their properties, but firstly we have to introduce two distributions:

* Let $X_1, \dots, X_n$ be i.i.d. $\sim \mathcal{N}(0, 1)$. Then random variable $Z = \sum_{i=1}^n X_i^2$ has chi-squared distribution with $n$ degrees of freedom (notation: $Z \sim \chi_n^2$). Its density:

  $$ f_{\chi_n^2}(x) = \frac{x^{\frac{n}{2}-1} e^{-\frac{x}{2}}}{2^{\frac{n}{2}}\Gamma\big(\frac{n}{2}\big)}, \quad x > 0, $$

  where $\Gamma(\cdot)$ is a gamma function:

  $$ \Gamma(\alpha) = \int_0^\infty x^{\alpha-1} e^{-x} dx, \quad \alpha > 0.$$

  It's easy to see that $\mathbb{E}[Z] = \sum_{i=1}^n \mathbb{E}[X_i^2] = n$ and 

  $$\operatorname{Var}(Z) = \sum_{i=1}^n \operatorname{Var}(X_i^2) = n\big(\mathbb{E}[X_1^4]) - \mathbb{E}[X_1^2]^2\big) = 2n.$$

* Let $Y \sim \mathcal{N}(0, 1)$ and $Z \sim \chi_n^2$, then 

  $$ T = \frac{Y}{\sqrt{Z/n}} $$

  has t-distribution with $n$ degrees of freedom (notation $T \sim t_n$). Its density:

  $$ f_{t_n}(x) = \frac{\Gamma \big( \frac{n+1}{2} \big) } { \sqrt{n \pi} \Gamma \big( \frac{n}{2} \big) } \Big( 1 + \frac{x^2}{n} \Big)^{\frac{n+1}{2}}. $$

![Chi-squared and t-distributions]({{'/assets/img/chi-t.gif'|relative_url}})
*Fig. 2. Probability density functions for $\chi_n^2$ and $t_n$-distributions.*

It can now be shown that

$$ \overline{X}_n = \frac{1}{n} \sum_{i=1}^n X_i \sim \mathcal{N} \Big( \mu, \frac{\sigma^2}{n} \Big) $$

and

$$ \hat{s}_n^2(X) = \frac{1}{n}\sum_{i=1}^n (X_i - \overline{X}_n)^2 \sim \frac{\sigma^2}{n} \chi^2_{n-1}. $$

As a consequence:

$$ \frac{(n-1)(\overline{X}_n-\mu)}{\sqrt{n}s_n^2(X)} \sim t_{n-1}.$$

<details>
<summary>Proof</summary>
TBD
</details>

Let's check which of these estimators are unbiased. We get $\mathbb{E}[\overline{X}_n] = \mu$, therefore $\overline{X}_n$ is unbiased. On the other hand

$$ \mathbb{E}[s_n^2(X)] = \frac{\sigma^2}{n} (n - 1) \neq \sigma^2.$$

We can easily check the (un-)biasedness of these estimators. Let's fix number of samples, for example $n=10$, and run sufficiently large amount of experiments, say $10000$. Then wash, rinse, repeat again $10$ times to observe multiple outputs.

```python
print(' Mean   Var ')
n = 10
for _ in range(10):
    means = []
    stds = []
    for experiment in range(10000):
        x = np.random.normal(0, 1, n)
        means.append(np.mean(x))
        stds.append(np.std(x) ** 2)
    print("{:6.3f}".format(np.mean(means)), "{:6.3f}".format(np.mean(stds)))
```
Output:

```
 Mean   Var 
-0.003  0.901
 0.000  0.897
 0.001  0.899
-0.001  0.900
 0.001  0.906
 0.002  0.905
-0.002  0.903
 0.001  0.900
 0.004  0.892
-0.001  0.904
```

We see here that while $\overline{X}_n$ varies around $\mu=0$, expected value of estimator $\hat{s}_n^2(X)$ is near $0.9 \neq \sigma^2$.

So far we figured the unbiasedness of $g(X) = \overline{X}_n$. But how can we tell if $\overline{X}_n$ is an UMVU estimator? Can we find an estimator of $\mu$ with variance lower than $\frac{\sigma^2}{n}$?

### Efficient estimator

Given a set of unbiased estimators, it is not an easy task to determine which one provides the smallest variance. Luckily, we have a theorem which gives us a lower bound for an estimator variance.

Suppose we have a family of densities $f(\cdot, \vartheta)$, such that set $M_f=\{x \in \mathcal{X} \mid f(x, \vartheta) > 0 \}$ doesn't depend on $\vartheta$ and derivative $\frac{\partial}{\partial \vartheta} \log f(x, \vartheta)$ exists $\forall x \in \mathcal{X}$. Let's define function

$$ U_\vartheta(x) = \left\{\begin{array}{ll}
	\frac{\partial}{\partial \vartheta} \log f(x, \vartheta), & \text{if } x \in M_f, \\
	0, & \text{otherwise,}
	\end{array} \right. $$

and function

$$ I(f(\cdot, \vartheta))=\mathbb{E}_\vartheta \big[\big(\frac{\partial}{\partial \vartheta} \log f(X, \vartheta)\big)^2\big]. $$

Under mild regularity conditions we have

$$ \mathbb{E}[U_\vartheta(X)] = \mathbb{E}\big[\frac{\partial}{\partial \vartheta} \log f(x, \vartheta)\big] = \frac{\partial}{\partial \vartheta}  \mathbb{E}[\log f(x, \vartheta)] = 0$$

and 

$$ \operatorname{Var}(U_\vartheta(X)) = \mathbb{E}[(U_\vartheta(X))^2]=I(f(\cdot, \vartheta)). $$ 

Then using Cauchy-Schwartz inequality we get 

$$ \begin{aligned}
	\big( \frac{\partial}{\partial \vartheta} \mathbb{E}[g(X)] \big)^2 &= \big( \mathbb{E}_\vartheta[g(X) \cdot U_\vartheta(X)] \big)^2 \\ 
& = \big(\operatorname{Cov}(g(X), U_\vartheta(X)) \big)^2 \\
& \leq \operatorname{Var}(g(X))\cdot \operatorname{Var}(U_\vartheta(X)) \\ 
&= I(f(\cdot, \vartheta))\cdot \operatorname{Var}(g(X)).
	\end{aligned} $$
	
The resulting inequality:

$$ \operatorname{Var}(g(X)) \geq \frac{\big(\frac{\partial}{\partial \vartheta} \mathbb{E}_\vartheta[g(X)]\big)^2}{I(f(\cdot, \vartheta))} \quad \forall \vartheta \in \Theta $$

is called **Cramér–Rao bound**. Function $I(f(\cdot, \vartheta))$ is called **Fisher information** for family $\mathcal{P} = \{P_\vartheta \mid \vartheta \in \Theta \}$. If an unbiased estimator $g$ satisfies the upper equation with equality, then it is called **efficient**.

This theorem gives a lower bound for the variance of an estimator for $\gamma(\vartheta) = \mathbb{E}[g(X)]$ and can be used in principle to obtain UMVU estimators. Whenever the regularity conditions (e.g. invariance of $M_f$) are satisfied for all $g \in \mathcal{E}_\gamma$, then any efficient and unbiased estimator is UMVU.

Also, for a set of i.i.d. variables $X_1, \dots X_n$, meaning that their joint density distribution is 

$$f(x,\vartheta) = \prod_{i=1}^n f^i(x,\vartheta),$$

we have

$$ I(f(\cdot, \vartheta))=nI(f^1(\cdot, \vartheta)). $$

Let's get back to the example with $X_1, \dots, X_n$ i.i.d. $\sim \mathcal{N}(\mu, 1)$ having the density

$$ f^1(x, \vartheta) = \frac{1}{\sqrt{2\pi}} e^{-\frac{(x-\mu)^2}{2}}. $$

Then 

$$ I(f^1(\cdot, \mu)) = \mathbb{E} \Big[ \big( \frac{\partial}{\partial \mu} \log f^1 (X_1, \mu)\big)^2 \Big] = \mathbb{E}[(X_1 - \mu)^2] = 1.$$

In particular, for $X = (X_1, \dots, X_n)$ Fisher information $I(f(X, \mu)) = n$ and Cramér–Rao bound for unbiased estimator:

$$ \operatorname{Var}(g(X)) \geq \frac{1}{n} \big( \frac{\partial}{\partial \mu} \mathbb{E}[g(X)] \big)^2 = \frac{1}{n}. $$

Therefore, $g(x) = \overline{x}_n$ is an UMVU estimator.


### Multidimensional Cramér–Rao inequality

Define function 

$$ G(\vartheta)=\Big( \frac{\partial}{\partial \vartheta_j} \mathbb{E}_\vartheta[g_i(X)] \Big)_{i,j} \in \mathbb{R}^{k \times d}. $$

Then with multidimensional Cauchy-Shwartz inequality one can prove that under similar regularity conditions we have:

$$ \operatorname{Cov}(g(X)) \geq G(\vartheta) I^{-1}(f(\cdot, \vartheta))G^T(\vartheta) \in \mathbb{R}^{k \times k}, $$

where

$$ I(f(\cdot, \vartheta))=\Big( \mathbb{E}\Big[\frac{\partial}{\partial \vartheta_i} \log f(X, \vartheta) \cdot \frac{\partial}{\partial \vartheta_j} \log f(X, \vartheta) \Big]  \Big)_{i,j=1}^d \in \mathbb{R}^{d \times d}. $$

EXAMPLE:....

### Exponential family

In the previous examples, we consider without proof the fulfillment of all regularity conditions of the Rao-Kramer inequality. Next, we will discuss a family of distributions for which the Rao-Kramer inequality turns into an equality.