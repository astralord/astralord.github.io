---
layout: post
title: 'Visual Guide to Statistics. Part III: Foundations of Testing'
date: 2022-04-12 03:13 +0800
categories: [Statistics]
tags: [statistics, hypothesis-test, significance-level, power-of-a-test, neyman-pearson-criterion]
math: true
published: false
---

> In this chapter we will test hypotheses about the unknown parameter $\vartheta$. As before, we have a statistical experiment with sample space $\mathcal{X}$ and family of probability measures $\mathcal{P} = \lbrace P_\vartheta \mid \vartheta \in \Theta \rbrace$.

Let's discuss a simplified clinical study, in which we want to decide whether a newly invented drug $B$ is better than a well-known drug $A$ or not. Suppose that you know from previous years that $A$ has a chance of healing about $65$%. The new drug $B$ was tested on $100$ persons and $80$ became healthy. Do we choose $A$ or $B$? In terms of mathematics we test

$$H: p \leq 0.65 \quad \text{vs} \quad K: p > 0.65, $$

where $p$ is the unknown chance of healing with $B$.

Let $\Theta = \Theta_H \cup \Theta_K$ be a partition of $\Theta$.

* $\Theta_H$ is called **(null) hypothesis**, $\Theta_K$ is called the **alternative**.
* A **randomized test** is a measurable map $\varphi: \mathcal{X} \rightarrow [0, 1]$. Here $\varphi(x)$ is the probability of a decision for $\Theta_K$ when $x=X(\omega)$ is observed.
* For a test $\varphi$ we call $\mathcal{K}= \{x \mid \varphi(x)=1 \}$ the **critical region** and $\mathcal{R}= \{x \mid \varphi(x) \in (0,1) \}$ - the **region of randomization**. A test $\varphi$ is called **non-randomized** if $\mathcal{R} = \emptyset$.

In our example we know that the statistic $\overline{X}_n$ is the UMVU estimator for $p$. A reasonable decision rule is to decide for $\mathcal{K}$ if $\overline{X}_n$ is "large". But what is "large"? For example, is

$$\varphi(x) =
	\left \{
	\begin{array}{cl}
	1, & \overline{X}_n > 0.7, \\
	0, & \overline{X}_n \leq 0.7 
	\end{array}
	\right.
$$

a reasonable test?

When deciding for $H$ or $K$ using $\varphi$, two errors can occur:

* **Error of the 1st kind**: decide for $K$ when $H$ is true.
* **Error of the 2nd kind**: decide for $H$ when $K$ is true.

Both errors occur with certain probabilities. In our example the probability of a decision for $K$ is

$$P_p(\varphi(X)=1)=P_p(\overline{X}_n > 0.7).$$

In practice, we can use approximation by normal distribution

$$
\begin{aligned}
	P_p(\overline{X}_n > 0.7) & = P_p\bigg(\frac{\sqrt{n}(\overline{X}_n - p)}{\sqrt{p(1-p)}} > \frac{\sqrt{n}(0.7 - p)}{\sqrt{p(1-p)}}\bigg) \\
	\color{Salmon}{\text{Central Limit Theorem} \rightarrow} & \approx P\bigg(\mathcal{N}(0,1) > \frac{\sqrt{n}(0.7 - p)}{\sqrt{p(1-p)}}\bigg) \\& = \Phi\bigg(\frac{\sqrt{n}(0.7 - p)}{\sqrt{p(1-p)}}\bigg),
	\end{aligned}	
$$

where $\Phi$ is the distribution function of $\mathcal{N}(0, 1)$. [Slutsky's Lemma](https://en.wikipedia.org/wiki/Slutsky%27s_theorem) tells that we may replace $p$ by $\overline{X}_n$ in the denominator. With $n=100$ and $\overline{X}_n=0.8$ we get

$$P_p(\varphi(X)=1) \approx \Phi(25(p-0.7)).$$

For example, if $p \leq 0.65$ we get

$$ P_p(\text{Error of the 1st kind}) \approx
	\left \{
	\begin{array}{cl}
	0, & p = 0.5, \\
	0.006, &  p = 0.6
	\end{array}
	\right.
$$

This probability if bounded from above:

$$P_p(\text{Error of the 1st kind}) \leq P_{0.65}(\text{Error of the 1st kind}) \approx \Phi(1.25) \approx 0.106.$$

By symmetry,

$$P_p(\text{Error of 2nd kind}) \approx
	\left \{
	\begin{array}{cl}
	0, & p = 0.9 \\
	0.006, &  p = 0.8 \\
	0.5, & p = 0.7
	\end{array}
	\right.
$$

and

$$ P_p(\text{Error of 2nd kind}) \leq P_{0.65}(\text{Error of 2nd kind}) \approx 0.894.$$

HERE: visualization

Ideally we want to minimize both errors simulaneously and pick the optimal test. The problem is