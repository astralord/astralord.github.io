---
layout: post
title: 'Visual Guide to Statistics. Part III: Asymptotic Properties Of Estimators'
date: 2022-03-12 03:13 +0800
categories: [Statistics]
tags: [statistics, consistent-estimator, central-limit-theorem, slutsky-lemma, delta-method, asymptotic-efficiency, maximum-likelihood-estimator]
math: true
published: false
---

> Consider random vector $X^{(n)}=(X_1, \dots, X_n)^T$ in $\mathcal{X}_n=\mathcal{X}^n$ with distribution $\mathcal{P}^n=\lbrace P_\vartheta^n \mid \vartheta \in \Theta \rbrace $. For any $n$ let $g_n(X^{(n)})$ be an estimator for $\gamma(\vartheta)$. A minimal condition for a good estimator is that $g_n$ is getting closer to $\gamma(\vartheta)$ with growing $n$. In this post we will focus on asymptotic properties of $g_n$.

### Consistency of estimators

Berore talking about estimators convergence, let's recall that there exist several different notions of convergence of random variables. Let $(X_n)$ be sequence of real-valued random variables, then we say

* $X_n$ **converges in distribution** towards the random variable $X$ if 

$$\lim\limits_{n \to \infty} F_{n}(x) = F(x),$$

for every $x \in \mathbb{R}$, at which $F$ is continuous. $F_n(x)$ and $F(x)$ are the cumulative distribution functions for $X_n$ and $X$ respectively. We denote convergence in distribution as $X_n \xrightarrow[]{\mathcal{L}} X$.

* $X_n$ **converges in probability** to random variable $X$ if 

$$\lim\limits_{n \to \infty} P(|X_n-X|>\varepsilon)=0 \quad \forall \varepsilon > 0.$$

Convergence in probability implies convergence in distribution. In the opposite direction, convergence in distribution implies convergence in probability when the limiting random variable $X$ is a constant. We denote convergence in probability as $X_n \xrightarrow[]{\mathbb{P}} X$.

* $X_n$ **converges almost surely** towards $X$ if 

$$P(\omega \in \Omega: \lim\limits_{n \to \infty} X_n(\omega) = X(\omega)) = 1.$$

Almost sure convergence implies convergence in probability, and hence implies convergence in distribution. Notation: $X_n \xrightarrow[]{\text{a.s.}} X$. 

The similar logic can be applied to a sequence of $d$-dimensional random variables. Also, recall [continuous mapping theorem](https://en.wikipedia.org/wiki/Continuous_mapping_theorem), which states that for a continuous function $f$ we have 

$$
\begin{aligned}
&X_n \xrightarrow[]{\mathcal{L}} X \quad \Rightarrow \quad f(X_n) \xrightarrow[]{\mathcal{L}} f(X), \\
&X_n \xrightarrow[]{\mathbb{P}} X \quad \Rightarrow \quad f(X_n) \xrightarrow[]{\mathbb{P}} f(X), \\
&X_n \xrightarrow[]{\text{a.s.}} X \quad \Rightarrow \quad f(X_n) \xrightarrow[]{\text{a.s.}} f(X). 
\end{aligned}$$

Now let $g_n:\mathcal{X}_n \rightarrow \Gamma$ be an estimator of $\gamma(\vartheta)$ with values in metric space. Assume that all experiments are defined on a joint probability space $P_\vartheta$ for all $n$. We say that

* $g_n$ is **(weakly) consistent** if

$$g_n \xrightarrow[]{\mathbb{P}}\gamma(\vartheta) \quad \forall \vartheta \in \Theta.$$

* $g_n$ is **strongly constistent** if

$$g_n \xrightarrow[]{\text{a.s.}} \gamma(\vartheta)  \quad \forall \vartheta \in \Theta.$$

Recall the method of moments from [Part I](https://astralord.github.io/posts/visual-guide-to-statistics-part-i-basics-of-point-estimation/#common-estimation-methods): $X_1, \dots, X_n$ i.i.d. $\sim P_\vartheta$, $\vartheta \in \Theta \subset \mathbb{R}^k$ and $\gamma: \Theta \rightarrow \Gamma \subset \mathbb{R}^l$. Also $m_j = \mathbb{E}_\vartheta[X_1^j] = \int x^j P_\vartheta(dx)$, $j = 1, \dots, k$ and

$$\gamma(\vartheta) = f(m_1, \dots, m_k).$$

Then choose

$$\hat{\gamma}(X) = f(\hat{m}_1, \dots, \hat{m}_k),$$

where $\hat{m}_j = \frac{1}{n} \sum_{i=1}^{n}X_k^j$. If $\mathbb{E}_\vartheta[|X|^k] < \infty$, then by Law of Large Numbers $\hat{m}_j \rightarrow m_j$ a.s. Since $f$ is continuous, we obtain

$$\hat{\gamma}(X) \xrightarrow[]{\text{a.s.}} \gamma(\vartheta).$$

Hence, $\hat{\gamma}(X)$ is a strongly consistent estimator.

### Central Limit Theorem

Let $(X_n)$ be a sequence of $d$-dimensional random variables. [Lévy's continuity theorem](https://en.wikipedia.org/wiki/L%C3%A9vy%27s_continuity_theorem#:~:text=In%20probability%20theory%2C%20L%C3%A9vy's%20continuity,convergence%20of%20their%20characteristic%20functions.) states that

$$X_n \xrightarrow[]{\mathcal{L}} X \quad \Longleftrightarrow \quad \mathbb{E}[\exp(iu^TX_n)] \rightarrow \mathbb{E}[\exp(iu^TX)] \quad \forall u \in \mathbb{R}^d.$$

If we write $u=ty$ for $t \in \mathbb{R}$, $y \in \mathbb{R}^d$, then we can say that $X_n \xrightarrow[]{\mathcal{L}} X$ if and only if

$$y^TX_n \xrightarrow[]{\mathcal{L}} y^TX \quad \forall y \in \mathbb{R}^d.$$

This statement is called **Cramér–Wold theorem**.

If $X_1, \dots, X_n$ are i.i.d. with $\mathbb{E}[X_j]=\mu \in \mathbb{R}^d$ and $\operatorname{Cov}(X_j)=\Sigma \in \mathbb{R}^{d \times d}$ (positive-definite, $\Sigma > 0$), then for random vector

$$Z^{(n)} = \frac{1}{n}\sum_{j=1}^n X_j \in \mathbb{R}^d$$

we know from one-dimensional Central Limit Theorem (CLT) that 

$$\sqrt{n}(y^TZ^{(n)}-y^T\mu) \xrightarrow[]{\mathcal{L}} \mathcal{N}(0, y^T\Sigma y) \quad \forall y \in \mathbb{R}^d.$$

Applying Cramér–Wold theorem we get

$$\sqrt{n}(Z^{(n)}-\mu) \xrightarrow[]{\mathcal{L}} \mathcal{N}(0, \Sigma).$$

We call this statement **Multidimensional Central Limit Theorem**.

### Delta-method

Let $(X_n)$ and $(Y_n)$ be sequences of $d$-dimensional random variables, such that

$$ X_n \xrightarrow[]{\mathcal{L}} X \quad \text{and} \quad Y_n \xrightarrow[]{\mathbb{P}} c$$

for some constant vector $c$. Then we can apply the continuous mapping theorem, recognizing the functions $f(x, y)=x+y$ and $f(x, y)=xy$ are continuous, and conclude that

* $X_n+Y_n \xrightarrow[]{\mathcal{L}} X + c,$
* $Y_n^TX_n \xrightarrow[]{\mathcal{L}} c^TX.$

This statement is called **Slutsky's lemma** and it can be extremely useful in estimating approximate distribution of estimators. For example, let $X_1, \dots X_n$ i.i.d. $\sim \operatorname{Bin}(1, p)$. Estimator of $p$ $g_n(X) = \overline{X}_n$ is unbiased and we know from CLT that

$$\sqrt{\overline{X}_n(1-\overline{X}_n)} \xrightarrow[]{\mathbb{P}} \sqrt{p(1-p)}.$$

By Slutsky's lemma,

$$\frac{\sqrt{n}(\overline{X}_n-p)}{\sqrt{\overline{X}_n(1-\overline{X}_n)}} \xrightarrow[]{\mathcal{L}} \mathcal{N}(0,1)$$

and for large $n$ we have

$$ P_p(|\overline{X}_n-p|<\varepsilon) \approx 2 \Phi\Bigg(\varepsilon\sqrt{\frac{n}{\overline{X}_n(1-\overline{X}_n)}}\Bigg) -1 \quad \forall p \in (0, 1), $$

where $\Phi$ is cumulative distribution function for $\mathcal{N}(0,1)$.

Slutsky's lemma also leads to important asymptotic property of estimator $g_n$, called **Delta-method**. Let $(X_n)$ be sequence of $d$-dimensional random variables, such that 

$$\frac{X_n-\mu}{c_n} \xrightarrow[]{\mathcal{L}} \mathcal{N}(0, \Sigma),$$

where $c_n \rightarrow 0$, $\mu \in \mathbb{R}^d$ и $\Sigma \geq 0 \in \mathbb{R}^{d \times d}$. Let also $g:\mathbb{R}^d \rightarrow \mathbb{R}^m$ be continuously differentiable in $\mu$ with Jacobian matrix $D \in \mathbb{R}^{m \times d}$. Then:

$$ \frac{g(X_n)-g(\mu)}{c_n} \xrightarrow[]{\mathcal{L}} \mathcal{N}(0, D\Sigma D^T).  $$

<details>
<summary>Proof</summary>
By Slutsky's Lemma

$$X_n-\mu = \frac{X_n-\mu}{c_n}c_n \xrightarrow[]{\mathcal{L}} 0.$$

Convergence in distribution to a constant implies convergence in probability: $X_n \xrightarrow[]{\mathbb{P}} \mu$. Then

$$\frac{g(X_n)-g(\mu)}{c_n}=g'(\mu)\frac{X_n-\mu}{c_n}+(g'(\xi_n)-g'(\mu))\frac{X_n-\mu}{c_n},$$

for some intermediate point $\xi_n$, such that $\|\xi_n-\mu \| \leq \|X_n-\mu \|$. From $X_n \xrightarrow[]{\mathbb{P}} \mu$ we have $\xi_n \xrightarrow[]{\mathbb{P}} \mu$ and $g'(\xi_n) \xrightarrow[]{\mathbb{P}} g'(\mu)$ (because $g$ is continuously differentiable). Applying again Slutsky's Lemma:

$$ g'(\mu) \frac{X_n-\mu}{c_n} \xrightarrow[]{\mathcal{L}} g'(\mu) \cdot \mathcal{N}(0, \Sigma) $$

finishes the proof. 
</details>

* Recall example with method of moments, but now with additional conditions on $\mathbb{E}[X_1^{2k}] < \infty$ for all $\vartheta \in \Theta$ and $\gamma$ being continuously differentiable with Jacobian matrix $D$. We know from CLT that 

$$\sqrt{n}((\hat{m}_1, \dots, \hat{m}_k)^T - (m_1, \dots, m_k)^T) \xrightarrow[]{\mathcal{L}} \mathcal{N}(0, \Sigma),$$

where $\Sigma = (\Sigma)_{i,j=1}^k = (m_{i+j} - m_i m_j)_{i,j=1}^k.$ Then 

$$\sqrt{n}(\gamma(\hat{m}_1, \dots, \hat{m}_k) - \gamma(m_1, \dots, m_k)) \xrightarrow[]{\mathcal{L}} \mathcal{N}(0, D \Sigma D^T).$$

* Take another example: let $X_1, \dots X_n$ be i.i.d. with $\mathbb{E}_\vartheta[X_i] = \mu$ and $\operatorname{Var}_\vartheta(X_i) = \sigma^2$. From CLT we have

$$\sqrt{n}(\overline{X}_n - \mu) \xrightarrow[]{\mathcal{L}} \mathcal{N}(0, \sigma^2).$$

Choose $\overline{X}_n^2$ as an estimator for $\mu^2$. Applying Delta-method we get

$$\sqrt{n}(\overline{X}_n^2-\mu^2) \xrightarrow[]{\mathcal{L}} \mathcal{N}(0, 4\mu^2\sigma^2).$$

* Let
 $$ (X_i, Y_i)^T \sim \mathcal{N}
		\begin{pmatrix}
		\begin{pmatrix}
		\mu_1 \\ \mu_2
		\end{pmatrix},
		\begin{pmatrix}
		\sigma^2 & \rho \sigma \tau \\
		\rho \sigma \tau & \tau^2
		\end{pmatrix}
		\end{pmatrix}, \quad
		i = 1, \dots, n, $$
be i.i.d with parameter $\vartheta = (\mu_1, \mu_2, \sigma^2, \tau^2, \rho)^T$. The estimator

$$\hat{\rho}_n = \frac{SQ_{xy}}{\sqrt{SQ_{xx} SQ_{yy}}}, $$

where $SQ_{xy} = \frac{1}{n} \sum_{i=1}^{n}(X_i-\overline{X}_n)(Y_i - \overline{Y}_n)$, ($SQ_{xx}, SQ_{yy}$ - likewise), is called **the Pearson correlation coefficient**. Without loss of generality, assume $\mu_1=\mu_2=0$, $\sigma=\tau=1$, because $\hat{\rho}_n$ is invariant under affine transformation. 

Prove first that  $S_n = (SQ_{xx}, SQ_{yy}, SQ_{xy})^T$ satisifies 

$$\sqrt{n}(S_n - m) \xrightarrow[]{\mathcal{L}} \mathcal{N}(0, V),$$

where $m=(1, 1, \rho)^T$ and

$$V = 2 
	    \begin{pmatrix}
	    1 & \rho^2 & \rho \\
	    \rho^2 & 1 & \rho \\
	    \rho & \rho & (1 + \rho^2)/2
	    \end{pmatrix}.$$

<details>
<summary> Sketch of the proof </summary>
We use Slutsky's Lemma and CLT to show that 
$$\sqrt{n}(\overline{X}_n \overline{Y}_n) \xrightarrow[]{\mathbb{P}} 0, \quad \sqrt{n}(\overline{X}_n)^2 \xrightarrow[]{\mathbb{P}} 0, \quad \sqrt{n}(\overline{Y}_n)^2 \xrightarrow[]{\mathbb{P}} 0.  $$

Then it is simple to conclude

$$\sqrt{n}(S_n - m) - \sqrt{n}\Big(\frac{1}{n}\sum_{i=1}^{n}Z_i - m \Big) \xrightarrow[]{\mathbb{P}} 0,$$

with $Z_i = (X_i^2, Y_i^2, X_iY_i)^T$. Then prove that 

$$\operatorname{Cov}(Z_i) = \mathbb{E}[Z_i Z_i^T]-\mathbb{E}[Z_i]\mathbb{E}[Z_i]^T = V. $$

The rest follows from multidimensional CLT.
</details>

Then take $g(S_n)=\hat{\rho}_n$ with $g(x_1, x_2, x_3) = \frac{x_3}{\sqrt{x_1 x_2}}$. Jacobian matrix of $g$ at $m$:

$$D = (-\rho/2, -\rho/2, 1).$$

In total,

$$\sqrt{n}(\hat{\rho}_n - \rho) \xrightarrow[]{\mathcal{L}} \mathcal{N}(0, DVD^T) = \mathcal{N}(0, (1-\rho^2)^2).$$

TODO: visualization

### Asympotic efficiency

Let $g_n:\mathcal{X}_n \rightarrow \Gamma \subset \mathbb{R}^l$ be a sequence of estimators with $\mu_n(\vartheta)=\mathbb{E}_\vartheta[g_n] \in \mathbb{R}^l$ and $\Sigma_n(\vartheta)=\operatorname{Cov}(\vartheta) \in \mathbb{R}^{l \times l}$, such that $\|\Sigma_n(\vartheta) \| \rightarrow 0$. Then

* $g_n$ is called **asymptotically unbiased** for $\gamma(\vartheta)$ if

$$ \mu_n(\vartheta) \rightarrow \gamma(\vartheta), $$

* $g_n$ is called **asymptotically normal** if

$$\Sigma_n^{-\frac{1}{2}}(\vartheta)(g_n-\mu_n(\vartheta)) \xrightarrow[]{\mathcal{L}} \mathcal{N}(0, \mathbb{I}_l),$$

where $\mathbb{I}_l$ is identity matrix.

Let $f_n: \mathcal{X} \rightarrow \mathbb{R}^l$ be asymptotically unbiased and asymptotically normal sequence of estimators. Under regularity conditions from [Cramér–Rao theorem](https://astralord.github.io/posts/visual-guide-to-statistics-part-i-basics-of-point-estimation/#efficient-estimator) we call $g_n$ **asymptotically efficient**, if

$$ \lim\limits_{n \rightarrow \infty} \Sigma_n(\vartheta) I(f_n(\cdot, \vartheta))=\mathbb{I}_l \quad \forall \vartheta \in \Theta,  $$

where $I(f_n(\cdot, \vartheta))$ is Fisher information.

The intuition behind definition above is the following: if $g_n$ is unbiased, then by Cramér–Rao theorem $\operatorname{Cov}_\vartheta(g_n) \geq I^{-1}(f_n(\cdot, \vartheta))$. Due to asymptotic normality:

$$\Sigma_n^{-\frac{1}{2}}(\vartheta)(g_n-\mu_n(\vartheta)) \xrightarrow[]{\mathcal{L}} \mathcal{N}(0, \mathbb{I}_l)$$

we have approximately

$$\operatorname{Cov}_\vartheta(g_n) \approx \Sigma_n(\vartheta) \approx I^{-1}(f_n(\cdot, \vartheta))$$

and $g_n$ is asymptotically unbiased and asymptotically efficient.

Recall example from [Part I](https://astralord.github.io/posts/visual-guide-to-statistics-part-i-basics-of-point-estimation/#multidimensional-cram%C3%A9rrao-inequality): for $X_1, \dots X_n$ i.i.d. $\sim \mathcal{N}(\mu, \sigma^2)$ estimator

$$g_n(X) = \begin{pmatrix}
	\overline{X}_n \\
	\frac{1}{n-1} \sum_{i=1}^{n} (X_i - \overline{X}_n)^2
	\end{pmatrix}$$
	
satisfies the equality

$$\operatorname{Cov}_\vartheta(g_n) = \begin{pmatrix}
	\sigma^2/n & 0 \\
	0 & 2\sigma^4 / (n - 1)
	\end{pmatrix}  
	= \Sigma_n(\vartheta).$$
	
But Fisher information is

$$I^{-1}(f_n(\cdot, \vartheta)) = \begin{pmatrix}
	\sigma^2/n & 0 \\
	0 & 2\sigma^4 / n
	\end{pmatrix} $$
	
and $g_n$ is not efficient, but asymptotically efficient.

### Maximum-likelihood estimators

Let $X_1, \dots X_n$ be i.i.d. $\sim P_\vartheta$, $\vartheta \in \Theta$ with densities $f(\cdot, \vartheta)$. We call

$$\ell(\cdot, \vartheta) = \log f(\cdot, \vartheta) $$

**the log-likelihood function** and set 

$$\begin{aligned}\hat{\theta}_n(X) &= \arg \sup_{\vartheta \in \Theta} f(X, \vartheta) \\&= \arg \sup_{\vartheta \in \Theta} \ell (X, \vartheta) \\&= \arg \sup_{\vartheta \in \Theta} \frac{1}{n} \sum_{i=1}^{n} \ell (X_i, \vartheta) \end{aligned}$$

as **the maximum-likelihood estimator** for $\vartheta$.

MATH-HEAVY STUFF

Take an example: let $X_1, \dots X_n$ be i.i.d. $\sim \operatorname{Exp}(\lambda)$ with joint density

$$f_n(X, \lambda) = \lambda^n \exp \Big(-\lambda \sum_{i=1}^n X_i \Big) \quad \forall x \in \mathbb{R}^+.$$

To find maximum-likelihood estimator one must maximize log-density

$$\ell_n(X, \lambda) = n \log(\lambda) - \lambda \sum_{i=1}^n X_i \quad \forall x \in \mathbb{R}^+$$

with respect to $\lambda$. Taking the derivative and equating it to zero we get

$$\frac{n}{\lambda} = \sum_{i=1}^{n} X_i,$$

and estimator is

$$\hat{\lambda}_n = \frac{1}{\overline{X}_n}.$$

Next, using the fact that

$$\mathbb{E}_\lambda[X] = \lambda^{-1} \quad \text{and} \quad  \operatorname{Var}_\lambda(X) = \lambda^{-2},$$

and $\dot{\ell}_1(X, \lambda) = -(X - \lambda^{-1})$, we calculate Fisher information:

$$I(f(\cdot, \lambda)) = \mathbb{E}_\lambda\Big[\Big(X - \frac{1}{\lambda}\Big)^2\Big]=\frac{1}{\lambda^2}.$$
	
By theorem of asymptotic efficiency of ML-estimators we get 

$$\sqrt{n}(\hat{\lambda}_n - \lambda) \xrightarrow[]{\mathcal{L}} \mathcal{N}(0, \lambda^2),$$

On the other hand by CLT

$$\sqrt{n}\Big(\overline{X}_n - \frac{1}{\lambda}\Big) \xrightarrow[]{\mathcal{L}} \mathcal{N}\Big(0, \frac{1}{\lambda^2}\Big).$$

Using Delta-method for $g(x) = x^{-1}$ we get the same result:

$$\sqrt{n}(\overline{X}_n^{-1} - \lambda) \xrightarrow[]{\mathcal{L}} \mathcal{N}(0, \lambda^2). $$
