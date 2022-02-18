---
layout: post
title: 'The Annotated Statistics. Part II: Bayesian Statistics'
date: 2022-02-09 03:22 +0800
categories: [Statistics]
tags: [statistics, parameter-estimation, bayesian-statistics, bayes-estimator, minimax-estimator, least-favorable-prior]
math: true
enable_d3: true
---

> Part II introduces different approach to parameters estimation called Bayesian statistics.

### Basic definitions

We noted in the previous part that it is extremely unlikely to get a uniformly best estimator. An alternative way to compare risk functions is to look at averaged values (weighting over parameters probabilities) or at maximum values for worst-case scenarios.

In Bayes interpretation parameter $\vartheta$ is random, namely instance of random variable $\theta: \Omega \rightarrow \Theta$ with distribution $\pi$. We call $\pi$ a **prior distribution** for $\vartheta$. For an estimator $g \in \mathcal{K}$ and its risk $R(\cdot, g)$

$$ R(\pi, g) = \int_{\Theta} R(\theta, g) \pi(d \vartheta) $$

is called the **Bayes risk of $g$ with respect to $\pi$**. An estimator $\tilde{g} \in \mathcal{K}$ is called a **Bayes estimator** if it minimizes the Bayes risk over all estimators, that is

$$ R(\pi, \tilde{g}) = \inf_{g \in \mathcal{K}} R(\pi, g). $$

The right hand side of the equation above is call the **Bayes risk**. The function $R(\pi, g)$ plays the role of the average value over all risk functions, where the possible values ​​of $\theta$ are weighted according to their probabilities. Distribution $\pi$ can interpreted as prior knowledge of statistician about unknown parameter.

In the following we will denote conditional distribution of $X$ (under condition $\theta = \vartheta$) as 

$$ P_\vartheta = Q^{X \mid \theta=\vartheta} $$

and joint distribution of $(X, \theta)$ as $Q^{X, \theta}$: 

$$ Q^{X, \theta}(A) = \int_\Theta \int_\mathcal{X} 1_A(x,\vartheta) P_\vartheta (dx) \pi(d \vartheta). $$ 

Before experiment we have $\pi = Q^\theta$, marginal distribution of $\theta$ under $Q^{X, \theta}$, assumed distribution of parameter $\vartheta$. After observation $X(\omega)=x$ the information about $\theta$ changes from $\pi$ to $Q^{\theta \mid X=x}$, which we will call a **posterior distribution**  of random variable $\theta$ under condition $X=x$.

### Posterior risk

Recall that risk function is an expected value of a loss function $L$:

$$ R(\vartheta, g) =  \int_{\mathcal{X}} L(\gamma(\vartheta), g(x)) P_\vartheta(dx). $$

Then 

$$ \begin{aligned}
R(\pi,g) & =\int_\Theta R(\vartheta, g) \pi(d\vartheta) \\
&=\int_{\Theta} \int_{\mathcal{X}} L(\gamma(\vartheta), g(x)) P_\vartheta(dx) \pi(d\vartheta)\\
& = \int_{\Theta \times \mathcal{X}} L(\gamma(\vartheta), g(x)) Q^{X,\theta} (dx, d\vartheta) \\
&=\int_{\mathcal{X}} {\color{Salmon}{ \int_{\Theta} L(\gamma(\vartheta), g(x)) Q^{\theta \mid X = x} (d\vartheta)}} Q^X(dx) \\
& = \int_{\mathcal{X}} {\color{Salmon}{R_{\pi}^x(g)}} Q^X(dx).
\end{aligned} $$

The term

$$ R_{\pi}^x(g) :=\int_{\Theta} L(\gamma(\vartheta), g(x)) Q^{\theta | X = x} (d\vartheta) $$

is called a **posterior risk** of $g$ with given $X=x$. It can be shown that for an estimator $g^*$ of $\vartheta$ to be Bayes, it must provide minimum posterior risk:

$$ R_{\pi}^x(g^*)=\inf_{g \in \mathcal{K}}R_{\pi}^x(g)=\inf_{a \in \Theta} \int L(\vartheta, a) Q^{\theta \mid X = x}(d\vartheta), $$

because $R(\pi, g)$ is minimal if and only if $R_\pi^x(g)$ is minimal. In particular, for quadratic loss $L(\vartheta,a) = (\vartheta-a)^2$ Bayes estimator is

$$ g^*(x) = \mathbb{E}[\theta \mid X = x] = \int_{\Theta} \vartheta Q^{\theta \mid X=x} (d \vartheta). $$

Say for $P_\vartheta$ we have density function $f(x \mid \vartheta)$, and for $\pi$ density is $h(\vartheta)$. Then posterior distribution of $Q^{\theta \mid X=x}$ has density 

$$ f(\vartheta|x) = \frac{f(x|\vartheta) h(\vartheta)}{ \int_\Theta f(x|\vartheta) h(\vartheta) d\vartheta }. $$

Posterior and Bayes risks respectively

$$ R_\pi^x(g) = \frac{\int_\Theta L(\vartheta, g(x))f(x|\vartheta) h(\vartheta) d\vartheta}{\int_\Theta f(x|\vartheta) h(\vartheta) d\vartheta} $$

and 

$$ R(\pi, g)=\int_{\mathcal{X}}\int_\Theta L(\vartheta, g(x))f(x|\vartheta) h(\vartheta) d\vartheta dx. $$	

Let's consider an example of an estimation of probability parameter for binomial distribution. Let $\Theta = (0, 1)$, $\mathcal{X} = \lbrace 0, \dots, n \rbrace$ and

$$ P_\vartheta(X=x) = \binom n x \vartheta^x (1-\vartheta)^{n-x}. $$

We take quadratic loss function $L(x,y)=(x-y)^2$. Say we only have observed one sample $X=x$. From previous post we know that binomial distribution belongs to exponential family and therefore $g(x) = \frac{x}{n}$ is an UMVU estimator for $\vartheta$ with

$$ \operatorname{Var}(g(X)) = \frac{\vartheta(1-\vartheta)}{n}. $$

On the other hand, we have density

$$ f(x | \vartheta) = \binom n x \vartheta^x (1-\vartheta)^{n-x} 1_{ \lbrace 0, \dots n \rbrace }(x). $$

If we take prior uniform distribution $\pi \sim \mathcal{U}(0, 1)$, then

$$ h(\vartheta) = 1_{(0, 1)}(\vartheta) $$

and posterior density

$$ f(\vartheta \mid x) = \frac{\vartheta^x (1-\vartheta)^{n-x} 1_{(0,1)}(\vartheta)}{B(x+1, n-x+1)}, $$ 

where we have beta-function in denominator:

$$ B(a,b)=\int_{0}^{1} \vartheta^{a-1} (1-\vartheta)^{b-1} d \vartheta. $$

Then Bayes estimator will be

$$ \begin{aligned}
g^*(x)&=\mathbb{E}[\theta|X=x]\\
&=\int_0^1 \frac{\vartheta^{x+1}(1-\vartheta^{n-x})}{B(x+1, n-x+1)}\\
&=\frac{B(x+2, n-x+1)}{B(x+1, n-x+1)} =\frac{x+1}{n+2},
\end{aligned} $$

and Bayes risk:

$$
\begin{aligned}
			R(\pi,g^*) & =\int_0^1 R(\vartheta, g^*) d\vartheta\\
			&=\int_0^1 \mathbb{E}\Big[\Big(\frac{X+1}{n+2}-\vartheta \Big)^2\Big]d\vartheta \\
			 & =\frac{1}{(n+2)^2} \int_0^1 (n\vartheta - n\vartheta^2+1-4\vartheta+4\vartheta^2)\ d\vartheta\\
			 &=\frac{1}{6(n+2)}.  
		\end{aligned}
$$

<button onclick="update()">Sample $X$</button>

<script src="https://d3js.org/d3.v4.min.js"></script>

<link href="../vendor/bootstrap/css/bootstrap.min.css" rel="stylesheet">
<script src="../vendor/bootstrap/js/bootstrap.bundle.min.js"></script>

<a class="btn btn-secondary">Sample $X$</a>

<div id="bin_bayes_plt"></div> 
<script>

var margin = {top: 20, right: 50, bottom: 30, left: 30},
    width = 250 - margin.left - margin.right,
    height = 175 - margin.top - margin.bottom;

var prior_svg = d3.select("#bin_bayes_plt")
  .append("svg")
    .attr("width", width + margin.left + margin.right)
    .attr("height", height + margin.top + margin.bottom)
  .append("g")
    .attr("transform",
          "translate(" + margin.left + "," + margin.top + ")");

var prior_data = [
   {x: -0.25, y: 0},
   {x: 0, y: 0},
   {x: 0, y: 1},
   {x: 1, y: 1},
   {x: 1, y: 0},
   {x: 1.25, y: 0}
];

var x = d3.scaleLinear()
        .domain([d3.min(prior_data, function(d) { return d.x }), d3.max(prior_data, function(d) { return d.x }) ])
        .range([0, width]);
        
prior_svg.append("g")
  .attr("transform", "translate(0," + height + ")")
  .call(d3.axisBottom(x));
  
var y = d3.scaleLinear()
        .range([height, 0])
        .domain([0, 1.25]);
        
prior_svg.append("g")
  .call(d3.axisLeft(y).ticks(5));
  

prior_svg.append("text")
.attr("text-anchor", "start")
.attr("y", -5)
.attr("x", 0)
.text(function(d){ return('Prior')})
.attr("font-family", function(d,i) {return "Saira"; })

prior_svg
    .append('g')
    .append("path")
      .datum(prior_data)
      .attr("fill", "#348ABD")
      .attr("border", 0)
      .attr("opacity", ".8")
      .attr("stroke", "#000")
      .attr("stroke-width", 1)
      .attr("stroke-linejoin", "round")
      .attr("d",  d3.line()
          .x(function(d) { return x(d.x); })
          .y(function(d) { return y(d.y); })
      );
      
function update() {}

</script>

Let's take another example: $X_1, \dots X_n$ i.i.d. $\sim P_\mu^1 = \mathcal{N}(\mu, \sigma^2)$ with $\sigma^2$ known in advance. Take for $\mu$ prior distribution with gaussian density

$$ h(\mu) = \frac{1}{\sqrt{2 \pi \tau^2}} \exp \Big( -\frac{(\mu-\mu_0)^2}{2\tau^2} \Big). $$

Taking density for $X$

$$ f(x|\mu)=\Big( \frac{1}{\sqrt{2\pi \sigma^2}}\Big)^n \exp \Big( \frac{1}{2\sigma^2}\sum_{j=1}^n(x_j-\mu)^2 \Big ), $$

we get posterior distribution

$$ Q^{\mu|X=x} \sim \mathcal{N} \Big( g_{\mu_0, \tau^2}(x), \Big( \frac{n}{\sigma^2} + \frac{1}{\tau^2}\Big)^{-1}  \Big), $$

where

$$ g_{\mu_0, \tau^2}(x)=\Big( 1 + \frac{\sigma^2}{n \tau^2} \Big)^{-1} \overline{x}_n+\Big( \frac{n \tau^2}{\sigma^2}+1 \Big)^{-1} \mu_0. $$

For quadratic loss function $g_{\mu_0, \tau^2}(x)$ is a Bayes estimator. It can be interpreted as following: for large values of $\tau$ (not enough prior information) estimator $g_{\mu_0, \tau^2}(x)$ $\approx \overline{x}_n$. Otherwise,  $g_{\mu_0, \tau^2}(x)$ $\approx \mu_0$.

HERE: JS EXAMPLE FOR NORMAL

### Minimax estimator

