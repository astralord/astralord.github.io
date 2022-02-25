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

If we take prior uniform distribution $\pi \sim \mathcal{U}(0, 1)$, then $ h(\vartheta) = 1_{(0, 1)}(\vartheta)$ and posterior density

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

Let's take another example: $X_1, \dots X_n$ i.i.d. $\sim P_\mu^1 = \mathcal{N}(\mu, \sigma^2)$ with $\sigma^2$ known in advance. Take for $\mu$ prior distribution with gaussian density

$$ h(\mu) = \frac{1}{\sqrt{2 \pi \tau^2}} \exp \Big( -\frac{(\mu-\mu_0)^2}{2\tau^2} \Big). $$

Taking density for $X$

$$ f(x|\mu)=\Big( \frac{1}{\sqrt{2\pi \sigma^2}}\Big)^n \exp \Big( \frac{1}{2\sigma^2}\sum_{j=1}^n(x_j-\mu)^2 \Big ), $$

we get posterior distribution

$$ Q^{\mu|X=x} \sim \mathcal{N} \Big( g_{\mu_0, \tau^2}(x), \Big( \frac{n}{\sigma^2} + \frac{1}{\tau^2}\Big)^{-1}  \Big), $$

where

$$ g_{\mu_0, \tau^2}(x)=\Big( 1 + \frac{\sigma^2}{n \tau^2} \Big)^{-1} \overline{x}_n+\Big( \frac{n \tau^2}{\sigma^2}+1 \Big)^{-1} \mu_0. $$

For quadratic loss function $g_{\mu_0, \tau^2}(x)$ is a Bayes estimator. It can be interpreted as following: for large values of $\tau$ (not enough prior information) estimator $g_{\mu_0, \tau^2}(x) \approx \overline{x}_n$. 

Otherwise, $g_{\mu_0, \tau^2}(x)$ $\approx \mu_0$.

HERE: JS EXAMPLE FOR NORMAL

### Minimax estimator

For an estimator $g$

$$ R^*(g) = \sup_{\vartheta \in \Theta} R(\vartheta, g)$$

is called the **maximum risk** and

$$ R^*(g^*) = \inf_{g \in \mathcal{K}} R^*(g) $$

is **minimax risk** and corresponding $g$ - **minimax estimator**. The use of minimax estimator is aimed at protecting against large losses. Also it's not hard to see, that

$$ R^*(g) = \sup_{\pi \in \mathcal{M}} R(\pi, g), $$

where $\mathcal{M}$ is a set of all prior measures $\pi$. If for some $\pi^*$ we have

$$ \inf_{g \in \mathcal{K}} R(\pi^*, g) \geq \inf_{g \in \mathcal{K}} R(\pi, g) \quad \forall \pi \in \mathcal{M}, $$

then $\pi^*$ is called the **least favorable prior**.

THEOREM

Let's get back to an example with binomial distribution:

$$ P_\vartheta(X = x) = \binom{n}{x} \vartheta^x (1-\vartheta)^{n-x}. $$

Again we use quadratic loss, but only this time we take parameterized beta distrubution $B(a, b)$ as our prior:

$$ h(\vartheta) = \frac{\vartheta^{a-1}(1-\vartheta)^{b-1}1_{[0,1]}(\vartheta)}{B(a, b)}. $$

Note that for $a = b = 1$ we have $\theta \sim \mathcal{U}(0, 1)$. Now posterior distribution will be $Q^{\vartheta \mid X=x} \sim B(x+a,n-x+b)$ with density

$$  f(\vartheta | x)= \frac{\vartheta^{x+a-1}(1-\vartheta)^{n-x+b-1}1_{[0,1](\vartheta)}}{B(x+a,n-x+b)}. $$

We pretend that we know (or it's not hard to show) that for random variable $Z \sim B(p, q)$

$$ \mathbb{E}[Z] = \frac{p}{p+q} \quad \text{and} \quad \operatorname{Var}(Z)=\frac{pq}{(p+q)^2(p+q+1)}. $$

Recall that for quadratic loss expected value of $\theta$ is Bayes estimator. Therefore,

$$ g_{a,b}(x)=\frac{x+a}{n+a+b} $$

is a Bayes estimator and it provides risk

$$ \begin{aligned} R(\vartheta, g_{a,b})&=\mathbb{E}[(g_{a,b}(X)-\vartheta)^2] \\ &=\frac{\vartheta^2(-n+(a+b)^2+\vartheta(n-2a(a+b))+a^2}{(n+a+b)^2}. \end{aligned}$$

If we choose $\hat{a}=\hat{b}=\frac{\sqrt{n}}{2}$ then risk will be

$$  R(\vartheta, g_{\hat{a}, \hat{b}})=\frac{1}{4(\sqrt{n} + 1)^2}. $$

Such risk doesn't depend on $\vartheta$ and hence an estimator $g_{\hat{a}, \hat{b}}(x) = \frac{x+\sqrt{n}/2}{n+\sqrt{n}}$ is minimax and $B(\hat{a}, \hat{b})$ is least favorable prior.


<script src="https://d3js.org/d3.v4.min.js"></script>

<link href="https://fonts.googleapis.com/css?family=Arvo" rel="stylesheet">

<div id="bin_bayes_plt"></div>

<input type="range" name="n_slider" id=n_slider min="1" max="10" value="8">

<script>

d3.json("../../../../assets/beta.json", function(error, data) {
  if (error) throw error;
  var sample = 1;
  var n = 8;
  
var margin = {top: 25, right: 0, bottom: 25, left: 25},
    width = 800 - margin.left - margin.right,
    height = 200 - margin.top - margin.bottom,
    fig_width = 200;
    
var prior_svg = d3.select("#bin_bayes_plt")
  .append("svg")
  .attr("width", width + margin.left + margin.right)
  .attr("height", height + margin.top + margin.bottom)
  .append("g")
  .attr("transform", "translate(" + margin.left + "," + margin.top + ")");

var prior_data = [
   {x: -0.05, y: 0},
   {x: 0, y: 0},
   {x: 0, y: 1},
   {x: 1, y: 1},
   {x: 1, y: 0},
   {x: 1.05, y: 0}
];

var x = d3.scaleLinear()
        .domain([d3.min(prior_data, function(d) { return d.x }), d3.max(prior_data, function(d) { return d.x }) ])
        .range([0, fig_width]);
        
prior_svg.append("g")
  .attr("transform", "translate(0," + height + ")")
  .call(d3.axisBottom(x).ticks(4));

var y = d3.scaleLinear()
        .range([height, 0])
        .domain([0, 12]);

prior_svg.append("g").call(d3.axisLeft(y).ticks(3));
  
var prior_curve = prior_svg
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
      
  prior_svg
    .append("text")
    .attr("text-anchor", "start")
    .attr("y", 40)
    .attr("x", 80)
    .attr("font-family", "Arvo")
    .attr("font-weight", 700)
    .text("Prior")
    .style("fill", "#348ABD");
      
    margin = {top: 0, right: 0, bottom: 35, left: 250};

	 var smpl_svg = prior_svg
	  .append("svg")
	    .attr("width", width + margin.left + margin.right)
	    .attr("height", height + margin.top + margin.bottom)
	  .append("g")
	    .attr("transform",
	          "translate(" + margin.left + "," + margin.top + ")");
	 
	 var smpl_x = d3.scaleBand()
	        .range([0, fig_width]);
	 var smpl_x_axis = smpl_svg.append("g")
	  .attr("transform", "translate(0," + height + ")");
	 var smpl_y = d3.scaleLinear().range([height, 0]).domain([0, 1]);
	 var smpl_y_axis = smpl_svg.append("g").call(d3.axisLeft(smpl_y).ticks(0)); 
    
    
  function updateRectSample() { 
    var rect_data = [];
    for (var i = 0; i <= n; i++) {
       rect_data.push({x: i, y: 1});
    }

	 smpl_x.domain(rect_data.map(function(d) { return d.x; }));
	 smpl_x_axis.call(d3.axisBottom(smpl_x));
	
	 var rect_sample = smpl_svg.selectAll("rect").data(rect_data);
	  
    rect_sample.enter()
	    .append("rect")
	      .merge(rect_sample)
	      .attr("x", function(d) { return smpl_x(d.x); })
	      .attr("y", function(d) { return smpl_y(d.y); })
	      .attr("width", smpl_x.bandwidth())
	      .attr("border", 0)
	      .attr("opacity", function(d) { return d.x == sample ? ".8" : "0"; })
	      .attr("stroke", "#000")
	      .attr("stroke-width", 1)
	      .attr("stroke-linejoin", "round")
	      .attr("height", function(d) { return height - smpl_y(d.y); })
	      .attr("fill", "#65AD69")
	      .on('mouseover', function(d, i) {
	        d3.select(this)
	          .transition()
	          .attr("opacity", function(d) { return d.x == sample ? ".8" : ".4"; });
	      })
	      .on('mouseout', function(d, i) {
	        d3.select(this)
	          .transition()
	          .attr("opacity", function(d) { return d.x == sample ? ".8" : "0"; });
	      })
	      .on('click', function(d, i) {
	        sample = i;
	        d3.selectAll("rect")
	          .transition()
		       .attr("x", function(d) { return smpl_x(d.x); })
		       .attr("y", function(d) { return smpl_y(d.y); })
	          .attr("opacity", function(d) { return d.x == sample ? ".8" : "0"; });
	        updatePosteriorCurve();
	    });
    
     rect_sample.exit().remove();
  }
    	
    updateRectSample();
	    
  smpl_svg
    .append("text")
    .attr("text-anchor", "start")
    .attr("transform", "rotate(270)")
    .attr("y", -7)
    .attr("x", -100)
    .attr("font-family", "Arvo")
    .attr("font-weight", 700)
    .text("Sample")
    .style("fill", "#65AD69");
    
margin = {top: 0, right: 0, bottom: 35, left: 250};
    
var post_svg = smpl_svg
  .append("svg")
    .attr("width", width + margin.left + margin.right)
    .attr("height", height + margin.top + margin.bottom)
  .append("g")
    .attr("transform",
          "translate(" + margin.left + "," + margin.top + ")");
        
  post_svg.append("g")
    .attr("transform", "translate(0, " + height + ")")
    .call(d3.axisBottom(x).ticks(4));
  
  post_svg.append("g").call(d3.axisLeft(y).ticks(3));
  
  post_svg
    .append("text")
    .attr("text-anchor", "start")
    .attr("y", 55)
    .attr("x", 65)
    .attr("font-family", "Arvo")
    .attr("font-weight", 700)
    .text("Posterior")
    .style("fill", "#EDA137");
    
  post_svg
    .append("text")
    .attr("text-anchor", "start")
    .attr("y", 55)
    .attr("x", 172)
    .attr("font-family", "Arvo")
    .attr("font-weight", 700)
    .attr("font-size", 10)
    .text("UMVU")
    .style("fill", "#E86456");
      
  post_svg.append("path")
        .attr("class", "line")
        .style("stroke-dasharray", ("3, 3"))
        .attr("stroke", "#000")
        .attr("stroke-width", 1)
        .datum([{x: 165, y: 45}, {x: 165, y: 60}])
        .attr("d",  d3.line()
          .x(function(d) { return d.x; })
          .y(function(d) { return d.y; }));
  
  post_svg.append('g')
    .selectAll("dot")
    .data([{x: 165, y: 45}])
    .enter()
    .append("circle")
      .attr("cx", function (d) { return d.x; } )
      .attr("cy", function (d) { return d.y; } )
      .attr("r", 3)
      .style("fill", "#E86456")
      .attr("stroke", "#000")
      .attr("stroke-width", 1);
    
  post_svg
    .append("text")
    .attr("text-anchor", "start")
    .attr("y", 85)
    .attr("x", 172)
    .attr("font-family", "Arvo")
    .attr("font-weight", 700)
    .attr("font-size", 10)
    .text("Bayes")
    .style("fill", "#348ABD");
      
  post_svg.append("path")
        .attr("class", "line")
        .style("stroke-dasharray", ("3, 3"))
        .attr("stroke", "#000")
        .attr("stroke-width", 1)
        .datum([{x: 165, y: 75}, {x: 165, y: 90}])
        .attr("d",  d3.line()
          .x(function(d) { return d.x; })
          .y(function(d) { return d.y; }));
  
  post_svg.append('g')
    .selectAll("dot")
    .data([{x: 165, y: 75}])
    .enter()
    .append("circle")
      .attr("cx", function (d) { return d.x; } )
      .attr("cy", function (d) { return d.y; } )
      .attr("r", 3)
      .style("fill", "#348ABD")
      .attr("stroke", "#000")
      .attr("stroke-width", 1);
      
  post_svg
    .append("text")
    .attr("text-anchor", "start")
    .attr("y", 115)
    .attr("x", 172)
    .attr("font-family", "Arvo")
    .attr("font-weight", 700)
    .attr("font-size", 10)
    .text("Minimax")
    .style("fill", "#F5CC18");
      
  post_svg.append("path")
        .attr("class", "line")
        .style("stroke-dasharray", ("3, 3"))
        .attr("stroke", "#000")
        .attr("stroke-width", 1)
        .datum([{x: 165, y: 105}, {x: 165, y: 120}])
        .attr("d",  d3.line()
          .x(function(d) { return d.x; })
          .y(function(d) { return d.y; }));
  
  post_svg.append('g')
    .selectAll("dot")
    .data([{x: 165, y: 105}])
    .enter()
    .append("circle")
      .attr("cx", function (d) { return d.x; } )
      .attr("cy", function (d) { return d.y; } )
      .attr("r", 3)
      .style("fill", "#F5CC18")
      .attr("stroke", "#000")
      .attr("stroke-width", 1);
          
  var posterior_data = [];
  updatePosteriorData();
        
  var posterior_curve = post_svg
    .append('g')
    .append("path")
      .datum(posterior_data)
      .attr("fill", "#EDA137")
      .attr("border", 0)
      .attr("opacity", ".8")
      .attr("stroke", "#000")
      .attr("stroke-width", 1)
      .attr("stroke-linejoin", "round")
      .attr("d",  d3.line()
          .curve(d3.curveBasis)
          .x(function(d) { return x(d.x); })
          .y(function(d) { return y(d.y); })
      );
     
  var umvu_x = sample / n;
  var umvu_y = Math.pow(umvu_x, sample) * Math.pow(1-umvu_x, n-sample) / data[n][sample];
      
  var umvu_dash = post_svg.append("path")
        .attr("class", "line")
        .style("stroke-dasharray", ("3, 3"))
        .attr("stroke", "#000")
        .attr("stroke-width", 1)
        .datum([{x: umvu_x, y: umvu_y}, {x: umvu_x, y: 0}])
        .attr("d",  d3.line()
          .x(function(d) { return x(d.x); })
          .y(function(d) { return y(d.y); }));
      
  var umvu_dot = post_svg.append('g')
    .selectAll("dot")
    .data([{x: umvu_x, y: umvu_y}])
    .enter()
    .append("circle")
      .attr("cx", function (d) { return x(d.x); } )
      .attr("cy", function (d) { return y(d.y); } )
      .attr("r", 3)
      .style("fill", "#E86456")
      .attr("stroke", "#000")
      .attr("stroke-width", 1);
      
  var bayes_x = (sample + 1) / (n + 2);
  var bayes_y = Math.pow(bayes_x, sample) * Math.pow(1-bayes_x, n-sample) / data[n][sample];
    
  var bayes_dash = post_svg.append("path")
        .attr("class", "line")
        .style("stroke-dasharray", ("3, 3"))
        .attr("stroke", "#000")
        .attr("stroke-width", 1)
        .datum([{x: bayes_x, y: bayes_y}, {x: bayes_x, y: 0}])
        .attr("d",  d3.line()
          .x(function(d) { return x(d.x); })
          .y(function(d) { return y(d.y); }));
      
  var bayes_dot = post_svg.append('g')
    .selectAll("dot")
    .data([{x: bayes_x, y: bayes_y}])
    .enter()
    .append("circle")
      .attr("cx", function (d) { return x(d.x); } )
      .attr("cy", function (d) { return y(d.y); } )
      .attr("r", 3)
      .style("fill", "#348ABD")
      .attr("stroke", "#000")
      .attr("stroke-width", 1);
      
  var minimax_x = (sample + Math.sqrt(n) / 2) / (n + Math.sqrt(n));
  var minimax_y = Math.pow(minimax_x, sample) * Math.pow(1-minimax_x, n-sample) / data[n][sample];
      
  var minimax_dash = post_svg.append("path")
        .attr("class", "line")
        .style("stroke-dasharray", ("3, 3"))
        .attr("stroke", "#000")
        .attr("stroke-width", 1)
        .datum([{x: minimax_x, y: minimax_y}, {x: minimax_x, y: 0}])
        .attr("d",  d3.line()
          .x(function(d) { return x(d.x); })
          .y(function(d) { return y(d.y); }));
      
  var minimax_dot = post_svg.append('g')
    .selectAll("dot")
    .data([{x: minimax_x, y: minimax_y}])
    .enter()
    .append("circle")
      .attr("cx", function (d) { return x(d.x); } )
      .attr("cy", function (d) { return y(d.y); } )
      .attr("r", 3)
      .style("fill", "#F5CC18")
      .attr("stroke", "#000")
      .attr("stroke-width", 1);
  
  function updatePosteriorData() {
    posterior_data = [];
    for (var i = -0.05; i < 0; i += 0.01) {
      posterior_data.push({x: i, y: 0});
    }
    posterior_data.push({x: 0, y: 0});
  
    for (var i = 0; i < 1; i += 0.01) {
  	   posterior_data.push({x: i, y: Math.pow(i, sample) * Math.pow(1-i, n-sample) / data[n][sample] });
    }

    posterior_data.push({x: 1, y: (sample < n ? 0 : 1) / data[n][sample] });
	
    for (var i = 1; i <= 1.05; i += 0.01) {
	   posterior_data.push({x: i, y: 0});
    }
    
    umvu_x = sample / n;
    umvu_y = Math.pow(umvu_x, sample) * Math.pow(1-umvu_x, n-sample) / data[n][sample];
    
    bayes_x = (sample + 1) / (n + 2);
    bayes_y = Math.pow(bayes_x, sample) * Math.pow(1-bayes_x, n-sample) / data[n][sample];
    
    minimax_x = (sample + Math.sqrt(n) / 2) / (n + Math.sqrt(n));
    minimax_y = Math.pow(minimax_x, sample) * Math.pow(1-minimax_x, n-sample) / data[n][sample];
  }
  
	function updatePosteriorCurve() {
	  updatePosteriorData();
	  
	  posterior_curve
	    .datum(posterior_data)
	    .transition()
	    .duration(1000)
	    .attr("d",  d3.line()
	      .curve(d3.curveBasis)
	      .x(function(d) { return x(d.x); })
	      .y(function(d) { return y(d.y); })
	  );
	  
     umvu_dot	
       .transition()
	    .duration(1000)
       .attr("cx", function (d) { return x(umvu_x); } )
       .attr("cy", function (d) { return y(umvu_y); } );
       
	 umvu_dash
	    .datum([{x: umvu_x, y: 0}, {x: umvu_x, y: umvu_y}])
       .transition()
	    .duration(1000)
	    .attr("d",  d3.line()
	      .curve(d3.curveBasis)
	      .x(function(d) { return x(d.x); })
	      .y(function(d) { return y(d.y); })
	    );
	    
    bayes_dot	
      .transition()
      .duration(1000)
      .attr("cx", function (d) { return x(bayes_x); } )
      .attr("cy", function (d) { return y(bayes_y); } );
        
	bayes_dash
	   .datum([{x: bayes_x, y: 0}, {x: bayes_x, y: bayes_y}])
      .transition()
      .duration(1000)
      .attr("d",  d3.line()
         .curve(d3.curveBasis)
         .x(function(d) { return x(d.x); })
         .y(function(d) { return y(d.y); })
      );
	    
    minimax_dot	
      .transition()
      .duration(1000)
      .attr("cx", function (d) { return x(minimax_x); } )
      .attr("cy", function (d) { return y(minimax_y); } );
        
	minimax_dash
	   .datum([{x: minimax_x, y: 0}, {x: minimax_x, y: minimax_y}])
      .transition()
      .duration(1000)
      .attr("d",  d3.line()
         .curve(d3.curveBasis)
         .x(function(d) { return x(d.x); })
         .y(function(d) { return y(d.y); })
      );

	}
  
  function updateN(value) {
    n = parseInt(value);
    sample = Math.min(sample, n);
    updateRectSample();
    updatePosteriorCurve();
  }
	
  d3.select("#n_slider").on("change", function(d) {
    selectedValue = this.value;
    updateN(selectedValue);
  })

});

</script>