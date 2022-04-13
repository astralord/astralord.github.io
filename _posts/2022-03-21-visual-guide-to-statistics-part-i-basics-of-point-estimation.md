---
layout: post
title: 'Visual Guide to Statistics. Part I: Basics of Point Estimation'
date: 2022-03-21 03:13 +0800
categories: [Statistics]
tags: [statistics, parameter-estimation, frequentist-inference, exponential-family, cramer-rao-inequality, fisher-information]
math: true
---

<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.13.13/dist/katex.min.css" integrity="sha384-RZU/ijkSsFbcmivfdRBQDtwuwVqK7GMOw6IMvKyeWL2K5UAlyp6WonmB8m7Jd0Hn" crossorigin="anonymous">
  <script defer src="https://cdn.jsdelivr.net/npm/katex@0.13.13/dist/katex.min.js" integrity="sha384-pK1WpvzWVBQiP0/GjnvRxV4mOb0oxFuyRxJlk6vVw146n3egcN5C925NCP7a7BY8" crossorigin="anonymous">
  </script>
  <script defer src="https://cdn.jsdelivr.net/npm/katex@0.13.13/dist/contrib/auto-render.min.js" integrity="sha384-vZTG03m+2yp6N6BNi5iM4rW4oIwk5DfcNdFfxkk9ZWpDriOkXX8voJBFrAO7MpVl" crossorigin="anonymous"
        onload="renderMathInElement(document.body);">
</script>
  
> This series of posts is a guidance for those who already have knowledge in probability theory and would like to become familiar with mathematical statistics. Basically, these are notes from lectures I attended while being a student in Christian-Albrechts University in Kiel, Germany. They helped me close all the gaps in my knowledge of math under the hood of modern statistics. For those who are interested in the lectures themselves can refer to the [original material](https://github.com/astralord/Statistics-lectures/blob/master/vorlesungonline.pdf) or my [translation to Russian](https://github.com/astralord/Statistics-lectures/blob/master/book.pdf).
> 
> This post in particular focuses on point estimators of distribution parameters and their characteristics.


### Intro
Imagine that you are a pharmaceutical company, which is about to introduce a new drug into production. Prior to launch you need to carry out experiments to assess its quality depending on the dosage. Say you give this medicine to an animal, after which the animal is examined and checked whether it has recovered or not by taking a dose of $X$. You can think of the result as random variable $Y$ following Bernoulli distribution:

$$ Y \sim \operatorname{Bin}(1, p(X)), $$

where $p(X)$ is a probability of healing given dose $X$. 

Typically, several independent experiments $Y_1, \dots, Y_n$ with different doses $X_1, \dots, X_n$ are made, such that

$$ Y_i \sim \operatorname{Bin}(1, p(X_i)). $$ 
	
Our goal is to estimate function $p: [0, \infty) \rightarrow [0, 1]$. For example, we can simplify to parametric model

$$ p(x) = 1 - e^{-\vartheta x}, \quad \vartheta > 0. $$

Then estimating $p(x)$ is equal to estimating parameter $\vartheta $.

<style>

.svg-container {
  display: inline-block;
  position: relative;
  width: 100%;
  padding-bottom: 100%;
  vertical-align: top;
  overflow: hidden;
}

.svg-content-responsive {
  display: inline-block;
  position: absolute;
  top: 10px;
  left: 0;
}

.ticks {
  font: 10px arvo;
}

.track,
.track-inset,
.track-overlay {
  stroke-linecap: round;
}

.track {
  stroke: #000;
  stroke-opacity: 0.8;
  stroke-width: 7px;
}

.track-inset {
  stroke: #ddd;
  stroke-width: 5px;
}

.track-overlay {
  pointer-events: stroke;
  stroke-width: 50px;
  stroke: transparent;
}

.handle {
  fill: #fff;
  stroke: #000;
  stroke-opacity: 0.8;
  stroke-width: 1px;
}

#sample-button {
  top: 15px;
  left: 15px;
  background: #65AD69;
  padding-right: 26px;
  border-radius: 3px;
  border: none;
  color: white;
  margin: 0;
  padding: 0 1px;
  width: 60px;
  height: 25px;
  font-family: Arvo;
  font-size: 11px;
}

#sample-button:hover {
  background-color: #696969;
}

#sample-button-2 {
  top: 15px;
  left: 15px;
  background: #65AD69;
  padding-right: 26px;
  border-radius: 3px;
  border: none;
  color: white;
  margin: 0;
  padding: 0 1px;
  width: 60px;
  height: 25px;
  font-family: Arvo;
  font-size: 11px;
}

#sample-button-2:hover {
  background-color: #696969;
}

#reset-button {
  top: 15px;
  left: 15px;
  background: #E86456;
  padding-right: 26px;
  border-radius: 3px;
  border: none;
  color: white;
  margin: 0;
  padding: 0 1px;
  width: 60px;
  height: 25px;
  font-family: Arvo;
  font-size: 11px;
}

#reset-button:hover {
  background-color: #696969;
}
   
</style>

<script src="https://d3js.org/d3.v4.min.js"></script>
<link href="https://fonts.googleapis.com/css?family=Arvo" rel="stylesheet">

<button id="sample-button">Sample</button>
<div id="drug_exp"></div> 

<script>

d3.select("#drug_exp")
  .style("position", "relative");
  
function drug_exp() {
var theta = 0.2;

var margin = {top: 10, right: 0, bottom: 10, left: 30},
    width = 750 - margin.left - margin.right,
    height = 150 - margin.top - margin.bottom,
    fig_height = 125 - margin.top - margin.bottom,
    fig_width = 650;
    
var svg = d3.select("div#drug_exp")
  .append("svg")
    .attr("width", width + margin.left + margin.right)
    .attr("height", height + margin.top + margin.bottom)
  .append("g")
    .attr("transform",
          "translate(" + margin.left + "," + margin.top + ")");

var x = d3.scaleLinear()
          .domain([0, 10])
          .range([10, fig_width]);
            
var xAxis = svg.append("g")
   .attr("transform", "translate(0," + fig_height + ")")
   .call(d3.axisBottom(x));
  
xAxis.selectAll(".tick text")
   .attr("font-family", "Arvo");

var y = d3.scaleLinear()
          .range([fig_height - 10, 0])
          .domain([0, 1]);
            
var yAxis = svg.append("g")
    .call(d3.axisLeft(y).ticks(1));
  
yAxis.selectAll(".tick text")
    .attr("font-family", "Arvo");
    
d3.select("#drug_exp")
  .append("div")
  .text("Dose \\(X_i \\)")
  .style('color', '#696969')
  .style("font-size", "13px")
  .style("font-weight", "700")
  .attr("font-family", "Arvo")
  .attr("font-weight", 700)
  .style("position", "absolute")
  .style("left", fig_width / 2 + margin.left - 10 + "px")
  .style("top", fig_height + margin.top + 15 + "px");
  
d3.select("#drug_exp")
  .append("div")
  .text("\\(Y_i \\)")
  .style('color', '#696969')
  .style("font-size", "13px")
  .style("font-weight", "700")
  .attr("font-family", "Arvo")
  .attr("font-weight", 700)
  .style("position", "absolute")
  .style("left", 10 + "px")
  .style("top", fig_height / 2 - 5 + "px");
    
var figs = [];
for (var i = 0; i < 11; i += 1) {
  if (Math.random() < 1 - Math.exp(-theta * i)) {
    figs.push(svg.append("path")
    .attr("d", d3.symbol().type(d3.symbolCross).size(100))
    .attr("transform", function(d) { return "translate(" + x(i) + "," + y(1) + ")"; })
    .style("fill", "#65AD69")
    .style('stroke', 'black')
    .style('stroke-width', '0.7')
    .style('opacity', 0.8));
  }
  else {
    figs.push(svg.append("path")
    .attr("d", d3.symbol().type(d3.symbolCross).size(100))
    .attr("transform", function(d) { return "translate(" + x(i) + "," + y(0) + ") rotate(-45)"; })
    .style("fill", "#E86456")
    .style('stroke', 'black')
    .style('stroke-width', '0.7')
    .style('opacity', 0.8));
  }
}
    
function updateSymbols() {
  for (var i = 0; i < 11; i += 1) {
    if (Math.random() < 1 - Math.exp(-theta * i)) {
      figs[i].transition()
             .duration(1000)
             .attr("d", d3.symbol().type(d3.symbolCross).size(100))
             .attr("transform", function(d) { return "translate(" + x(i) + "," + y(1) + ")"; })
             .style("fill", "#65AD69");
    }
    else {
      figs[i].transition()
             .duration(1000)
             .attr("d", d3.symbol().type(d3.symbolCross).size(100))
             .attr("transform", function(d) { return "translate(" + x(i) + "," +y(0) + ") rotate(-45)"; })
             .style("fill", "#E86456");
    }
  }
}

var sampleButton = d3.select("#sample-button");

sampleButton
    .on("click", function() {
      updateSymbols();
});

}

drug_exp();

</script>

![](.)
*Fig. 1. Visualization of statistical experiments. The question arises: how do we estimate the value of $\vartheta$ based on our observations?*

Formally, we can define **parameter space** $\Theta$ with $\vert \Theta \vert \geq 2$ and family of probability measures $\mathcal{P} = \lbrace P_\vartheta \mid \vartheta \in \Theta \rbrace$, where $P_\vartheta \neq P_{\vartheta'} \ \forall \vartheta \neq \vartheta'$. Then we are interested in the true distribution $P \in \mathcal{P}$ of random variable $X$. 

Recall from probability theory that random variable $X$ is a mapping from set of all possible outcomes $\Omega$ to a **sample space** $\mathcal{X}$. On the basis of given sample $x = X(\omega)$, $\omega \in \Omega$ we make a decision about the unknown $P$. By identifying family $\mathcal{P}$ with the parameter space $\Theta$, a decision for $P$ is equivalent to a decision for $\vartheta$. In our example above

$$ Y_i \sim \operatorname{Bin}(1, 1 - e^{-\vartheta X_i}) = P_\vartheta^i $$ 

and

$$ \mathcal{X} = \{0, 1\}^n, \quad \Theta=\left[0, \infty\right), \quad \mathcal{P}=\{\otimes_{i=1}^nP_{\vartheta}^i \mid \vartheta>0 \}. $$


### Uniformly best estimator

Mandatory parameter estimation example which can be found in every statistics handbook is mean and variance estimation for Normal distribution. Let $X_1, \dots, X_n$ i.i.d. $\sim \mathcal{N}(\mu, \sigma^2) = P_{\mu, \sigma^2}$. The typical estimation for $\vartheta = (\mu, \sigma^2)$ would be

$$ g(x) = \begin{pmatrix} \overline{x}_n \\ \hat{s}_n^2 \end{pmatrix} = \begin{pmatrix} \frac{1}{n} \sum_{i=1}^n x_i \\ \frac{1}{n} \sum_{i=1}^n (x_i-\overline{x}_n)^2 \end{pmatrix}. $$

We will get back to characteristics of this estimation later. But now it is worth noting that we are not always interested in $\vartheta$ itself, but in an appropriate functional $\gamma(\vartheta)$. We can see it in another example.

Let $X_1, \dots, X_n$ i.i.d. $\sim F$, where $F(x) = \mathbb{P}(X \leq x)$ is unknown distribution function. Here $\Theta$ is an infinite-dimensional family of distribution functions. Say we are interested in value of this function at point $k$: 

$$\gamma(F) = F(k).$$ 

Then a point estimator could be $g(x) = \frac{1}{n} \sum_{i=1}^n \mathbf{1}_{\lbrace X_i \leq k \rbrace }$.

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
 
It is reasonable (at least at the start) to put constraint on unbiasedness for $g$ and search only in 

$$ \mathcal{E}_\gamma = \lbrace g \in \mathcal{K} \mid B_\vartheta(g) = 0 \rbrace.$$

Surely there can be infinite number of unbiased estimators, and we not only interested in expected value of $g$, but also in how $g$ can vary from it. Variance of $g$ can be chosen as our metric for goodness. We call estimator $\tilde{g}$ **uniformly minimum variance unbiased (UMVU)** if

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

<div id="chi_t_plt"></div> 

<script>

d3.select("#chi_t_plt")
  .style("position", "relative");
  
function plt_label_path(svg, color, x, y) {

	svg.append("path")
	   .attr("stroke", color)
	   .attr("stroke-width", 4)
	   .attr("opacity", ".8")
	   .datum([{x: x, y: y + 2}, {x: x + 25, y: y + 2}])
	   .attr("d",  d3.line()
	       .x(function(d) { return d.x; })
	       .y(function(d) { return d.y; }));
	       
	svg.append("path")
	   .attr("stroke", "#000")
	   .attr("stroke-width", 1)
	   .datum([{x: x, y: y}, {x: x + 25, y: y}])
	   .attr("d",  d3.line()
	       .x(function(d) { return d.x; })
	       .y(function(d) { return d.y; }));
	       
}

function chi_t_plts() {

var margin = {top: 20, right: 0, bottom: 30, left: 50},
    width = 700 - margin.left - margin.right,
    height = 200 - margin.top - margin.bottom,
    fig_width = 300;

var chi_svg = d3.select("#chi_t_plt")
  .append("svg")
    .attr("width", width + margin.left + margin.right)
    .attr("height", height + margin.top + margin.bottom)
  .append("g")
    .attr("transform",
          "translate(" + margin.left + "," + margin.top + ")");

var margin = {top: 0, right: 0, bottom: 35, left: 350};
    
var t_svg = chi_svg
  .append("svg")
    .attr("width", width + margin.left + margin.right)
    .attr("height", height + margin.top + margin.bottom)
  .append("g")
    .attr("transform",
          "translate(" + margin.left + "," + margin.top + ")");

plt_label_path(chi_svg, "#EDA137", fig_width * 0.72, height * 0.13);

var span_chi = d3.select("#chi_t_plt")
  .append("span")
  .text("\\(f_{\\chi_n^2}(x)\\)")
  .style('color', '#EDA137')
  .style("font-size", "17px")
  .style("font-weight", "700")
  .attr("font-family", "Arvo")
  .attr("font-weight", 700)
  .attr("font-size", 20)
  .style("position", "absolute")
  .style("left", fig_width * 0.98 + "px")
  .style("top", height * 0.18 + "px");
  

plt_label_path(t_svg, "#348ABD", fig_width * 0.72, height * 0.13);

var span_t = d3.select("#chi_t_plt")
  .append("div")
  .text("\\(f_{t_n}(x) \\)")
  .style('color', '#348ABD')
  .style("font-size", "17px")
  .style("font-weight", "700")
  .attr("font-family", "Arvo")
  .attr("font-weight", 700)
  .attr("font-size", 20)
  .style("position", "absolute")
  .style("left", fig_width * 2.145 + "px")
  .style("top", height * 0.18 + "px");
    
d3.csv("../../../../assets/chi-t.csv", function(error, data) {
  if (error) throw error;

  var chi_x = d3.scaleLinear()
            .domain([-0, 40])
            .range([0, fig_width]);
            
  var xAxis = chi_svg.append("g")
      .attr("transform", "translate(0," + height + ")")
      .call(d3.axisBottom(chi_x));
  
  xAxis.selectAll(".tick text")
     .attr("font-family", "Arvo");

  var t_x = d3.scaleLinear()
            .domain([-20, 20])
            .range([0, fig_width]);
            
  xAxis = t_svg.append("g")
      .attr("transform", "translate(0," + height + ")")
      .call(d3.axisBottom(t_x));
       
  xAxis.selectAll(".tick text")
     .attr("font-family", "Arvo");

  var y = d3.scaleLinear()
            .range([height, 0])
            .domain([0, 0.5]);
            
  var yAxis = chi_svg.append("g")
      .call(d3.axisLeft(y).ticks(5));
  
  yAxis.selectAll(".tick text")
     .attr("font-family", "Arvo");
     
  var t_y = d3.scaleLinear()
            .range([height, 5])
            .domain([0, 0.5]);
                
  yAxis = t_svg.append("g")
      .call(d3.axisLeft(t_y).ticks(5));
      
  yAxis.selectAll(".tick text")
     .attr("font-family", "Arvo");

  var chi_curve = chi_svg
    .append('g')
    .append("path")
      .datum(data)
      .attr("fill", "#EDA137")
      .attr("border", 0)
      .attr("opacity", ".8")
      .attr("stroke", "#000")
      .attr("stroke-width", 1)
      .attr("stroke-linejoin", "round")
      .attr("d",  d3.line()
        .curve(d3.curveBasis)
          .x(function(d) { return chi_x(d.chi_x); })
          .y(function(d) { return y(d["chi_5"]); })
      );
      
  var t_curve = t_svg
    .append('g')
    .append("path")
      .datum(data)
      .attr("fill", "#348ABD")
      .attr("border", 0)
      .attr("opacity", ".8")
      .attr("stroke", "#000")
      .attr("stroke-width", 1)
      .attr("stroke-linejoin", "round")
      .attr("d",  d3.line()
        .curve(d3.curveBasis)
          .x(function(d) { return t_x(d.t_x); })
          .y(function(d) { return t_y(d["t_5"]); })
      );

  function updateChart(n) {
    n = parseInt(n);
    
    chi_curve
      .datum(data)
      .transition()
      .duration(1000)
      .attr("d",  d3.line()
        .curve(d3.curveBasis)
          .x(function(d) { return chi_x(d.chi_x); })
          .y(function(d) { return y(d["chi_" + n]); })
      );
      
    t_curve
      .datum(data)
      .transition()
      .duration(1000)
      .attr("d",  d3.line()
        .curve(d3.curveBasis)
          .x(function(d) { return t_x(d.t_x); })
          .y(function(d) { return t_y(d["t_" + n]); })
      );
  }
  
var slider_svg = d3.select("#chi_t_plt")
  .append("svg")
  .attr("width", width + 20)
  .attr("height", 70)
  .append("g")
  .attr("transform", "translate(" + 25 + "," + 20 + ")");

var n_x = d3.scaleLinear()
    .domain([1, 12])
    .range([0, width / 2])
    .clamp(true);
    
function roundN(x) { return Math.round(x - 0.5); }

function createSlider(svg_, parameter_update, x, loc_x, loc_y, letter, color, init_val, round_fun) {
    var slider = svg_.append("g")
      .attr("class", "slider")
      .attr("transform", "translate(" + loc_x + "," + loc_y + ")");
    
    var drag = d3.drag()
	        .on("start.interrupt", function() { slider.interrupt(); })
	        .on("start drag", function() { 
	          handle.attr("cx", x(round_fun(x.invert(d3.event.x))));  
	          parameter_update(round_fun(x.invert(d3.event.x)));
	         });
	         
    slider.append("line")
	    .attr("class", "track")
	    .attr("x1", x.range()[0])
	    .attr("x2", x.range()[1])
	  .select(function() { return this.parentNode.appendChild(this.cloneNode(true)); })
	    .attr("class", "track-inset")
	  .select(function() { return this.parentNode.appendChild(this.cloneNode(true)); })
	    .attr("class", "track-overlay")
	    .call(drag);

	slider.insert("g", ".track-overlay")
    .attr("class", "ticks")
    .attr("transform", "translate(0," + 18 + ")")
  .selectAll("text")
  .data(x.ticks(6))
  .enter().append("text")
    .attr("x", x)
    .attr("text-anchor", "middle")
    .attr("font-family", "Arvo")
    .text(function(d) { return d; });

   var handle = slider.insert("circle", ".track-overlay")
      .attr("class", "handle")
      .attr("r", 6).attr("cx", x(init_val));
      
	svg_
	  .append("text")
	  .attr("text-anchor", "middle")
	  .attr("y", loc_y + 3)
	  .attr("x", loc_x - 21)
	  .attr("font-family", "Arvo")
	  .attr("font-size", 17)
	  .text(letter)
	  .style("fill", color);
	  	  
	return handle;
}

createSlider(slider_svg, updateChart, n_x, 190, 0.1 * height, "n", "#696969", 5, roundN);

});
}

chi_t_plts();

</script>

![](.)
*Fig. 2. Probability density functions for $\chi_n^2$ and $t_n$-distributions. Move slider to observe how they look for different degrees of freedom $n$. Note that with large $n$ $t_n$ converges to normal distribution.*

It can now be shown that

$$ \overline{X}_n = \frac{1}{n} \sum_{i=1}^n X_i \sim \mathcal{N} \Big( \mu, \frac{\sigma^2}{n} \Big) $$

and

$$ \hat{s}_n^2(X) = \frac{1}{n}\sum_{i=1}^n (X_i - \overline{X}_n)^2 \sim \frac{\sigma^2}{n} \chi^2_{n-1}. $$

As a consequence:

$$ \frac{(n-1)(\overline{X}_n-\mu)}{\sqrt{n}s_n^2(X)} \sim t_{n-1}.$$

<details>
<summary>Proof</summary>
Distribution of $\overline{X}_n$ follows from properties of Normal distribution. Let

$$ Y_i = \frac{X_i - \mu}{\sigma} \sim \mathcal{N}(0, 1)$$

and $Y = (Y_1, \dots, Y_n)^T$. Choose orthogonal matrix $A$ such that its last row:

$$ v^T = \Big( \frac{1}{\sqrt{n}} \dots \frac{1}{\sqrt{n}} \Big).$$

Then for $Z = AY$ the following equality holds:

$$ \sum_{i=1}^n Z_i^2 = Z^TZ = Y^TA^TAY = Y^TY= \sum_{i=1}^n Y_i^2.$$

From $\operatorname{Cov}(Z)=A^TA = \mathbb{I}_n$ we have $Z \sim \mathcal{N}(0, \mathbb{I}_n).$ Also

$$ \begin{aligned}
	\sqrt{n} \overline{X}_n &= \frac{1}{\sqrt{n}} \sum_{i=1}^n (\sigma Y_i + \mu) \\ 
& = \sigma v^T Y + \sqrt{n} \mu \\
&= \sigma Z_n + \sqrt{n} \mu
	\end{aligned} $$
	
and

$$ \begin{aligned}
	n \hat{s}_n^2(X) &= \sum_{i=1}^n (X_i - \overline{X}_n)^2  = \sigma^2 \sum_{i=1}^n(Y_i - \overline{Y}_n)^2 \\
& = \sigma^2 \big(\sum_{i=1}^n Y_i^2 - n \overline{Y}_n^2\big) = \sigma^2 \big(\sum_{i=1}^n Y_i^2 -  \big(\frac{1}{n} \sum_{i=1}^n Y_i^2 \big)^2 \big) \\
& = \sigma^2 (\sum_{i=1}^n Z_i^2 - Z_n^2) = \sigma^2 \sum_{i=1}^{n-1} Z_i^2 \sim \chi_{n-1}^2.
	\end{aligned} $$

Both estimators are independent as functions of $Z_n$ and $Z_1, \dots, Z_{n-1}$ respectively.

</details>

Let's check which of these estimators are unbiased. We have $\mathbb{E}[\overline{X}_n] = \mu$, therefore $\overline{X}_n$ is unbiased. On the other hand

$$ \mathbb{E}[\hat{s}_n^2(X)] = \frac{\sigma^2}{n} (n - 1) \neq \sigma^2.$$

<button id="sample-button-2">Sample</button> <button id="reset-button">Reset</button>
<div id="biased_viz"></div> 

<script>

d3.select("#biased_viz")
  .style("position", "relative");
  
function biasedness() {

var mu = 0,
    sigma = 1,
    n = 6,
    xn_dots = [],
    sn_dots = [];

var avg_dur = 1200;

function randn_bm() {
    var u = 0, v = 0;
    while(u === 0) u = Math.random();
    while(v === 0) v = Math.random();
    return Math.sqrt( -2.0 * Math.log( u ) ) * Math.cos( 2.0 * Math.PI * v );
}

var margin = {top: 20, right: 0, bottom: 5, left: 70},
    width = 750 - margin.left - margin.right,
    height = 400 - margin.top - margin.bottom,
    fig_height = 250 - margin.top - margin.bottom,
    fig_width = 500,
    cfs = 100;
    
var svg = d3.select("#biased_viz")
  .append("svg")
    .attr("width", width + margin.left + margin.right)
    .attr("height", height + margin.top + margin.bottom)
  .append("g")
    .attr("transform",
          "translate(" + margin.left + "," + margin.top + ")");

var x = d3.scaleLinear()
          .domain([-4, 4])
          .range([cfs, fig_width + cfs]);
            
var xAxis = svg.append("g")
   .attr("transform", "translate(0," + fig_height + ")")
   .call(d3.axisBottom(x));
  
xAxis.selectAll(".tick text")
   .attr("font-family", "Arvo");

var y = d3.scaleLinear()
          .range([fig_height, 0])
          .domain([0, 5]);
            
var yAxis = svg.append("g")
   .attr("transform", "translate(" + cfs + ",0)")
   .call(d3.axisLeft(y).ticks(5));
  
yAxis.selectAll(".tick text")
    .attr("font-family", "Arvo");
  
var gauss_data = [{x: -4, y: 0}];
for (var i = -4; i < 4; i += 0.01) {
  gauss_data.push({x: i, y: Math.exp(-0.5 * ((i - mu) / sigma) ** 2) / (sigma * Math.sqrt(2 * Math.PI)) });
}
gauss_data.push({x: 4, y: 0});
  
var gauss_curve = svg
  .append('g')
  .append("path")
    .datum(gauss_data)
    .attr("fill", "#65AD69")
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
    
var xn_data = [{x: -3, y: 0}];
for (var i = -3; i < 3; i += 0.01) {
    xn_data.push({x: i, y: Math.exp(-0.5 * ((i - mu) / sigma * Math.sqrt(n)) ** 2) / (sigma * Math.sqrt(2 * Math.PI / n)) });
}
xn_data.push({x: 3, y: 0});
  
var xn_curve = svg
  .append('g')
  .append("path")
    .datum(xn_data)
    .attr("fill", "#E86456")
    .attr("border", 0)
    .attr("opacity", ".8")
    .attr("stroke", "#000")
    .attr("stroke-width", 1)
    .attr("stroke-linejoin", "round")
    .attr("d",  d3.line()
      .curve(d3.curveBasis)
        .x(function(d) { return x(d.x); })
        .y(function(d) { return y(-d.y - 0.5); })
);

var std_curve;

d3.csv("../../../../assets/chi-t.csv", function(error, chi_data) {
  if (error) throw error;
  
  std_curve = svg
    .append('g')
    .append("path")
      .datum(chi_data)
      .attr("fill", "#EDA137")
      .attr("border", 0)
      .attr("opacity", ".8")
      .attr("stroke", "#000")
      .attr("stroke-width", 1)
      .attr("stroke-linejoin", "round")
      .attr("d",  d3.line()
        .curve(d3.curveBasis)
          .x(function(d) { return x(-(n - 1) * d["chi_" + (n-1)] - 4.5); })
          .y(function(d) { return y(Math.min(4, d.chi_x / n)); })
   );
}

);

var labels_x = 500;

plt_label_path(svg, "#65AD69", labels_x, 0);

var span_sample = d3.select("#biased_viz")
  .append("span")
  .text("\\(f_X(x) \\sim \\mathcal{N}(0, 1)\\)")
  .style('color', '#65AD69')
  .style("font-size", "13px")
  .style("font-weight", "700")
  .attr("font-family", "Arvo")
  .attr("font-weight", 700)
  .style("position", "absolute")
  .style("left", labels_x + 100 + "px")
  .style("top", 10 + "px");

plt_label_path(svg, "#E86456", labels_x, 25);
       
var span_mean = d3.select("#biased_viz")
  .append("span")
  .text("\\( f_{\\overline{X}_n}(x) \\sim \\mathcal{N}(0, \\frac{1}{n}) \\)")
  .style('color', '#E86456')
  .style("font-size", "13px")
  .style("font-weight", "700")
  .attr("font-family", "Arvo")
  .attr("font-weight", 700)
  .style("position", "absolute")
  .style("left", labels_x + 100 + "px")
  .style("top", 35 + "px");
  
plt_label_path(svg, "#EDA137", labels_x, 50);
  
var span_std = d3.select("#biased_viz")
  .append("span")
  .text("\\( f_{\\hat{s}_n^2(X)}(x) \\sim \\frac{1}{n} \\chi_{n-1}^2 \\)")
  .style('color', '#EDA137')
  .style("font-size", "13px")
  .style("font-weight", "700")
  .attr("font-family", "Arvo")
  .attr("font-weight", 700)
  .style("position", "absolute")
  .style("left", labels_x + 100 + "px")
  .style("top", 60 + "px");
  
var xn_avg_curve = svg
    .append('g')
    .append("path")
      .datum([{x: 0, y: 0}, {x: 0, y: -2}, {x: -0.5, y: -2}])
      .attr("fill", "none")
      .attr("border", 0)
      .attr("opacity", ".8")
      .attr("stroke", "#000")
      .attr("stroke-width", 1)
      .attr("stroke-linejoin", "round")
      .attr("stroke-dasharray", "3 3")
      .attr("d",  d3.line()
          .x(function(d) { return x(d.x); })
          .y(function(d) { return y(d.y); })
   );

var span_mean_avg = d3.select("#biased_viz")
  .append("span")
  .text("\\( \\mathbb{E}[\\overline{X}_n] \\)")
  .style('color', '#696969')
  .style("font-size", "13px")
  .style("font-weight", "700")
  .attr("font-family", "Arvo")
  .attr("font-weight", 700)
  .style("position", "absolute")
  .style("left", x(-0.05) + "px")
  .style("top", y(-2.15) + "px");
   
var sn_avg_curve = svg
    .append('g')
    .append("path")
      .datum([{x: 1 - 1/n, y: -4}, {x: 1 - 1 / n, y: -6}, {x: 1.5 - 1/n, y: -6}])
      .attr("fill", "none")
      .attr("border", 0)
      .attr("opacity", ".8")
      .attr("stroke", "#000")
      .attr("stroke-width", 1)
      .attr("stroke-linejoin", "round")
      .attr("stroke-dasharray", "3 3")
      .attr("d",  d3.line()
          .x(function(d) { return x(d.y); })
          .y(function(d) { return y(d.x); })
   );

var span_mean_std = d3.select("#biased_viz")
  .append("span")
  .text("\\( \\mathbb{E}[\\hat{s}_n^2(X)] \\)")
  .style('color', '#696969')
  .style("font-size", "13px")
  .style("font-weight", "700")
  .attr("font-family", "Arvo")
  .attr("font-weight", 700)
  .style("position", "absolute")
  .style("left", x(-5.3) + "px")
  .style("top", y(1.6 - 1 / n) + "px");
  
function sample() {
  random_samples = [];
  smpl_dots = [];
  smpl_copy_dots = [];
  var average = 0;
  
  var sq_curve = svg
    .append('g')
    .append("path");
    
  for (var i = 0; i < n; i += 1) {
    random_samples.push(mu + sigma * randn_bm());
    
    smpl_dots.push(svg.append('g')
      .selectAll("dot")
      .data([{x: random_samples[i], y: 5}])
      .enter()
      .append("circle")
        .attr("cx", function (d) { return x(d.x); } )
        .attr("cy", function (d) { return y(d.y); } )
        .attr("r", 3)
        .style("fill", "#65AD69")
        .attr("stroke", "#000")
        .attr("stroke-width", 1));
        
    smpl_copy_dots.push(svg.append('g')
      .selectAll("dot")
      .data([{x: random_samples[i], y: 0}])
      .enter()
      .append("circle")
        .attr("cx", function (d) { return x(d.x); } )
        .attr("cy", function (d) { return y(d.y); } )
        .attr("r", 0)
        .style("fill", "#65AD69")
        .attr("stroke", "#000")
        .attr("stroke-width", 1));
        
     smpl_dots[i].transition()
           .duration(avg_dur)
           .attr("cx", function (d) { return x(random_samples[i]); } )
           .attr("cy", function (d) { return y(0); } );
     
     average += random_samples[i];
   }
   average /= n;
   
   for (var i = 0; i < n; i += 1) {
     smpl_copy_dots[i]
           .transition()
           .delay(avg_dur) 
           .duration(0)
           .attr("r", 3);
     
     smpl_copy_dots[i]
           .transition()
           .delay(avg_dur) 
           .duration(avg_dur)
           .style("fill", "#E86456")
           .attr("cx", function (d) { return x(average); } )
           .attr("cy", function (d) { return y(0); } );
     
     smpl_dots[i]
           .transition()
           .delay(3 * avg_dur) 
           .duration(avg_dur)
           .attr("cx", function (d) { return x(random_samples[i]); } )
           .attr("cy", function (d) { return y( (random_samples[i] - average) ** 2); } );
   }
   
   var xn_dot = svg.append('g')
      .selectAll("dot")
      .data([{x: average, y: 0}])
      .enter()
      .append("circle")
        .attr("cx", function (d) { return x(d.x); } )
        .attr("cy", function (d) { return y(d.y); } )
        .attr("r", 0)
        .style("fill", "#E86456")
        .attr("stroke", "#000")
        .attr("stroke-width", 1);
   
   xn_dot.transition().delay(2 * avg_dur).attr("r", 3);
      
   var sq_data = [];
   for (var i = -4; i <= 4; i += 0.1) {
       sq_data.push({x: i, y: (i - average) ** 2 });
   }
   
   sq_curve
      .datum(sq_data)
      .attr("fill", "none")
      .attr("border", 0)
      .attr("opacity", ".9")
      .attr("stroke", "black")
      .attr("stroke-width", 1)
      .attr("stroke-linejoin", "round")
      .attr("d",  d3.line()
        .curve(d3.curveBasis)
          .x(function(d) { return x(d.x); })
          .y(function(d) { return y(d.y); })
      );
   
   var average_y = 0;
   for (var i = 0; i < n; i += 1) {
       average_y += (random_samples[i] - average) ** 2;
   }
   average_y /= n;
   
   for (var i = 0; i < n; i += 1) {
       smpl_dots[i]
           .transition()
           .delay(4 * avg_dur) 
           .duration(avg_dur)
           .attr("cx", function (d) { return x(-4); } )
           .attr("cy", function (d) { return y( (random_samples[i] - average) ** 2); } );
           
       smpl_dots[i]
           .transition()
           .delay(5 * avg_dur) 
           .duration(avg_dur)
           .style("fill", "#EDA137")
           .attr("cx", function (d) { return x(-4); } )
           .attr("cy", function (d) { return y(average_y);  } );
   }
      
   var totalLength = sq_curve.node().getTotalLength();
   sq_curve.attr("stroke-dasharray", totalLength + " " + totalLength)
           .attr("stroke-dashoffset", totalLength)
           .transition().duration(4 * avg_dur)
           .attr("stroke-dashoffset", 0);
    
   sq_curve.transition().delay(6 * avg_dur).remove();
   
   var sn_dot = svg.append('g')
      .selectAll("dot")
      .data([{x: -4, y: average_y}])
      .enter()
      .append("circle")
        .attr("cx", function (d) { return x(d.x); } )
        .attr("cy", function (d) { return y(d.y); } )
        .attr("r", 0)
        .style("fill", "#EDA137")
        .attr("stroke", "#000")
        .attr("stroke-width", 1);
   
   sn_dot.transition().delay(6 * avg_dur).attr("r", 3);
   
   for (var i = 0; i < n; i += 1) {
      smpl_dots[i].transition().delay(6 * avg_dur).remove();
      smpl_copy_dots[i].transition().delay(2 * avg_dur).remove();
   }
   
   xn_dots.push(xn_dot);
   sn_dots.push(sn_dot);
}

function updateNGauss(new_n) { 
   n = new_n;
   reset();
   
   xn_data = [{x: -3, y: 0}];
   for (var i = -3; i < 3; i += 0.01) {
       xn_data.push({x: i, y: Math.exp(-0.5 * ((i - mu) / sigma * Math.sqrt(n)) ** 2) / (sigma * Math.sqrt(2 * Math.PI / n)) });
    }
   xn_data.push({x: 3, y: 0});

   xn_curve
      .datum(xn_data)
      .transition()
      .duration(1000)
      .attr("d",  d3.line()
        .curve(d3.curveBasis)
          .x(function(d) { return x(d.x); })
          .y(function(d) { return y(-d.y - 0.5); })
   );

   d3.csv("../../../../assets/chi-t.csv", function(error, chi_data) {
     if (error) throw error;
  
     std_curve
        .datum(chi_data)
        .transition()
        .duration(1000)
        .attr("d",  d3.line()
          .curve(d3.curveBasis)
            .x(function(d) { return x(-(n - 1) * d["chi_" + (n-1)] - 4.5); })
            .y(function(d) { return y(Math.min(4, d.chi_x / n)); })
         );
   });
   
   sn_avg_curve
     .datum([{x: 1 - 1/n, y: -4}, {x: 1-1/n, y: -6}, {x: 1.5-1/n, y: -6}])
     .transition()
     .duration(1000)
     .attr("d",  d3.line()
        .x(function(d) { return x(d.y); })
        .y(function(d) { return y(d.x); })
   );

     
    span_mean_std
      .transition()
      .duration(1000)
      .style("left", x(-5.3) + "px")
      .style("top", y(1.6 - 1 / n) + "px");
     
}

var sampleButton = d3.select("#sample-button-2");

sampleButton
    .on("click", function() {
      sample();
});

function reset() {
   for (var i = 0; i < xn_dots.length; i += 1) {
       xn_dots[i].remove();
       sn_dots[i].remove();
   }
   xn_dots = [];
   sn_dots = [];
}

var resetButton = d3.select("#reset-button");

resetButton
    .on("click", function() {
      reset();
});

var ng_x = d3.scaleLinear()
    .domain([2, 13])
    .range([0, width / 2])
    .clamp(true);
    
function roundN(x) { return Math.round(x - 0.5); }

function createSlider(svg_, parameter_update, slider_x, loc_x, loc_y, letter, color, init_val, round_fun) {
    var slider = svg_.append("g")
      .attr("class", "slider")
      .attr("transform", "translate(" + loc_x + "," + loc_y + ")");
    
    var drag = d3.drag()
	        .on("start.interrupt", function() { slider.interrupt(); })
	        .on("start drag", function() { 
	          handle.attr("cx", slider_x(round_fun(slider_x.invert(d3.event.x))));  
	          parameter_update(round_fun(slider_x.invert(d3.event.x)));
	         });
	         
    slider.append("line")
	    .attr("class", "track")
	    .attr("x1", slider_x.range()[0])
	    .attr("x2", slider_x.range()[1])
	  .select(function() { return this.parentNode.appendChild(this.cloneNode(true)); })
	    .attr("class", "track-inset")
	  .select(function() { return this.parentNode.appendChild(this.cloneNode(true)); })
	    .attr("class", "track-overlay")
	    .call(drag);

	slider.insert("g", ".track-overlay")
    .attr("class", "ticks")
    .attr("transform", "translate(0," + 18 + ")")
  .selectAll("text")
  .data(slider_x.ticks(6))
  .enter().append("text")
    .attr("x", slider_x)
    .attr("text-anchor", "middle")
    .attr("font-family", "Arvo")
    .text(function(d) { return d; });

   var handle = slider.insert("circle", ".track-overlay")
      .attr("class", "handle")
      .attr("r", 6).attr("cx", slider_x(init_val));
      
	svg_
	  .append("text")
	  .attr("text-anchor", "middle")
	  .attr("y", loc_y + 3)
	  .attr("x", loc_x - 21)
	  .attr("font-family", "Arvo")
	  .attr("font-size", 17)
	  .text(letter)
	  .style("fill", color);
	  	  
	return handle;
}

createSlider(svg, updateNGauss, ng_x, 190, 350, "n", "#696969", 6, roundN);

}

biasedness();

</script>

![](.)
*Fig. 3. Statistical experiments in estimating $\sigma$ for $X_1, \dots, X_n$ i.i.d. $\sim \mathcal{N}(0, 1)$. We see here that while $\overline{X}_n$ varies around $\mu=0$, expected value of estimator $\hat{s}_n^2(X)$ is lower than $\sigma^2 = 1$.*

So far we figured the unbiasedness of $g(X) = \overline{X}_n$. But how can we tell if $\overline{X}_n$ is an UMVU estimator? Can we find an estimator of $\mu$ with variance lower than $\frac{\sigma^2}{n}$?

### Efficient estimator

Given a set of unbiased estimators, it is not an easy task to determine which one provides the smallest variance. Luckily, we have a theorem which gives us a lower bound for an estimator variance.

Suppose we have a family of densities $f(\cdot, \vartheta)$, such that set $M_f=\lbrace x \in \mathcal{X} \mid f(x, \vartheta) > 0 \rbrace$ doesn't depend on $\vartheta$ and derivative $\frac{\partial}{\partial \vartheta} \log f(x, \vartheta)$ exists $\forall x \in \mathcal{X}$. Let's define function

$$ U_\vartheta(x) = \left\{\begin{array}{ll}
	\frac{\partial}{\partial \vartheta} \log f(x, \vartheta), & \text{if } x \in M_f, \\
	0, & \text{otherwise,}
	\end{array} \right. $$

and function

$$ \mathcal{I}(f(\cdot, \vartheta))=\mathbb{E} \big[\big(\frac{\partial}{\partial \vartheta} \log f(X, \vartheta)\big)^2\big]. $$

Under mild regularity conditions we have

$$ \mathbb{E}[U_\vartheta(X)] = \mathbb{E}\big[\frac{\partial}{\partial \vartheta} \log f(x, \vartheta)\big] = \frac{\partial}{\partial \vartheta}  \mathbb{E}[\log f(x, \vartheta)] = 0$$

and 

$$ \operatorname{Var}(U_\vartheta(X)) = \mathbb{E}[(U_\vartheta(X))^2]=\mathcal{I}(f(\cdot, \vartheta)). $$ 

Then using Cauchy-Schwartz inequality we get 

$$ \begin{aligned}
	\big( \frac{\partial}{\partial \vartheta} \mathbb{E}[g(X)] \big)^2 &= \big( \mathbb{E}[g(X) \cdot U_\vartheta(X)] \big)^2 \\ 
& = \big(\operatorname{Cov}(g(X), U_\vartheta(X)) \big)^2 \\
& \leq \operatorname{Var}(g(X))\cdot \operatorname{Var}(U_\vartheta(X)) \\ 
&= \mathcal{I}(f(\cdot, \vartheta))\cdot \operatorname{Var}(g(X)).
	\end{aligned} $$
	
The resulting inequality:

$$ \operatorname{Var}(g(X)) \geq \frac{\big(\frac{\partial}{\partial \vartheta} \mathbb{E}[g(X)]\big)^2}{\mathcal{I}(f(\cdot, \vartheta))} \quad \forall \vartheta \in \Theta $$

gives us **Cramér–Rao bound**. Function $\mathcal{I}(f(\cdot, \vartheta))$ is called **Fisher information** for family $\mathcal{P} = \lbrace P_\vartheta \mid \vartheta \in \Theta \rbrace$. If an unbiased estimator $g$ satisfies the upper equation with equality, then it is called **efficient**.

This theorem gives a lower bound for the variance of an estimator for $\gamma(\vartheta) = \mathbb{E}[g(X)]$ and can be used in principle to obtain UMVU estimators. Whenever the regularity conditions (e.g. invariance of $M_f$) are satisfied for all $g \in \mathcal{E}_\gamma$, then any efficient and unbiased estimator is UMVU.

Also, for a set of i.i.d. variables $X_1, \dots X_n$, meaning that their joint density distribution is 

$$f(x,\vartheta) = \prod_{i=1}^n f^i(x,\vartheta),$$

we have

$$ \mathcal{I}(f(\cdot, \vartheta))=n\mathcal{I}(f^1(\cdot, \vartheta)). $$

Let's get back to the example with $X_1, \dots, X_n$ i.i.d. $\sim \mathcal{N}(\mu, 1)$ having the density

$$ f^1(x, \vartheta) = \frac{1}{\sqrt{2\pi}} e^{-\frac{(x-\mu)^2}{2}}. $$

Then 

$$ \mathcal{I}(f^1(\cdot, \mu)) = \mathbb{E} \Big[ \big( \frac{\partial}{\partial \mu} \log f^1 (X_1, \mu)\big)^2 \Big] = \mathbb{E}[(X_1 - \mu)^2] = 1.$$

In particular, for $X = (X_1, \dots, X_n)$ Fisher information $\mathcal{I}(f(X, \mu)) = n$ and Cramér–Rao bound for unbiased estimator:

$$ \operatorname{Var}(g(X)) \geq \frac{1}{n} \big( \frac{\partial}{\partial \mu} \mathbb{E}[g(X)] \big)^2 = \frac{1}{n}. $$

Therefore, $g(x) = \overline{x}_n$ is an UMVU estimator.


### Multidimensional Cramér–Rao inequality

Define function 

$$ G(\vartheta)=\Big( \frac{\partial}{\partial \vartheta_j} \mathbb{E}[g_i(X)] \Big)_{i,j} \in \mathbb{R}^{k \times d}. $$

Then with multidimensional Cauchy-Shwartz inequality one can prove that under similar regularity conditions we have:

$$ \operatorname{Cov}(g(X)) \geq G(\vartheta) \mathcal{I}^{-1}(f(\cdot, \vartheta))G^T(\vartheta) \in \mathbb{R}^{k \times k}, $$

where

$$ \mathcal{I}(f(\cdot, \vartheta))=\Big( \mathbb{E}\Big[\frac{\partial}{\partial \vartheta_i} \log f(X, \vartheta) \cdot \frac{\partial}{\partial \vartheta_j} \log f(X, \vartheta) \Big]  \Big)_{i,j=1}^d \in \mathbb{R}^{d \times d}. $$

For an example with $X_1, \dots X_n$ i.i.d. $\sim \mathcal{N}(\mu, \sigma^2)$ with density

$$ f^1(x,\vartheta)=\frac{1}{\sqrt{2\pi \sigma^2}} \exp \Big(-\frac{(x-\mu)^2}{2\sigma^2}\Big) $$

we have

$$ U_\vartheta = \Big(\frac{\partial}{\partial \mu} \log f^1(X_1,\vartheta), \frac{\partial}{\partial \sigma^2} \log f^1(X_1,\vartheta)\Big)^T = \begin{pmatrix}
	(X_1-\mu)/\sigma^2 \\
	-\frac{1}{2\sigma^2}+\frac{1}{\sigma^4}(X_1-\mu)^2
	\end{pmatrix}. $$ 
	
Fisher information then

$$ \mathcal{I}(f^1(\cdot, \vartheta))=\mathbb{E}[U_\vartheta U_\vartheta^T]=
	\begin{pmatrix}
	\sigma^{-2} & 0 \\
	0 & \frac{1}{2}\sigma^{-4}
	\end{pmatrix}
	= \frac{1}{n}\mathcal{I}(f(\cdot, \vartheta)). $$
	
If $g(X)$ is an unbiased estimator, then $G(\vartheta)$ is identity matrix and Cramér–Rao bound then

$$ \begin{aligned}
\operatorname{Cov}_\vartheta(g(X)) & \geq G(\vartheta) \  \mathcal{I}^{-1} (f(\cdot, \vartheta)) \   G^T(\vartheta) \\ &= \mathcal{I}^{-1}(f(\cdot, \vartheta)) =
	 \begin{pmatrix}
	 \frac{\sigma^{2}}{n} & 0 \\
	 0 & \frac{2\sigma^{4}}{n}
	 \end{pmatrix}. 
	\end{aligned}$$

In particular for an unbiased estimator 

$$ \widetilde{g}(X)=\Big(\overline{X}_n, \frac{1}{n-1} \sum_{i=1}^n(X_j-\overline{X}_n)^2 \Big)^T $$

the following inequality holds

$$ \operatorname{Cov}_\vartheta(\widetilde{g}(X)) = 
	 \begin{pmatrix}
	 \frac{\sigma^{2}}{n} & 0 \\
      0 & \frac{2\sigma^{4}}{n-1}
	 \end{pmatrix} \geq \mathcal{I}(f(\cdot, \vartheta)), $$
	 
therefore $\widetilde{g}$ is not efficient. 

### Exponential family

In the previous examples, we consider without proof the fulfillment of all regularity conditions of the Cramér–Rao inequality. Next, we will discuss a family of distributions for which the Cramér–Rao inequality turns into an equality.

Proposition: let $P_\vartheta$ be distribution with density

$$ f(x, \vartheta) = c(\vartheta) h(x) \exp(\vartheta T(x)) \quad \forall \vartheta \in \Theta.$$

Then equality in Cramér–Rao theorem holds for $g(x) = T(x)$.

<details>
<summary>Proof</summary>
First let us note that $\int_{\mathcal{X}}f(x)\mu(dx) = 1$ for all $\vartheta \in \Theta$, hence

$$ c(\vartheta)=\Big( \int_{\mathcal{X}} h(x)\exp (\vartheta T(x) ) dx \Big)^{-1} $$

and

$$ \begin{aligned}
	0 & = \frac{\partial}{\partial \vartheta} \int_{\mathcal{X}} c(\vartheta) h(x) \exp ( \vartheta T(x) ) dx \\
	& = \int_{\mathcal{X}} (c'(\vartheta)+c(\vartheta)T(x)) h(x) \exp ( \vartheta T(x) ) dx.
	\end{aligned} $$

Using these two equations we get

$$ \begin{aligned} 
\mathbb{E}[T(X)] & = c(\vartheta) \int_{\mathcal{X}} h(x) T(x) \exp ( \vartheta T(x)) dx \\
	 & = -c'(\vartheta) \int_{\mathcal{X}}h(x) \exp ( \vartheta T(x) ) dx \\
	 & = -\frac{c'(\vartheta)}{c(\vartheta)}=(-\log c(\vartheta))'.
	 \end{aligned} $$

Fisher information:

$$ \mathcal{I}(f(\cdot, \vartheta)) = \mathbb{E}\Big[\Big( \frac{\partial}{\partial \vartheta} \log f(X, \vartheta) \Big)^2\Big]=\mathbb{E}[(T(X)+(\log c(\vartheta))')^2]=\operatorname{Var}(T(X)). $$

Also

$$ \begin{aligned}
	 \frac{\partial}{\partial \vartheta} \mathbb{E}[T(X)] & =\int_{\mathcal{X}} c'(\vartheta) h(x) T(x) \exp ( \vartheta T(x) ) dx + \int_{\mathcal{X}} c(\vartheta) h(x) T^2(x) \exp ( \vartheta T(x) ) dx \\
	 & = \frac{c'(\vartheta)}{c(\vartheta)} \int_{\mathcal{X}} c(\vartheta) h(x) T(x) \exp ( \vartheta T(x) ) dx + \mathbb{E}[(T(X))^2] \\
	 & = \mathbb{E}[(T(X))^2] - (\mathbb{E}[T(X)])^2.
	 \end{aligned} $$
	 
Therefore, 

$$ \frac{\Big(\frac{\partial}{\partial\vartheta}\mathbb{E}[T(X)] \Big)^2}{\mathcal{I}(f(\cdot, \vartheta))}= \operatorname{Var}(T(X)). $$

</details>

Formally, family $\mathcal{P} = \lbrace P_\vartheta \mid \vartheta \in \Theta \rbrace $ is called an **exponential family** if there exist mappings $c, Q_1, \dots Q_k: \Theta \rightarrow \mathbb{R}$ and $h, T_1, \dots T_k: \mathcal{X} \rightarrow \mathbb{R}$ such that

$$ f(x,\vartheta) = c(\vartheta) h(x) \exp \Big( \sum_{j=1}^k Q_j(\vartheta) T_j(x) \Big).$$

$\mathcal{P}$ is called **$k$-parametric exponential family** if functions $1, Q_1, \dots Q_k$ and $1, T_1, \dots T_k$ are linearly independent. Then we have equality to Cramér–Rao bound for $g = (T_1, \dots T_k)^T$. 

Here are some examples:

* If $X \sim \operatorname{Bin}(n, \vartheta)$, then

  $$ \begin{aligned}
  f(x, \vartheta) &= \binom n x \vartheta^x (1-\vartheta)^{n-x} \\
  &= (1-\vartheta)^n \binom n x \exp \Big(x \log \frac{\vartheta}{1-\vartheta} \Big).
  \end{aligned} $$
  
  Here $c(\vartheta) = (1-\vartheta)^n$, $h(x) = \binom n x$, $T_1(x) = x$ and $Q_1(\vartheta) = \log \frac{\vartheta}{1-\vartheta}$.

* If $X \sim \mathcal{N}(\mu, \sigma^2)$, then $\vartheta = (\mu, \sigma^2)^T$ and

  $$ \begin{aligned}
  f(x, \vartheta) &= \frac{1}{\sqrt{2\pi\sigma^2}} \exp\Big( \frac{(x-\mu)^2}{2\sigma^2} \Big) \\
  &= \frac{1}{\sqrt{2\pi\sigma^2}} \exp \Big( -\frac{\mu^2}{2\sigma^2} \Big) \exp\Big( -\frac{x^2}{2\sigma^2} + \frac{\mu x}{\sigma^2} \Big),
  \end{aligned} $$
  
  where $c(\vartheta) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp \big( -\frac{\mu^2}{2\sigma^2} \big) $, $Q_1(\vartheta) = -\frac{1}{2\sigma^2}$, $Q_2(\vartheta) = \frac{\mu}{\sigma^2}$, $T_1(x)=x^2$ and $T_2(x)=x$.
  
* If $X \sim \operatorname{Poisson}(\lambda)$, then

 $$ f(x, \lambda) = \frac{\lambda^x e^{-\lambda}}{x!} = e^{-\lambda} \frac{1}{x!} \exp \big(x \log \lambda \big). $$
 
Denoting $Q(\vartheta) = (Q_1(\vartheta), \dots, Q_k(\vartheta))^T$ we get transformed parametric space $ \Theta^* =  Q(\Theta) $, which we call **natural parametric space**. In examples above

* $X \sim \operatorname{Bin}(n, \vartheta)$: $\Theta^* = \lbrace \log \frac{\vartheta}{1-\vartheta} \mid \vartheta \in (0, 1) \rbrace = \mathbb{R}$.
* $X \sim \mathcal{N}(\mu, \sigma^2)$: $\Theta^* = \big\lbrace \big( \frac{\mu}{\sigma^2}, -\frac{1}{\sigma^2} \big) \mid \mu \in \mathbb{R}, \sigma^2 \in \mathbb{R}^+ \big\rbrace = \mathbb{R} \times \mathbb{R}^-.$
* $X \sim \operatorname{Poisson}(\lambda)$: $\Theta^* = \lbrace \log \lambda \mid \lambda \in \mathbb{R}^+ \rbrace = \mathbb{R}$.

It must be noted that for an exponential family $\mathcal{P}$ estimator $T(X) = (T_1(X), \dots T_k(X))$ is UMVU for $\mathbb{E}[T(X)]$. For example, if $X_1, \dots X_n$ i.i.d. $\sim \mathcal{N}(\mu, \sigma^2)$ with joint density

$$ f(x,\vartheta) = c(\vartheta) \exp \Big( -\frac{n}{2\sigma^2}\Big( \frac{1}{n} \sum_{i=1}^n x_i^2 \Big) + \frac{n\mu}{\sigma^2}\Big( \frac{1}{n}x_i \Big) \Big),$$

then estimator 

$$ T(X) = \Big( \frac{1}{n} \sum_{i=1}^n X_i, \frac{1}{n} \sum_{i=1}^n X_i^2  \Big) $$

is efficient for $(\mu, \mu^2 + \sigma^2)^T$.

### Common estimation methods

If distribution doesn't belong to exponential family, then for such case there exist two classical estimation methods:

* **Method of moments**. Let $X_1, \dots X_n$ i.i.d. $\sim P_\vartheta$ and 

  $$ \gamma(\vartheta) = f(m_1, \dots, m_k), $$
  
  where $m_j = \mathbb{E}[X_1^j]$. Then **estimation by method of moments** will be
  
  $$ \hat{\gamma} (X) = f(\hat{m}_1, \dots, \hat{m}_k),$$
  
  where $m_j = \frac{1}{n}\sum_{i=1}^nX_i^j$. 
  
* **Maximum likelihood method**. Say $\gamma(\vartheta) = \vartheta \in \mathbb{R}^k$. Then $\hat{\vartheta}(x)$ is a **maximum likelihood estimator** if

  $$ f(x, \hat{\vartheta}) = \sup_{\vartheta \in \Theta} f(x, \vartheta).$$
  
Again in example $X_1, \dots X_n$ i.i.d. $\sim \mathcal {N}(\mu, \sigma^2)$ an estimator for $\vartheta = (\mu, \sigma^2)^T = (m_1, m_2 - m_1^2)^T$ by method of moments will be

$$ \hat{\gamma}(\vartheta)=(\hat{m}_1, \hat{m}_2-\hat{m}_1^2)^T=(\overline{x}_n, \hat{s}_n^2)^T. $$

I'm going to leave it as an exercise to prove that this estimator coincides with the estimation obtained by the maximum likelihood method.

Let's take another example, $X_1, \dots X_n$ i.i.d. $\sim \mathcal{U}(0, \vartheta)$, where estimated parameter $\vartheta > 0$. One can show that estimator

$$g_{ML}(X) = X_{(n)} = \max \lbrace X_1, \dots X_n \rbrace $$

is a maximum-likelihood estimator. On the other hand, 

$$g_{MM}(X) = 2 \overline{X}_n$$

is an estimator by method of moments. Also, maximum-likelihood estimator follows scaled Beta-distribution, $g_{ML}(X) \sim \vartheta B(n, 1)$, and therefore it is biased:

$$\mathbb{E}[g_{ML}(X)] = \vartheta\frac{n}{n+1}.$$

UMVU estimator is $g(X) = X_{(n)} (1 + \frac{1}{n})$, and its variance:

$$\operatorname{Var}[g(X)] = \vartheta^2\frac{1}{n(n+2)} < \frac{\vartheta^2}{n}$$

However, the Cramér-Rao lower bound is $\frac{\vartheta^2}{n}$. And this is another exercise to figure out why Cramér-Rao inequality here is not satisfied.
