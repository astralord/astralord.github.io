---
layout: post
title: 'Visual Guide to Statistics. Part III: Asymptotic Properties of Estimators'
date: 2022-04-12 03:13 +0800
categories: [Statistics]
tags: [statistics, consistent-estimator, central-limit-theorem, slutsky-lemma, delta-method, asymptotic-efficiency, maximum-likelihood-estimator]
math: true
published: true
---

> A minimal condition for a good estimator is that it is getting closer to estimated parameter with growing size of sample vector. In this post we will focus on asymptotic properties of estimators.

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

Now let $g_n$ be an estimator of $\gamma(\vartheta)$ with values in metric space. Assume that all experiments are defined on a joint probability space $P_\vartheta$ for all $n$. We say that

* $g_n$ is **(weakly) consistent** if

$$g_n \xrightarrow[]{\mathbb{P}}\gamma(\vartheta) \quad \forall \vartheta \in \Theta.$$

* $g_n$ is **strongly constistent** if

$$g_n \xrightarrow[]{\text{a.s.}} \gamma(\vartheta)  \quad \forall \vartheta \in \Theta.$$

Recall the method of moments from [Part I](https://astralord.github.io/posts/visual-guide-to-statistics-part-i-basics-of-point-estimation/#common-estimation-methods): $X_1, \dots, X_n$ i.i.d. $\sim P_\vartheta$, $\vartheta \in \Theta \subset \mathbb{R}^k$ and $\gamma: \Theta \rightarrow \Gamma \subset \mathbb{R}^l$. Also 

$$m_j = \mathbb{E}_\vartheta[X_1^j] = \int x^j P_\vartheta(dx)$$

for $j = 1, \dots, k$, and

$$\gamma(\vartheta) = f(m_1, \dots, m_k).$$

Then choose

$$\hat{\gamma}(X) = f(\hat{m}_1, \dots, \hat{m}_k),$$

where

$$\hat{m}_j = \frac{1}{n} \sum_{i=1}^{n}X_k^j.$$

By Law of Large Numbers $\hat{m}_j \rightarrow m_j$ a.s. Since $f$ is continuous, we obtain

$$\hat{\gamma}(X) \xrightarrow[]{\text{a.s.}} \gamma(\vartheta).$$

Hence, $\hat{\gamma}(X)$ is a strongly consistent estimator.

### Central Limit Theorem

Let $(X_n)$ be a sequence of $d$-dimensional random variables. [Lévy's continuity theorem](https://en.wikipedia.org/wiki/L%C3%A9vy%27s_continuity_theorem#:~:text=In%20probability%20theory%2C%20L%C3%A9vy's%20continuity,convergence%20of%20their%20characteristic%20functions.) states that

$$X_n \xrightarrow[]{\mathcal{L}} X \quad \Longleftrightarrow \quad \mathbb{E}[\exp(iu^TX_n)] \rightarrow \mathbb{E}[\exp(iu^TX)] \quad \forall u \in \mathbb{R}^d.$$

If we write $u=ty$ for $t \in \mathbb{R}$, $y \in \mathbb{R}^d$, then we can say that $X_n \xrightarrow[]{\mathcal{L}} X$ if and only if

$$y^TX_n \xrightarrow[]{\mathcal{L}} y^TX \quad \forall y \in \mathbb{R}^d.$$

This statement is called **Cramér–Wold theorem**.

If $X_1, \dots, X_n$ are i.i.d. with $\mathbb{E}[X_j]=\mu \in \mathbb{R}^d$ and $\operatorname{Cov}(X_j)=\Sigma \in \mathbb{R}^{d \times d}$ (positive-definite, $\Sigma > 0$), then for random vector

$$X^{(n)} = \frac{1}{n}\sum_{j=1}^n X_j \in \mathbb{R}^d$$

we know from one-dimensional Central Limit Theorem (CLT) that 

$$\sqrt{n}(y^TX^{(n)} -y^T\mu) \xrightarrow[]{\mathcal{L}} \mathcal{N}(0, y^T\Sigma y) \quad \forall y \in \mathbb{R}^d.$$

Applying Cramér–Wold theorem we get

$$\sqrt{n}(X^{(n)}-\mu) \xrightarrow[]{\mathcal{L}} \mathcal{N}(0, \Sigma).$$

This statement is known as **Multidimensional Central Limit Theorem**.

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
  width: 80px;
  height: 25px;
  font-family: Arvo;
  font-size: 11px;
}

#sample-button-2:hover {
  background-color: #696969;
}

#sample-button-3 {
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

#sample-button-3:hover {
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

#reset-button-2 {
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

#reset-button-2:hover {
  background-color: #696969;
}
   
</style>

<script src="https://d3js.org/d3.v4.min.js"></script>
<script src="https://d3js.org/d3-contour.v1.min.js"></script>
<link href="https://fonts.googleapis.com/css?family=Arvo" rel="stylesheet">

<button id="sample-button">Sample</button>
<button id="sample-button-2">Sample 100x</button>
<button id="reset-button">Reset</button>
<div id="mclt"></div> 

<script>
d3.select("#mclt")
  .style("position", "relative");

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

function gamma_rand(k) {
    x = 0;
    for (var i = 0; i < k; i += 1) {
        x -= Math.log(Math.random());
    }
    return x;
}

function dirichlet(ks) {
    xs = [];
    x0 = 0;
    for (var i = 0; i < ks.length; i += 1) {
        xs.push(gamma_rand(ks[i]));
        x0 += xs[i];
    }
    return [xs[0] / x0, xs[1] / x0];
}


function erf(x) {
    if (Math.abs(x) > 3) {
      return x / Math.abs(x);
    }
    var m = 1.00;
    var s = 1.00;
    var sum = x * 1.0;
    for(var i = 1; i < 50; i++){
        m *= i;
        s *= -1;
        sum += (s * Math.pow(x, 2.0 * i + 1.0)) / (m * (2.0 * i + 1.0));
    }  
    return 1.1283791671 * sum;
}

function Phi(x) {
    return 0.5 * (1 + erf(x / 1.41421356237));
}

function randn_bm() {
    var u = 0, v = 0;
    while(u === 0) u = Math.random();
    while(v === 0) v = Math.random();
    return Math.sqrt( -2.0 * Math.log( u ) ) * Math.cos( 2.0 * Math.PI * v );
}

function biv_uni(r) {
    var rho = 2 * Math.sin(r * Math.PI / 6);
    var z1 = randn_bm();
    var z2 = rho * z1 + Math.sqrt(1 - rho * rho) * randn_bm();
    return [erf(z1), erf(z2)];
}

function phi(x, mu, sigma) {
    var y = (x - mu) / sigma;
    y *= y;
    y = Math.exp(-y / 2);
    y /= (sigma * 1.41421356237 * Math.PI);
    return y;
}


function mclt() {
var n = 7,
    rho = 0;

const margin = {top: 20, right: 0, bottom: 5, left: 70},
    width = 750 - margin.left - margin.right,
    height = 400 - margin.top - margin.bottom,
    fig_height = 200,
    fig_width = 250,
    fig_margin = 100,
    fig_trans = fig_width + fig_margin;
    
const avg_dur = 1000;
    
var svg = d3.select("div#mclt")
  .append("svg")
    .attr("width", width + margin.left + margin.right)
    .attr("height", height + margin.top + margin.bottom)
  .append("g")
    .attr("transform",
          "translate(" + margin.left + "," + margin.top + ")");

var x = d3.scaleLinear()
          .range([0, fig_width])
          .domain([-1, 1]);
            
var xAxis = svg.append("g")
   .attr("transform", "translate(0," + fig_height + ")")
   .call(d3.axisBottom(x).ticks(4));
  
xAxis.selectAll(".tick text")
   .attr("font-family", "Arvo");
   
var xAvg = d3.scaleLinear()
          .range([fig_trans, 2 * fig_width + fig_margin])
          .domain([-3, 3]);

var xAvgAxis = svg.append("g")
   .attr("transform", "translate("+ 0 + "," + fig_height + ")")
   .call(d3.axisBottom(xAvg).ticks(5));
  
xAvgAxis.selectAll(".tick text")
   .attr("font-family", "Arvo");

var y = d3.scaleLinear()
          .range([fig_height, 0])
          .domain([-1, 1]);
            
var yAxis = svg.append("g")
    .call(d3.axisLeft(y).ticks(4));
  
yAxis.selectAll(".tick text")
    .attr("font-family", "Arvo");

var yAvg = d3.scaleLinear()
          .range([fig_height, 0])
          .domain([-3, 3]);
            
var yAvgAxis = svg.append("g")
   .attr("transform", "translate("+ fig_trans + ",0)")
    .call(d3.axisLeft(yAvg).ticks(5));
  
yAvgAxis.selectAll(".tick text")
    .attr("font-family", "Arvo");

const axs_mrgn = 0.25;
const uni_data = [{x: -1, y: -1.25}, {x: -1, y: -1.5}, {x: 1, y: -1.5}, {x: 1, y: -1.25}];
var uni_x_curve = svg
    .append('g')
    .append("path")
      .datum(uni_data)
      .attr("fill", "#65AD69")
      .attr("border", 0)
      .attr("opacity", ".8")
      .attr("stroke", "#000")
      .attr("stroke-width", 1)
      .attr("stroke-linejoin", "round")
      .attr("d",  d3.line()
          .x(function(d) { return x(d.x); })
          .y(function(d) { return y(d.y); })
      );
      
var uni_y_curve = svg
    .append('g')
    .append("path")
      .datum(uni_data)
      .attr("fill", "#65AD69")
      .attr("border", 0)
      .attr("opacity", ".8")
      .attr("stroke", "#000")
      .attr("stroke-width", 1)
      .attr("stroke-linejoin", "round")
      .attr("d",  d3.line()
          .x(function(d) { return x(d.y); })
          .y(function(d) { return y(d.x); })
      );
      
var gauss_data = [];
var mu = 0, sigma = 1 / Math.sqrt(3);
var scale = 3 * axs_mrgn * (sigma * 1.41421356237 * Math.PI);
for (var i = -3; i <= 3; i += 0.01) {
    gauss_data.push({x: i, y: -3.75 - scale * phi(i, mu, sigma)});
}
      
var gauss_x_curve = svg
    .append('g')
    .append("path")
      .datum(gauss_data)
      .attr("fill", "#E86456")
      .attr("border", 0)
      .attr("opacity", ".8")
      .attr("stroke", "#000")
      .attr("stroke-width", 1)
      .attr("stroke-linejoin", "round")
      .attr("d",  d3.line()
          .x(function(d) { return xAvg(d.x); })
          .y(function(d) { return yAvg(d.y); })
      );
      
var gauss_y_curve = svg
    .append('g')
    .append("path")
      .datum(gauss_data)
      .attr("fill", "#E86456")
      .attr("border", 0)
      .attr("opacity", ".8")
      .attr("stroke", "#000")
      .attr("stroke-width", 1)
      .attr("stroke-linejoin", "round")
      .attr("d",  d3.line()
          .x(function(d) { return xAvg(d.y); })
          .y(function(d) { return yAvg(d.x); })
      );


var avg_dots = [];
var gauss_density = [];
	      
function sampleUniform() {
    var uni_data = [], uni_dots = [];
    var avg_x = 0, avg_y = 0;
    var sqrt_n = Math.sqrt(n);
    for (var i = 0; i < n; i += 1) {
        var uni_point = biv_uni(rho);
        uni_data.push({x: uni_point[0], y: uni_point[1]});
        avg_x += uni_data[i].x;
        avg_y += uni_data[i].y;
    }
    avg_x /= n;
    avg_y /= n;
    
    for (var i = 0; i < n; i += 1) {
        
	    uni_dots.push(svg.append('g')
	      .selectAll("dot")
	      .data([uni_data[i]])
	      .enter()
	      .append("circle")
	        .attr("cx", function (d) { return x(d.x); } )
	        .attr("cy", function (d) { return y(d.y); } )
	        .attr("r", 0)
	        .style("fill", "#65AD69")
	        .attr("stroke", "#000")
	        .attr("stroke-width", 1));
	        
        uni_dots[i]
            .transition()
            .duration(avg_dur)
	         .attr("r", 3);
	        
        uni_dots[i]
            .transition()
            .duration(avg_dur)
            .delay(avg_dur)
            .style("fill", "#E86456")
            .attr("cx", function (d) { return x(avg_x); } )
            .attr("cy", function (d) { return y(avg_y); } );
            
        if (i > 0) {
            uni_dots[i].transition().delay(2 * avg_dur)
	        .attr("opacity", 0);
	        uni_dots[i].transition().delay(2 * avg_dur)
	        .remove();
        }
    }
    
    avg_dots.push(uni_dots[0]);

    avg_dots[avg_dots.length - 1]
        .transition()
        .duration(avg_dur)
        .delay(2 * avg_dur)
        .attr("cx", function (d) { return xAvg(sqrt_n * avg_x); } )
        .attr("cy", function (d) { return yAvg(sqrt_n * avg_y); } );

}

function reset() {
    for (var i = 0; i < avg_dots.length; i += 1) {
        avg_dots[i].remove();
    }
    avg_dots = [];
}

var sampleButton = d3.select("#sample-button")
    .on("click", function() {
    sampleUniform();
});

var sampleButton = d3.select("#sample-button-2")
    .on("click", function() {
    for (var i = 0; i < 100; i += 1) {
        sampleUniform();
    }
});

var resetButton = d3.select("#reset-button")
    .on("click", function() {
      reset();
});

var rho_x = d3.scaleLinear()
    .domain([-1, 1])
    .range([0, width / 4])
    .clamp(true);
    
function trivialRound(x) { return x; }

function updateRho(r) {
    rho = r;
    reset();
}

createSlider(svg, updateRho, rho_x, 50, 0.95 * height, "", "#65AD69", rho, trivialRound);

var n_x = d3.scaleLinear()
    .domain([1, 12])
    .range([0, width / 4])
    .clamp(true);
    
function roundN(x) { return Math.round(x - 0.5); }

function updateN(num) {
    n = num;
    reset();
}

createSlider(svg, updateN, n_x, 3 * fig_width / 2 + 20, 0.95 * height, "n", "#696969", n, roundN);

d3.select("#mclt")
  .append("div")
  .text("\\(\\rho \\)")
  .style('color', '#696969')
  .style("font-size", "17px")
  .style("font-weight", "700")
  .attr("font-family", "Arvo")
  .attr("font-weight", 700)
  .style("position", "absolute")
  .style("left", 95 + "px")
  .style("top", 0.95 * height + 5 + "px");
  
d3.select("#mclt")
  .append("div")
  .text("\\(X \\sim \\mathcal{U}(-1, 1), \\quad \\Sigma = \\frac{1}{\\sqrt{3}} \\begin{pmatrix} 1 & \\rho \\\\ \\rho & 1 \\end{pmatrix} \\)")
  .style('color', '#65AD69')
  .style("font-size", "13px")
  .style("font-weight", "700")
  .attr("font-family", "Arvo")
  .attr("font-weight", 700)
  .style("position", "absolute")
  .style("left", fig_width / 2 - 35 + "px")
  .style("top", 0.78 * height + "px");
  
d3.select("#mclt")
  .append("div")
  .text("\\(\\sqrt{n} (X^{(n)}-\\mu) \\sim \\mathcal{N}(0, \\Sigma) \\)")
  .style('color', '#E86456')
  .style("font-size", "13px")
  .style("font-weight", "700")
  .attr("font-family", "Arvo")
  .attr("font-weight", 700)
  .style("position", "absolute")
  .style("left", 3 * fig_width / 2 + 80 + "px")
  .style("top", 0.8 * height + "px");
  
}

mclt();

</script>

![](.)
*Fig. 1. Visualization of multidimensional CLT for two-dimensional case. On the left-hand side there is random vector of two uniformly distributed random variables: $X_1, X_2 \sim \mathcal{U}(-1, 1)$ with mean $\mu=(0, 0)^T$ and correlation $\rho$. On the right-hand side is $\sqrt{n} X^{(n)}$ which for large $n$ has approximately normal distribution with zero mean and the same covariance as $X$.*

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

where 

$$\Sigma = (\Sigma)_{i,j=1}^k = (m_{i+j} - m_i m_j)_{i,j=1}^k.$$

Then 

$$\sqrt{n}(\gamma(\hat{m}_1, \dots, \hat{m}_k) - \gamma(m_1, \dots, m_k)) \xrightarrow[]{\mathcal{L}} \mathcal{N}(0, D \Sigma D^T).$$

* Take another example: let $X_1, \dots X_n$ be i.i.d. with 

$$\mathbb{E}_\vartheta[X_i] = \mu \quad \text{and} \quad \operatorname{Var}_\vartheta(X_i) = \sigma^2.$$

From CLT we have

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

where 

$$SQ_{xy} = \frac{1}{n} \sum_{i=1}^{n}(X_i-\overline{X}_n)(Y_i - \overline{Y}_n),$$

$SQ_{xx}, SQ_{yy}$ - likewise, is called **the Pearson correlation coefficient**. Without loss of generality, assume $\mu_1=\mu_2=0$, $\sigma=\tau=1$, because $\hat{\rho}_n$ is invariant under affine transformation. 

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

### Asympotic efficiency

Let $g_n \subset \mathbb{R}^l$ be a sequence of estimators with 

$$\mu_n(\vartheta)=\mathbb{E}_\vartheta[g_n] \in \mathbb{R}^l \quad \text{and} \quad \Sigma_n(\vartheta)=\operatorname{Cov}(\vartheta) \in \mathbb{R}^{l \times l},$$

such that $\lVert \Sigma_n(\vartheta) \rVert \rightarrow 0$. Then

* $g_n$ is called **asymptotically unbiased** for $\gamma(\vartheta)$ if

$$ \mu_n(\vartheta) \rightarrow \gamma(\vartheta), $$

* $g_n$ is called **asymptotically normal** if

$$\Sigma_n^{-\frac{1}{2}}(\vartheta)(g_n-\mu_n(\vartheta)) \xrightarrow[]{\mathcal{L}} \mathcal{N}(0, \mathbb{I}_l),$$

where $\mathbb{I}_l$ is identity matrix.

Let $f_n: \mathcal{X} \rightarrow \mathbb{R}^l$ be asymptotically unbiased and asymptotically normal sequence of estimators. Under regularity conditions from [Cramér–Rao theorem](https://astralord.github.io/posts/visual-guide-to-statistics-part-i-basics-of-point-estimation/#efficient-estimator) we call $g_n$ **asymptotically efficient**, if

$$ \lim\limits_{n \rightarrow \infty} \Sigma_n(\vartheta) \mathcal{I}(f_n(\cdot, \vartheta))=\mathbb{I}_l \quad \forall \vartheta \in \Theta,  $$

where $\mathcal{I}(f_n(\cdot, \vartheta))$ is Fisher information.

The intuition behind definition above is the following: if $g_n$ is unbiased, then by Cramér–Rao theorem $\operatorname{Cov}_\vartheta(g_n) \geq \mathcal{I}^{-1}(f_n(\cdot, \vartheta))$. Due to asymptotic normality:

$$\Sigma_n^{-\frac{1}{2}}(\vartheta)(g_n-\mu_n(\vartheta)) \xrightarrow[]{\mathcal{L}} \mathcal{N}(0, \mathbb{I}_l)$$

we have approximately

$$\operatorname{Cov}_\vartheta(g_n) \approx \Sigma_n(\vartheta) \approx \mathcal{I}^{-1}(f_n(\cdot, \vartheta))$$

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

$$\mathcal{I}^{-1}(f_n(\cdot, \vartheta)) = \begin{pmatrix}
	\sigma^2/n & 0 \\
	0 & 2\sigma^4 / n
	\end{pmatrix} $$
	
and $g_n$ is not efficient, but asymptotically efficient.

### Asymptotic properties of maximum-likelihood estimators

In [Part I](https://astralord.github.io/posts/visual-guide-to-statistics-part-i-basics-of-point-estimation/#common-estimation-methods) we briefly mentioned maximum-likelihood estimators as one of the most common estimation methods in statistic. It is worth knowing what their asymptotic properties are. Let's rewrite the definition here: let $X_1, \dots X_n$ be i.i.d. $\sim P_\vartheta$, $\vartheta \in \Theta$ with densities $f(\cdot, \vartheta)$. We call

$$\ell(\cdot, \vartheta) = \log f(\cdot, \vartheta) $$

**the log-likelihood function** and set 

$$\begin{aligned}\hat{\theta}_n(X) &= \arg \sup_{\vartheta \in \Theta} f(X, \vartheta) \\&= \arg \sup_{\vartheta \in \Theta} \ell (X, \vartheta) \\&= \arg \sup_{\vartheta \in \Theta} \frac{1}{n} \sum_{i=1}^{n} \ell (X_i, \vartheta) \end{aligned}$$

as **the maximum-likelihood estimator** for $\vartheta$.

Now, say the following conditions are satisfied:

1. $\Theta \subset \mathbb{R}^k$ is compact space
2. $L(\eta, \vartheta) = \mathbb{E}[\ell(X_i, \eta)]$ and $L_n(\eta) = \frac{1}{n}\sum_{i=1}^n\ell(X_i, \eta)$ are a.s. continuous functions over $\eta$.
3. $$\sup_{\eta \in \Theta} | L_n(\eta)-L(\eta, \vartheta)|\xrightarrow{\mathcal{L}}0.$$

Then ml-estimator $\hat{\theta}_n$ is consistent.

Proof: for any $\eta \in \Theta$:

$$L(\eta, \vartheta) = \int \ell(x, \eta) f(x,\vartheta) dx = \int \ell(x,\vartheta)f(x,\vartheta)dx - KL(\vartheta | \eta),  $$

where $KL$ is **Kullback-Leibler divergence**:

$$ KL(\vartheta | \eta) = \int_{\mathcal{X}} \log\Big(\frac{f(x,\vartheta)}{f(x,\eta)}\Big) f(x,\vartheta)dx.
$$

It can be shown that 

$$\begin{aligned} 
KL(\vartheta | \eta) & = \int_{\mathcal{X}} -\log\Big(\frac{f(x,\eta)}{f(x,\vartheta)}\Big) f(x,\vartheta)dx \\ \text{Jensen inequality} \rightarrow & \geq -\log\int_{\mathcal{X}} \frac{f(x,\eta)}{f(x,\vartheta)} f(x,\vartheta)dx
\\ & = 0.
\end{aligned}
$$

and $KL(\vartheta | \eta) =0$ only when $f(x,\vartheta) = f(x,\eta)$ for almost every $x$. Therefore we conclude that $L(\eta, \vartheta)$ reaches maximum at $\eta = \vartheta$.

Using the fact that function $m_f = \arg\max_{\eta \in \Theta} f(\eta)$ is continuous if $m_f$ is unique, we finish the proof from

$$ \vartheta = \arg \max L(\eta, \vartheta)\quad \text{and} \quad \hat{\theta}_n=\arg \max L_n(\eta) $$

and condition 3. 

### Asymptotic efficiency of maximum-likelihood estimators

If the following conditions are satisfied:

* $\Theta \subset \mathbb{R}^k$ is compact and $\vartheta \subset \operatorname{int}(\Theta)$.
* $\ell(x, \eta)$ is continuous $\forall \eta \in \Theta$ and twice continuously differentiable over $\vartheta$ for almost every $x \in \mathcal{X}$.
* There exist functions $H_0, H_2 \in L^1(P_\vartheta)$ and $H_1 \in L^2(P_\vartheta)$, such that:

$$\sup_{\eta \in \Theta} \|\ell(x, \eta)\| \leq H_0(x), \quad \sup_{\eta \in \Theta} \|\dot{\ell}(x, \eta)\| \leq H_1(x), \quad \sup_{\eta \in \Theta} \|\ddot{\ell}(x, \eta)\| \leq H_2(x) \quad \forall x \in \mathcal{X}. $$

* Fisher information

$$ \mathcal{I}(f(\cdot, \vartheta))=\mathbb{E}_\vartheta[\dot{\ell}(X,\vartheta)\dot{\ell}(X,\vartheta)^T] $$

   is positive definite (and therefore invertible),

then $\hat{\theta}_n$ is asymptotically normal:

$$\sqrt{n}(\hat{\theta}_n-\vartheta) \xrightarrow[]{\mathcal{L}} \mathcal{N}(0, \mathcal{I}(f(\cdot, \vartheta))^{-1}).$$

We will prove it in 4 steps:

<span style="color:salmon">*Step 1.*</span> Prove the constistency of $\hat{\theta}_n$. For this we need to verify that all conditions from theorem about consistency of ml-estimator are satisfied:

1. Satisfied by the assumption. 
2. $L_n(\eta)$ is a.s. continuous. Using 2-3 conditions and dominated convergence we get
$$ |L(\eta_1, \vartheta) - L(\eta_2, \vartheta)| \leq \int_{\mathcal{X}} |\ell(x, \eta_1) - \ell(x,\eta_2)| f(x,\vartheta) \mu(dx) \rightarrow 0,  $$
for  $\eta_1 \rightarrow \eta_2$.
3. By Law of Large Numbers:
$$ \begin{aligned}
		\limsup_{n \rightarrow \infty} \sup_{\| \eta_1 - \eta_2 \| < \delta} | L_n(\eta_1) - L_n(\eta_2)| & \leq \limsup_{n \rightarrow \infty} \frac{1}{n} \sum_{i=1}^{n} \sup_{\| \eta_1 - \eta_2 \| < \delta} |\ell(X_i, \eta_1) - \ell(X_i, \eta_2) |\\
		& = \mathbb{E}_\vartheta[\sup_{\| \eta_1 - \eta_2 \| < \delta}|\ell(X,\eta_1) - \ell(X, \eta_2)|]
				\end{aligned}
$$

Because $\Theta$ is compact, function $\ell(X, \eta)$ is a.s. uniformly continuous in $\eta$. As a consequence, the last statement converges to zero for $\delta \rightarrow 0$ (using again dominated convergence).
	
<span style="color:salmon">*Step 2.*</span> Let 

$$\dot{L}_n(\vartheta) := \frac{1}{n} \sum_{i=1}^{n} \dot{\ell}(X_i, \vartheta).$$

Prove that $\sqrt{n}\dot{L}_n(\vartheta) \xrightarrow[]{\mathcal{L}} \mathcal{N}(0, \mathcal{I}(f(\cdot, \vartheta))).$ 

Let $A_n$ be $k$-dimensional rectangle with vertices in $\hat{\theta}_n$ and $\vartheta$. Because $\hat{\theta}_n \xrightarrow{\mathcal{L}} \vartheta$ and $\vartheta \in \operatorname{int}(\Theta)$, we have 

$$P_\vartheta(A_n \subset \operatorname{int}(\Theta)) \rightarrow 1.$$

Also 

$$ \dot{L}_n(\hat{\theta}_n) = \frac{1}{n} \sum_{i=1}^{n} \dot{\ell}(X_i, \hat{\theta}_n) = 0 $$

by definition of $\hat{\theta}_n$, and 

$$
	\begin{aligned}
	 \mathbb{E}[\dot{\ell}(X_i, \vartheta)] & = \int_{\mathcal{X}} \dot{\ell}(x, \vartheta) f(x, \vartheta) dx \\
	  & = \int_{\mathcal{X}} \dot{f}(x, \vartheta) dx \\ & =\frac{\partial}{\partial \vartheta}\int_{\mathcal{X}} f(x, \vartheta) dx = 0.
	\end{aligned}
	 $$
	 
By definition

$$ \operatorname{Cov}(\dot{\ell}(X_i, \vartheta)) = \mathcal{I}(f(\cdot, \vartheta)). $$

Then by CLT:

$$ \sqrt{n} \dot{L}_n(\vartheta) \xrightarrow[]{\mathcal{L}} \mathcal{N}(0, \mathcal{I}(f(\cdot, \vartheta))). $$
	 

<span style="color:salmon">*Step 3.*</span> By Mean Theorem:

$$ -\dot{L}_n(\vartheta) = \dot{L}_n(\hat{\theta}_n) - \dot{L}_n(\vartheta) = \ddot{L}_n(\widetilde{\theta}_n)(\hat{\theta}_n - \vartheta) $$

for some $\widetilde{\theta}_n \in A_n$. Prove that $\ddot{L}_n(\widetilde{\theta}_n) \xrightarrow{\mathcal{L}} -\mathcal{I}(f(\cdot, \vartheta))$. 

We use the equation

$$\ddot{\ell}(x, \vartheta) = \frac{\ddot{f}(x, \vartheta)}{f(x,\vartheta)} - \dot{\ell}(x, \vartheta)\dot{\ell}(x, \vartheta)^T.$$

to show that

$$\mathbb{E}_\vartheta[\ddot{\ell}(X, \vartheta)] + \mathcal{I}(f(\cdot, \vartheta)) = \mathbb{E}_\vartheta\Big[ \frac{\ddot{f}(X, \vartheta)}{f(X,\vartheta)} \Big] = 0, $$

From Law of Large Numbers it follows that

$$\ddot{L}_n(\vartheta) \xrightarrow{\mathcal{L}} - \mathcal{I}(f(\cdot, \vartheta)).$$

Finally, we use the equality

$$\lim\limits_{\delta \rightarrow 0} \lim\limits_{n \rightarrow \infty} P_\vartheta(\| \widetilde{\theta}_n - \vartheta \| < \delta) = 1 $$

and continuity of $\ddot{\ell}$ over $\vartheta$ to finish the proof.

<span style="color:salmon">*Step 4.*</span> Now we conclude that

$$ \lim\limits_{n \rightarrow \infty} P_\vartheta(\ddot{L}_n(\widetilde{\theta}_n) \text{ is invertible}) = 1. $$

and applying Slutsky's lemma we get

$$\begin{aligned}
	\sqrt{n}(\hat{\theta}_n - \vartheta) & = -\ddot{L}_n(\widetilde{\theta}_n)^{-1} \sqrt{n} \dot{L}_n(\vartheta) \\
	& \rightarrow \mathcal{I}(f(\cdot, \vartheta))^{-1} \mathcal{N}(0, \mathcal{I}(f(\cdot, \vartheta))) \\&= \mathcal{N}(0, \mathcal{I}(f(\cdot, \vartheta))^{-1}). \color{Salmon}{\square}
	\end{aligned}$$


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

$$\mathcal{I}(f(\cdot, \lambda)) = \mathbb{E}_\lambda\Big[\Big(X - \frac{1}{\lambda}\Big)^2\Big]=\frac{1}{\lambda^2}.$$
	
By theorem of asymptotic efficiency of ML-estimators we get 

$$\sqrt{n}(\hat{\lambda}_n - \lambda) \xrightarrow[]{\mathcal{L}} \mathcal{N}(0, \lambda^2),$$

On the other hand by CLT

$$\sqrt{n}\Big(\overline{X}_n - \frac{1}{\lambda}\Big) \xrightarrow[]{\mathcal{L}} \mathcal{N}\Big(0, \frac{1}{\lambda^2}\Big).$$

Using Delta-method for $g(x) = x^{-1}$ we get the same result:

$$\sqrt{n}(\overline{X}_n^{-1} - \lambda) \xrightarrow[]{\mathcal{L}} \mathcal{N}(0, \lambda^2). $$