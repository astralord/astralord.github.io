---
layout: post
title: 'Visual Guide to Statistics. Part II: Bayesian Statistics'
date: 2022-03-21 11:11 +0800
categories: [Statistics, Visual Guide]
tags: [statistics, parameter estimation, bayesian inference, bayes estimator, minimax estimator, least favorable prior]
math: true
enable_d3: true
---

> Part II introduces different approach to parameters estimation called Bayesian statistics.

## Basic definitions

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

## Posterior risk

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

Let's take an example of an estimation of probability parameter for binomial distribution. Let $\Theta = (0, 1)$, $\mathcal{X} = \lbrace 0, \dots, n \rbrace$ and

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

$$ h(\mu) = \frac{1}{\sqrt{2 \pi \tau^2}} \exp \Big( -\frac{(\mu-\nu)^2}{2\tau^2} \Big). $$

Taking density for $X$

$$ f(x|\mu)=\Big( \frac{1}{\sqrt{2\pi \sigma^2}}\Big)^n \exp \Big( \frac{1}{2\sigma^2}\sum_{j=1}^n(x_j-\mu)^2 \Big ), $$

we get posterior distribution

$$ Q^{\mu|X=x} \sim \mathcal{N} \Big( g_{\nu, \tau^2}(x), \Big( \frac{n}{\sigma^2} + \frac{1}{\tau^2}\Big)^{-1}  \Big), $$

where

$$ g_{\nu, \tau^2}(x)=\Big( 1 + \frac{\sigma^2}{n \tau^2} \Big)^{-1} \overline{x}_n+\Big( \frac{n \tau^2}{\sigma^2}+1 \Big)^{-1} \nu. $$

For quadratic loss function $g_{\nu, \tau^2}(x)$ is a Bayes estimator. It can be interpreted as following: for large values of $\tau$ (not enough prior information) estimator $g_{\nu, \tau^2}(x) \approx \overline{x}_n$. 

Otherwise, $g_{\nu, \tau^2}(x)$ $\approx \nu$.

<script src="https://d3js.org/d3.v7.min.js"></script>
<link href="https://fonts.googleapis.com/css?family=Arvo" rel="stylesheet">

<style>

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
  stroke-width: 3px;
}

.track-inset {
  stroke: #bababa;
  stroke-width: 3px;
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

#n-text {
  font-family: Arvo;
  font-size: 11px;
}

#n-num {
  font-family: Arvo;
  font-size: 11px;
}
    
</style>

<button id="sample-button">Sample</button>
<label id="n-text">n:</label>
<input type="number" min="1" max="100" step="1" value="3" id="n-num">
<div id="gauss_bayes_plt"></div>

<script>

const COLORS = {
  prior: '#348ABD',
  posterior: '#EDA137',
  mu: '#65AD69',
  xBar: '#E86456',
  minimax: '#F5CC18'
};

const ANIM_DURATION = 1200;

d3.select("#gauss_bayes_plt")
  .style("position", "relative");

let mu = -1,
    sigma = 3,
    nu = 0,
    tau = 1,
    avg = -2,
    n = 3;

function randn_bm() {
  let u = 0, v = 0;
  while (u === 0) u = Math.random();
  while (v === 0) v = Math.random();
  return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
}

function gaussianPDF(x, mean, std) {
  return Math.exp(-0.5 * ((x - mean) / std) ** 2) / (std * Math.sqrt(2 * Math.PI));
}

function computeBayesEstimator(avg, n, sigma, tau, nu) {
  return avg / (1 + sigma ** 2 / (n * tau ** 2)) + nu / (1 + (n * tau ** 2) / sigma ** 2);
}

function computePosteriorStd(n, sigma, tau) {
  return 1 / Math.sqrt(n / sigma ** 2 + 1 / tau ** 2);
}

function gaussianCurveData(mean, std) {
  const pts = [{x: -7, y: 0}];
  for (let i = -7; i < 7; i += 0.01) {
    pts.push({x: i, y: gaussianPDF(i, mean, std)});
  }
  pts.push({x: 7, y: 0});
  return pts;
}

const margin = {top: 20, right: 0, bottom: 25, left: 25};
const width = 700 - margin.left - margin.right;
const height = 400 - margin.top - margin.bottom;
const fig_height = 200;

const svg = d3.select("#gauss_bayes_plt")
  .append("svg")
  .attr("width", width + margin.left + margin.right)
  .attr("height", height + margin.top + margin.bottom)
  .append("g")
  .attr("transform", `translate(${margin.left},${margin.top})`);

let prior_data = [], posterior_data = [];
updateData();

const x = d3.scaleLinear()
  .domain([d3.min(prior_data, d => d.x), d3.max(prior_data, d => d.x)])
  .range([0, width]);

const xAxis = svg.append("g")
  .attr("transform", `translate(0,${fig_height})`)
  .call(d3.axisBottom(x).ticks(8));

xAxis.selectAll(".tick text")
  .attr("font-family", "Arvo");

const y = d3.scaleLinear()
  .range([fig_height, 0])
  .domain([0, d3.max(posterior_data, d => d.y)]);

const yAxis = svg.append("g").call(d3.axisLeft(y).ticks(3));
yAxis.selectAll(".tick text")
  .attr("font-family", "Arvo");

let g, post_std, avg_y, mode_y, mu_y;

function updateData() {
  prior_data = gaussianCurveData(nu, tau);
  g = computeBayesEstimator(avg, n, sigma, tau, nu);
  post_std = computePosteriorStd(n, sigma, tau);
  posterior_data = gaussianCurveData(g, post_std);

  avg_y = gaussianPDF(avg, g, post_std);
  mode_y = gaussianPDF(g, g, post_std);
  mu_y = gaussianPDF(mu, g, post_std);
}

function dashedLine(svg, points) {
  return svg.append("path")
    .attr("class", "line")
    .style("stroke-dasharray", "3, 3")
    .attr("stroke", "currentColor")
    .attr("stroke-width", 1)
    .datum(points)
    .attr("d", d3.line()
      .x(d => d.x)
      .y(d => d.y));
}

function indicator(svg, x0, y0, color) {
  return svg.append('g')
    .selectAll("dot")
    .data([{x: x0, y: y0}])
    .enter()
    .append("circle")
      .attr("cx", d => d.x)
      .attr("cy", d => d.y)
      .attr("r", 3)
      .style("fill", color)
      .attr("stroke", "black")
      .attr("stroke-width", 1);
}

function legendStrip(svg, yPos, color) {
  svg.append("path")
    .attr("stroke", color)
    .attr("stroke-width", 4)
    .attr("opacity", ".9")
    .datum([{x: LEGEND_X, y: yPos}, {x: LEGEND_X + 25, y: yPos}])
    .attr("d", d3.line()
      .x(d => d.x)
      .y(d => d.y));

  svg.append("path")
    .attr("stroke", "currentColor")
    .attr("stroke-width", 1)
    .datum([{x: LEGEND_X, y: yPos - 2}, {x: LEGEND_X + 25, y: yPos - 2}])
    .attr("d", d3.line()
      .x(d => d.x)
      .y(d => d.y));
}

function updateCurves(delay) {
  updateData();

  y.domain([0, d3.max(posterior_data, d => d.y)]);
  yAxis
    .transition()
    .delay(delay)
    .duration(ANIM_DURATION)
    .call(d3.axisLeft(y).ticks(3))
    .selectAll(".tick text")
    .attr("font-family", "Arvo");

  prior_curve
    .datum(prior_data)
    .transition()
    .delay(delay)
    .duration(ANIM_DURATION)
    .attr("d", d3.line()
      .curve(d3.curveBasis)
      .x(d => x(d.x))
      .y(d => y(d.y))
    );

  posterior_curve
    .datum(posterior_data)
    .transition()
    .delay(delay)
    .duration(ANIM_DURATION)
    .attr("d", d3.line()
      .curve(d3.curveBasis)
      .x(d => x(d.x))
      .y(d => y(d.y))
    );

  avg_dash.datum([{x: avg, y: 0}, {x: avg, y: avg_y}])
    .transition()
    .delay(delay)
    .duration(ANIM_DURATION)
    .attr("d", d3.line()
      .x(d => x(d.x))
      .y(d => y(d.y))
    );

  avg_dot
    .transition()
    .delay(delay)
    .duration(ANIM_DURATION)
    .attr("cx", () => x(avg))
    .attr("cy", () => y(avg_y));

  mode_dash.datum([{x: g, y: 0}, {x: g, y: mode_y}])
    .transition()
    .delay(delay)
    .duration(ANIM_DURATION)
    .attr("d", d3.line()
      .x(d => x(d.x))
      .y(d => y(d.y))
    );

  mode_dot
    .transition()
    .delay(delay)
    .duration(ANIM_DURATION)
    .attr("cx", () => x(g))
    .attr("cy", () => y(mode_y));

  mu_dash.datum([{x: mu, y: 0}, {x: mu, y: mu_y}])
    .transition()
    .delay(delay)
    .duration(ANIM_DURATION)
    .attr("d", d3.line()
      .x(d => x(d.x))
      .y(d => y(d.y))
    );

  mu_dot
    .transition()
    .delay(delay)
    .duration(ANIM_DURATION)
    .attr("cx", () => x(mu))
    .attr("cy", () => y(mu_y));
}

const prior_curve = svg
  .append('g')
  .append("path")
    .datum(prior_data)
    .attr("fill", COLORS.prior)
    .attr("border", 0)
    .attr("opacity", ".9")
    .attr("stroke", "currentColor")
    .attr("stroke-width", 1)
    .attr("stroke-linejoin", "round")
    .attr("d", d3.line()
      .curve(d3.curveBasis)
      .x(d => x(d.x))
      .y(d => y(d.y))
    );

const posterior_curve = svg
  .append('g')
  .append("path")
    .datum(posterior_data)
    .attr("fill", COLORS.posterior)
    .attr("border", 0)
    .attr("opacity", ".9")
    .attr("stroke", "currentColor")
    .attr("stroke-width", 1)
    .attr("stroke-linejoin", "round")
    .attr("d", d3.line()
      .curve(d3.curveBasis)
      .x(d => x(d.x))
      .y(d => y(d.y))
    );

const avg_dash = svg.append("path")
  .attr("class", "line")
  .style("stroke-dasharray", "3, 3")
  .attr("stroke", "black")
  .attr("stroke-width", 1)
  .datum([{x: avg, y: avg_y}, {x: avg, y: 0}])
  .attr("d", d3.line()
    .x(d => x(d.x))
    .y(d => y(d.y)));

const mode_dash = svg.append("path")
  .attr("class", "line")
  .style("stroke-dasharray", "3, 3")
  .attr("stroke", "black")
  .attr("stroke-width", 1)
  .datum([{x: g, y: mode_y}, {x: g, y: 0}])
  .attr("d", d3.line()
    .x(d => x(d.x))
    .y(d => y(d.y)));

const mu_dash = svg.append("path")
  .attr("class", "line")
  .style("stroke-dasharray", "3, 3")
  .attr("stroke", "black")
  .attr("stroke-width", 1)
  .datum([{x: mu, y: mu_y}, {x: mu, y: 0}])
  .attr("d", d3.line()
    .x(d => x(d.x))
    .y(d => y(d.y)));

const mu_dot = svg.append('g')
  .selectAll("dot")
  .data([{x: mu, y: mu_y}])
  .enter()
  .append("circle")
    .attr("cx", d => x(d.x))
    .attr("cy", d => y(d.y))
    .attr("r", 3)
    .style("fill", COLORS.mu)
    .attr("stroke", "black")
    .attr("stroke-width", 1);

const avg_dot = svg.append('g')
  .selectAll("dot")
  .data([{x: avg, y: avg_y}])
  .enter()
  .append("circle")
    .attr("cx", d => x(d.x))
    .attr("cy", d => y(d.y))
    .attr("r", 3)
    .style("fill", COLORS.xBar)
    .attr("stroke", "black")
    .attr("stroke-width", 1);

const mode_dot = svg.append('g')
  .selectAll("dot")
  .data([{x: g, y: mode_y}])
  .enter()
  .append("circle")
    .attr("cx", d => x(d.x))
    .attr("cy", d => y(d.y))
    .attr("r", 3)
    .style("fill", COLORS.prior)
    .attr("stroke", "black")
    .attr("stroke-width", 1);

const LEGEND_X = 550;

legendStrip(svg, -5, COLORS.prior);
svg.append("text")
  .attr("text-anchor", "start")
  .attr("y", 0)
  .attr("x", LEGEND_X + 30)
  .attr("font-family", "Arvo")
  .attr("font-weight", 700)
  .text("Prior")
  .style("fill", COLORS.prior);

legendStrip(svg, 15, COLORS.posterior);
svg.append("text")
  .attr("text-anchor", "start")
  .attr("y", 20)
  .attr("x", LEGEND_X + 30)
  .attr("font-family", "Arvo")
  .attr("font-weight", 700)
  .text("Posterior")
  .style("fill", COLORS.posterior);

d3.select("#gauss_bayes_plt")
  .append("div")
  .text("\\(\\mu \\)")
  .style('color', COLORS.mu)
  .style("font-size", "13px")
  .style("font-weight", "700")
  .attr("font-family", "Arvo")
  .attr("font-weight", 700)
  .style("position", "absolute")
  .style("left", `${LEGEND_X + 55}px`)
  .style("top", "45px");

dashedLine(svg, [{x: LEGEND_X + 20, y: 30}, {x: LEGEND_X + 20, y: 45}]);
indicator(svg, LEGEND_X + 20, 30, COLORS.mu);

d3.select("#gauss_bayes_plt")
  .append("div")
  .text("\\(\\overline{X}_n \\)")
  .style('color', COLORS.xBar)
  .style("font-size", "13px")
  .style("font-weight", "700")
  .attr("font-family", "Arvo")
  .attr("font-weight", 700)
  .style("position", "absolute")
  .style("left", `${LEGEND_X + 55}px`)
  .style("top", "75px");

dashedLine(svg, [{x: LEGEND_X + 20, y: 60}, {x: LEGEND_X + 20, y: 75}]);
indicator(svg, LEGEND_X + 20, 60, COLORS.xBar);

svg.append("text")
  .attr("text-anchor", "start")
  .attr("y", 100)
  .attr("x", LEGEND_X + 30)
  .attr("font-family", "Arvo")
  .attr("font-weight", 700)
  .text("Bayes")
  .attr("font-size", 12)
  .style("fill", COLORS.prior);

dashedLine(svg, [{x: LEGEND_X + 20, y: 90}, {x: LEGEND_X + 20, y: 105}]);
indicator(svg, LEGEND_X + 20, 90, COLORS.prior);

const mu_x = d3.scaleLinear()
  .domain([-3, 3])
  .range([0, width / 3])
  .clamp(true);

const sigma_x = d3.scaleLinear()
  .domain([0.1, 3])
  .range([0, width / 3])
  .clamp(true);

const nu_x = d3.scaleLinear()
  .domain([-3, 3])
  .range([0, width / 3])
  .clamp(true);

const tau_x = d3.scaleLinear()
  .domain([0.1, 3])
  .range([0, width / 3])
  .clamp(true);

function createSlider(svg_, parameter_update, x, loc_x, loc_y, letter, color, init_val, round_fun) {
  const slider = svg_.append("g")
    .attr("class", "slider")
    .attr("transform", `translate(${loc_x},${loc_y})`);

  const drag = d3.drag()
    .on("start.interrupt", function() { slider.interrupt(); })
    .on("start drag", function(event, d) {
      handle.attr("cx", x(round_fun(x.invert(event.x))));
      parameter_update(x.invert(event.x));
      updateCurves(0);
    });

  slider.append("line")
    .attr("class", "track")
    .attr("x1", x.range()[0])
    .attr("x2", x.range()[1])
    .select(function() { return this.parentNode.appendChild(this.cloneNode(true)); })
      .attr("class", "track-inset")
    .select(function() { return this.parentNode.appendChild(this.cloneNode(true)); })
      .attr("class", "track-overlay")
      .call(drag)
      .attr('stroke-width', 1);

  slider.insert("g", ".track-overlay")
    .attr("class", "ticks")
    .attr("transform", `translate(0,18)`)
    .selectAll("text")
    .data(x.ticks(6))
    .enter().append("text")
      .attr("x", x)
      .attr("text-anchor", "middle")
      .attr("font-family", "Arvo")
      .style('fill', "currentColor")
      .text(d => d);

  const handle = slider.insert("circle", ".track-overlay")
    .attr("class", "handle")
    .attr("r", 5)
    .attr("cx", x(init_val));

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

function updateNu(x) { nu = x; }
function updateTau(x) { tau = x; }
function updateMu(x) { mu = x; }
function updateSigma(x) { sigma = x; }
function trivialRound(x) { return x; }

createSlider(svg, updateMu, mu_x, margin.left, 0.75 * height, "", COLORS.mu, mu, trivialRound);
createSlider(svg, updateSigma, sigma_x, margin.left, 0.9 * height, "", COLORS.mu, sigma, trivialRound);
createSlider(svg, updateNu, nu_x, margin.left + width / 2, 0.75 * height, "", COLORS.prior, nu, trivialRound);
createSlider(svg, updateTau, tau_x, margin.left + width / 2, 0.9 * height, "", COLORS.prior, tau, trivialRound);

d3.select("#gauss_bayes_plt")
  .append("div")
  .text("\\(\\mu \\)")
  .style('color', COLORS.mu)
  .style("font-size", "17px")
  .style("font-weight", "700")
  .attr("font-family", "Arvo")
  .attr("font-weight", 700)
  .style("position", "absolute")
  .style("left", `${margin.left}px`)
  .style("top", `${0.75 * height + 5}px`);

d3.select("#gauss_bayes_plt")
  .append("div")
  .text("\\(\\sigma \\)")
  .style('color', COLORS.mu)
  .style("font-size", "17px")
  .style("font-weight", "700")
  .attr("font-family", "Arvo")
  .attr("font-weight", 700)
  .style("position", "absolute")
  .style("left", `${margin.left}px`)
  .style("top", `${0.9 * height + 5}px`);

d3.select("#gauss_bayes_plt")
  .append("div")
  .text("\\(\\nu \\)")
  .style('color', COLORS.prior)
  .style("font-size", "17px")
  .style("font-weight", "700")
  .attr("font-family", "Arvo")
  .attr("font-weight", 700)
  .style("position", "absolute")
  .style("left", `${margin.left + width / 2}px`)
  .style("top", `${0.75 * height + 5}px`);

d3.select("#gauss_bayes_plt")
  .append("div")
  .text("\\(\\tau \\)")
  .style('color', COLORS.prior)
  .style("font-size", "17px")
  .style("font-weight", "700")
  .attr("font-family", "Arvo")
  .attr("font-weight", 700)
  .style("position", "absolute")
  .style("left", `${margin.left + width / 2}px`)
  .style("top", `${0.9 * height + 5}px`);

d3.select("#n-num").on("input", function() {
  n = this.value;
  updateCurves(0);
});

const sampleButton = d3.select("#sample-button");

sampleButton
  .on("click", function() {
    const random_samples = [];
    const smpl_dots = [];
    avg = 0;
    for (let i = 0; i < n; i += 1) {
      random_samples.push(mu + sigma * randn_bm());
      smpl_dots.push(svg.append('g')
        .selectAll("dot")
        .data([{x: random_samples[i], y: d3.max(posterior_data, d => d.y)}])
        .enter()
        .append("circle")
          .attr("cx", d => x(d.x))
          .attr("cy", d => y(d.y))
          .attr("r", 3)
          .style("fill", COLORS.mu)
          .attr("stroke", "black")
          .attr("stroke-width", 1));

      smpl_dots[i].transition()
        .duration(ANIM_DURATION)
        .attr("cx", () => x(random_samples[i]))
        .attr("cy", () => y(0));

      avg += random_samples[i];
    }
    avg /= n;

    for (let i = 0; i < n; i += 1) {
      smpl_dots[i]
        .transition()
        .delay(ANIM_DURATION)
        .duration(ANIM_DURATION)
        .style("fill", COLORS.xBar)
        .attr("cx", () => x(avg))
        .attr("cy", () => y(0));

      smpl_dots[i]
        .transition()
        .delay(2.5 * ANIM_DURATION)
        .duration(ANIM_DURATION)
        .attr("r", 0);

      smpl_dots[i].transition().delay(3.5 * ANIM_DURATION).remove();
    }

    updateCurves(2 * ANIM_DURATION);
  });

</script>

![](.)
*Fig. 1. Bayesian inference for normal distribution.*

## Minimax estimator

For an estimator $g$

$$ R^*(g) = \sup_{\vartheta \in \Theta} R(\vartheta, g)$$

is called the **maximum risk** and

$$ R^*(g^*) = \inf_{g \in \mathcal{K}} R^*(g) $$

is **minimax risk** and corresponding $g$ - **minimax estimator**. The use of minimax estimator is aimed at protecting against large losses. Also it's not hard to see, that

$$ R^*(g) = \sup_{\pi \in \mathcal{M}} R(\pi, g), $$

where $\mathcal{M}$ is a set of all prior measures $\pi$. If for some $\pi^*$ we have

$$ \inf_{g \in \mathcal{K}} R(\pi^*, g) \geq \inf_{g \in \mathcal{K}} R(\pi, g) \quad \forall \pi \in \mathcal{M}, $$

then $\pi^*$ is called the **least favorable prior**. If $g_\pi$ is a Bayes estimator for prior $\pi$ and also

$$ R(\pi, g_\pi) = \sup_{\vartheta \in \Theta} R(\vartheta, g_\pi),$$ 

then for any $g \in \mathcal{K}$:

$$ \sup_{\vartheta \in \Theta}R(\vartheta, g) \geq \int_{\Theta}R(\vartheta, g)\pi(d\vartheta) \geq \int_{\Theta}R(\vartheta, g_\pi)\pi(d\vartheta)=R(\pi, g_\pi)=\sup_{\vartheta \in \Theta}R(\vartheta, g_\pi) $$

and therefore $g_\pi$ is a minimax estimator. Also, $\pi$ is a least favorable prior, because for any distribution $\mu$

$$ 
\begin{aligned}
\inf_{g \in \mathcal{K}} \int_{\Theta} R(\vartheta, g)\mu(d\vartheta) &\leq \int_{\Theta}R(\vartheta, g_\pi)\mu(d\vartheta) \\& \leq \sup_{\vartheta \in \Theta} R(\vartheta, g_\pi) \\&= R(\pi, g_\pi) \\ &= \inf_{g \in \mathcal{K}} \int_{\Theta}R(\vartheta, g) \pi(d\vartheta).
\end{aligned} $$

Sometimes Bayes risk can be constant:

$$ R(\vartheta, g_\pi) = c \quad \forall \vartheta \in \Theta. $$

Then

$$ \sup_{\vartheta \in \Theta} R(\vartheta, g_\pi) = c = \int_{\Theta} R(\vartheta, g_\pi) \pi(d\vartheta) = R(\pi, g_\pi), $$

$g_\pi$ is minimax and $\pi$ is least favorable prior.

Let's get back to an example with binomial distribution:

$$ P_\vartheta(X = x) = \binom{n}{x} \vartheta^x (1-\vartheta)^{n-x}. $$

Again we use quadratic loss, but only this time we take parameterized beta distrubution $B(a, b)$ as our prior:

$$ h(\vartheta) = \frac{\vartheta^{a-1}(1-\vartheta)^{b-1}1_{[0,1]}(\vartheta)}{B(a, b)}. $$

Note that for $a = b = 1$ we have $\theta \sim \mathcal{U}(0, 1)$. Now posterior distribution will be $Q^{\vartheta \mid X=x} \sim B(x+a,n-x+b)$ with density

$$  f(\vartheta | x)= \frac{\vartheta^{x+a-1}(1-\vartheta)^{n-x+b-1}1_{[0,1](\vartheta)}}{B(x+a,n-x+b)}. $$

We use our prior knowledge that for random variable $Z \sim B(p, q)$

$$ \mathbb{E}[Z] = \frac{p}{p+q} \quad \text{and} \quad \operatorname{Var}(Z)=\frac{pq}{(p+q)^2(p+q+1)}. $$

Recall that for quadratic loss expected value of $\theta$ is Bayes estimator. Therefore,

$$ g_{a,b}(x)=\frac{x+a}{n+a+b} $$

is a Bayes estimator and it provides risk

$$ \begin{aligned} R(\vartheta, g_{a,b})&=\mathbb{E}[(g_{a,b}(X)-\vartheta)^2] \\ &=\frac{\vartheta^2(-n+(a+b)^2+\vartheta(n-2a(a+b))+a^2}{(n+a+b)^2}. \end{aligned}$$

If we choose $\hat{a}=\hat{b}=\frac{\sqrt{n}}{2}$ then risk will be

$$  R(\vartheta, g_{\hat{a}, \hat{b}})=\frac{1}{4(\sqrt{n} + 1)^2}. $$

Such risk doesn't depend on $\vartheta$ and hence an estimator $g_{\hat{a}, \hat{b}}(x) = \frac{x+\sqrt{n}/2}{n+\sqrt{n}}$ is minimax and $B(\hat{a}, \hat{b})$ is least favorable prior.

<style>
  #minimax-button {
  top: 15px;
  left: 15px;
  background: #F5CC18;
  padding-right: 26px;
  border-radius: 3px;
  border: none;
  color: white;
  margin: 0;
  padding: 0 1px;
  width: 90px;
  height: 25px;
  font-family: Arvo;
  font-size: 11px;
}

#minimax-button:hover {
  background-color: #696969;
}
</style>


<div id="bin_bayes_plt">
  <button id="minimax-button">Least fav. prior</button>
</div>

<script>

d3.json("../../../../assets/beta.json").then(data => {
  const COLORS2 = {
    prior: '#348ABD',
    posterior: '#EDA137',
    mu: '#65AD69',
    xBar: '#E86456',
    minimax: '#F5CC18'
  };

  let sample = 1;
  let n = 8;
  let a = 1, b = 1;
  let a_key = 10, b_key = 10;

  const margin_main = {top: 25, right: 0, bottom: 25, left: 20};
  const width = 800 - margin_main.left - margin_main.right;
  const height = 200 - margin_main.top - margin_main.bottom;
  const fig_width = 200;

  const prior_svg = d3.select("#bin_bayes_plt")
    .append("svg")
    .attr("width", width + margin_main.left + margin_main.right)
    .attr("height", height + margin_main.top + margin_main.bottom)
    .append("g")
    .attr("transform", `translate(${margin_main.left},${margin_main.top})`);

  let prior_data = [];
  updatePriorData();

  const x = d3.scaleLinear()
    .domain([d3.min(prior_data, d => d.x), d3.max(prior_data, d => d.x)])
    .range([0, fig_width]);

  const xAxis = prior_svg.append("g")
    .attr("transform", `translate(0,${height})`)
    .call(d3.axisBottom(x).ticks(4));

  xAxis.selectAll(".tick text")
    .attr("font-family", "Arvo");

  const y = d3.scaleLinear()
    .range([height, 0])
    .domain([0, 12]);

  const yAxis = prior_svg.append("g").call(d3.axisLeft(y).ticks(3));
  yAxis.selectAll(".tick text")
    .attr("font-family", "Arvo");

  const prior_curve = prior_svg
    .append('g')
    .append("path")
      .datum(prior_data)
      .attr("fill", COLORS2.prior)
      .attr("border", 0)
      .attr("opacity", ".9")
      .attr("stroke", "currentColor")
      .attr("stroke-width", 1)
      .attr("stroke-linejoin", "round")
      .attr("d", d3.line()
        .curve(d3.curveBasis)
        .x(d => x(d.x))
        .y(d => y(d.y))
      );

  prior_svg
    .append("text")
    .attr("text-anchor", "start")
    .attr("y", 40)
    .attr("x", 80)
    .attr("font-family", "Arvo")
    .attr("font-weight", 700)
    .text("Prior")
    .style("fill", COLORS2.prior);

  const margin_smpl = {top: 0, right: 0, bottom: 35, left: 250};

  const smpl_svg = prior_svg
    .append("svg")
      .attr("width", width + margin_smpl.left + margin_smpl.right)
      .attr("height", height + margin_smpl.top + margin_smpl.bottom)
    .append("g")
      .attr("transform", `translate(${margin_smpl.left},${margin_smpl.top})`);

  const smpl_x = d3.scaleBand()
    .range([0, fig_width]);
  const smpl_x_axis = smpl_svg.append("g")
    .attr("transform", `translate(0,${height})`);

  const smpl_y = d3.scaleLinear().range([height, 0]).domain([0, 1]);
  const smpl_y_axis = smpl_svg.append("g").call(d3.axisLeft(smpl_y).ticks(0));

  function updateRectSample() {
    const rect_data = [];
    for (let i = 0; i <= n; i++) {
      rect_data.push({x: i, y: 1});
    }

    smpl_x.domain(rect_data.map(d => d.x));
    smpl_x_axis.call(d3.axisBottom(smpl_x));
    smpl_x_axis.selectAll(".tick text").attr("font-family", "Arvo");

    const rect_sample = smpl_svg.selectAll("rect").data(rect_data);

    rect_sample.enter()
      .append("rect")
        .merge(rect_sample)
        .attr("x", d => smpl_x(d.x))
        .attr("y", d => smpl_y(d.y))
        .attr("width", smpl_x.bandwidth())
        .attr("border", 0)
        .attr("opacity", d => d.x == sample ? ".8" : "0")
        .attr("stroke", "currentColor")
        .attr("stroke-width", 1)
        .attr("stroke-linejoin", "round")
        .attr("height", d => height - smpl_y(d.y))
        .attr("fill", COLORS2.mu)
        .on('mouseover', function(event, d) {
          d3.select(this)
            .transition()
            .attr("opacity", d => d.x == sample ? ".8" : ".4");
        })
        .on('mouseout', function(event, d) {
          d3.select(this)
            .transition()
            .attr("opacity", d => d.x == sample ? ".8" : "0");
        })
        .on('click', function(event, d) {
          sample = d.x;

          d3.selectAll("rect")
            .transition()
            .attr("x", d2 => smpl_x(d2.x))
            .attr("y", d2 => smpl_y(d2.y))
            .attr("opacity", d2 => d2.x == sample ? ".8" : "0");

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
    .style("fill", COLORS2.mu);

  const margin_post = {top: 0, right: 0, bottom: 35, left: 250};

  const post_svg = smpl_svg
    .append("svg")
      .attr("width", width + margin_post.left + margin_post.right)
      .attr("height", height + margin_post.top + margin_post.bottom)
    .append("g")
      .attr("transform", `translate(${margin_post.left},${margin_post.top})`);

  const xAxis2 = post_svg.append("g")
    .attr("transform", `translate(0,${height})`)
    .call(d3.axisBottom(x).ticks(4));

  xAxis2.selectAll(".tick text")
    .attr("font-family", "Arvo");

  const yAxis2 = post_svg.append("g").call(d3.axisLeft(y).ticks(3));
  yAxis2.selectAll(".tick text")
    .attr("font-family", "Arvo");

  post_svg
    .append("text")
    .attr("text-anchor", "start")
    .attr("y", 40)
    .attr("x", 65)
    .attr("font-family", "Arvo")
    .attr("font-weight", 700)
    .text("Posterior")
    .style("fill", COLORS2.posterior);

  function postLegendItem(svg, yLabel, yDash, yDot, label, color) {
    svg
      .append("text")
      .attr("text-anchor", "start")
      .attr("y", yLabel)
      .attr("x", 217)
      .attr("font-family", "Arvo")
      .attr("font-weight", 700)
      .attr("font-size", 10)
      .text(label)
      .style("fill", color);

    svg.append("path")
      .attr("class", "line")
      .style("stroke-dasharray", "3, 3")
      .attr("stroke", "currentColor")
      .attr("stroke-width", 1)
      .datum([{x: 210, y: yDash}, {x: 210, y: yDot}])
      .attr("d", d3.line()
        .x(d => d.x)
        .y(d => d.y));

    svg.append('g')
      .selectAll("dot")
      .data([{x: 210, y: yDash}])
      .enter()
      .append("circle")
        .attr("cx", d => d.x)
        .attr("cy", d => d.y)
        .attr("r", 3)
        .style("fill", color)
        .attr("stroke", "black")
        .attr("stroke-width", 1);
  }

  postLegendItem(post_svg, 55, 45, 60, "UMVU", COLORS2.xBar);
  postLegendItem(post_svg, 85, 75, 90, "Bayes", COLORS2.prior);
  postLegendItem(post_svg, 115, 105, 120, "Minimax", COLORS2.minimax);

  let posterior_data = [];
  updatePosteriorData();

  const posterior_curve = post_svg
    .append('g')
    .append("path")
      .datum(posterior_data)
      .attr("fill", COLORS2.posterior)
      .attr("border", 0)
      .attr("opacity", ".9")
      .attr("stroke", "currentColor")
      .attr("stroke-width", 1)
      .attr("stroke-linejoin", "round")
      .attr("d", d3.line()
        .curve(d3.curveBasis)
        .x(d => x(d.x))
        .y(d => y(d.y))
      );

  let umvu_x = sample / n;
  let umvu_y = Math.pow(umvu_x, sample + a - 1) * Math.pow(1 - umvu_x, n - sample + b - 1) / data[n][a_key][b_key][sample];

  const umvu_dash = post_svg.append("path")
    .attr("class", "line")
    .style("stroke-dasharray", "3, 3")
    .attr("stroke", "black")
    .attr("stroke-width", 1)
    .datum([{x: umvu_x, y: umvu_y}, {x: umvu_x, y: 0}])
    .attr("d", d3.line()
      .x(d => x(d.x))
      .y(d => y(d.y)));

  const umvu_dot = post_svg.append('g')
    .selectAll("dot")
    .data([{x: umvu_x, y: umvu_y}])
    .enter()
    .append("circle")
      .attr("cx", d => x(d.x))
      .attr("cy", d => y(d.y))
      .attr("r", 3)
      .style("fill", COLORS2.xBar)
      .attr("stroke", "black")
      .attr("stroke-width", 1);

  let bayes_x = (sample + 1) / (n + 2);
  let bayes_y = Math.pow(bayes_x, sample + a - 1) * Math.pow(1 - bayes_x, n - sample + b - 1) / data[n][a_key][b_key][sample];

  const bayes_dash = post_svg.append("path")
    .attr("class", "line")
    .style("stroke-dasharray", "3, 3")
    .attr("stroke", "black")
    .attr("stroke-width", 1)
    .datum([{x: bayes_x, y: bayes_y}, {x: bayes_x, y: 0}])
    .attr("d", d3.line()
      .x(d => x(d.x))
      .y(d => y(d.y)));

  const bayes_dot = post_svg.append('g')
    .selectAll("dot")
    .data([{x: bayes_x, y: bayes_y}])
    .enter()
    .append("circle")
      .attr("cx", d => x(d.x))
      .attr("cy", d => y(d.y))
      .attr("r", 3)
      .style("fill", COLORS2.prior)
      .attr("stroke", "black")
      .attr("stroke-width", 1);

  let minimax_x = (sample + Math.sqrt(n) / 2) / (n + Math.sqrt(n));
  let minimax_y = Math.pow(minimax_x, sample + a - 1) * Math.pow(1 - minimax_x, n - sample + b - 1) / data[n][a_key][b_key][sample];

  const minimax_dash = post_svg.append("path")
    .attr("class", "line")
    .style("stroke-dasharray", "3, 3")
    .attr("stroke", "black")
    .attr("stroke-width", 1)
    .datum([{x: minimax_x, y: minimax_y}, {x: minimax_x, y: 0}])
    .attr("d", d3.line()
      .x(d => x(d.x))
      .y(d => y(d.y)));

  const minimax_dot = post_svg.append('g')
    .selectAll("dot")
    .data([{x: minimax_x, y: minimax_y}])
    .enter()
    .append("circle")
      .attr("cx", d => x(d.x))
      .attr("cy", d => y(d.y))
      .attr("r", 3)
      .style("fill", COLORS2.minimax)
      .attr("stroke", "black")
      .attr("stroke-width", 1);

  function updatePriorData() {
    prior_data = [];
    prior_data.push({x: 0, y: 0});
    prior_data.push({x: 0, y: (a == 1 ? 1 : 0) / data[0][a_key][b_key][0]});

    for (let i = 0.002; i < 1; i += 0.002) {
      prior_data.push({x: i, y: Math.pow(i, a - 1) * Math.pow(1 - i, b - 1) / data[0][a_key][b_key][0]});
    }

    prior_data.push({x: 1, y: (b == 1 ? 1 : 0) / data[0][a_key][b_key][0]});
    prior_data.push({x: 1, y: 0});
  }

  function updatePriorCurve() {
    updatePriorData();

    prior_curve
      .datum(prior_data)
      .transition()
      .duration(1000)
      .attr("d", d3.line()
        .curve(d3.curveBasis)
        .x(d => x(d.x))
        .y(d => y(d.y))
      );
  }

  function updatePosteriorData() {
    posterior_data = [];
    posterior_data.push({x: 0, y: 0});
    posterior_data.push({x: 0, y: (sample < 1 - a ? 1 : 0) / data[n][a_key][b_key][sample]});

    for (let i = 0.002; i < 1; i += 0.002) {
      posterior_data.push({x: i, y: Math.pow(i, sample + a - 1) * Math.pow(1 - i, n - sample + b - 1) / data[n][a_key][b_key][sample]});
    }

    posterior_data.push({x: 1, y: (sample < n + b - 1 ? 0 : 1) / data[n][a_key][b_key][sample]});
    posterior_data.push({x: 1, y: 0});

    umvu_x = sample / n;
    if ((umvu_x == 0 && sample < 1 - a) || (umvu_x == 1 && n - sample < 1 - b)) {
      umvu_y = 13;
    } else {
      umvu_y = Math.pow(umvu_x, sample + a - 1) * Math.pow(1 - umvu_x, n - sample + b - 1) / data[n][a_key][b_key][sample];
    }

    bayes_x = (sample + a) / (n + a + b);
    bayes_y = Math.pow(bayes_x, sample + a - 1) * Math.pow(1 - bayes_x, n - sample + b - 1) / data[n][a_key][b_key][sample];

    minimax_x = (sample + Math.sqrt(n) / 2) / (n + Math.sqrt(n));
    minimax_y = Math.pow(minimax_x, sample + a - 1) * Math.pow(1 - minimax_x, n - sample + b - 1) / data[n][a_key][b_key][sample];
  }

  function updatePosteriorCurve() {
    updatePosteriorData();

    posterior_curve
      .datum(posterior_data)
      .transition()
      .duration(1000)
      .attr("d", d3.line()
        .curve(d3.curveBasis)
        .x(d => x(d.x))
        .y(d => y(d.y))
      );

    umvu_dot
      .transition()
      .duration(1000)
      .attr("cx", () => x(umvu_x))
      .attr("cy", () => y(umvu_y));

    umvu_dash
      .datum([{x: umvu_x, y: 0}, {x: umvu_x, y: umvu_y}])
      .transition()
      .duration(1000)
      .attr("d", d3.line()
        .curve(d3.curveBasis)
        .x(d => x(d.x))
        .y(d => y(d.y))
      );

    bayes_dot
      .transition()
      .duration(1000)
      .attr("cx", () => x(bayes_x))
      .attr("cy", () => y(bayes_y));

    bayes_dash
      .datum([{x: bayes_x, y: 0}, {x: bayes_x, y: bayes_y}])
      .transition()
      .duration(1000)
      .attr("d", d3.line()
        .curve(d3.curveBasis)
        .x(d => x(d.x))
        .y(d => y(d.y))
      );

    minimax_dot
      .transition()
      .duration(1000)
      .attr("cx", () => x(minimax_x))
      .attr("cy", () => y(minimax_y));

    minimax_dash
      .datum([{x: minimax_x, y: 0}, {x: minimax_x, y: minimax_y}])
      .transition()
      .duration(1000)
      .attr("d", d3.line()
        .curve(d3.curveBasis)
        .x(d => x(d.x))
        .y(d => y(d.y))
      );
  }

  function updateN(value) {
    n = parseInt(value);
    sample = Math.min(sample, n);
    updateRectSample();
    updatePosteriorCurve();
  }

  function updatePrior(a_val, b_val) {
    a_key = parseInt(a_val);
    b_key = parseInt(b_val);
    a = 0.1 * a_key;
    b = 0.1 * b_key;
    updatePriorCurve();
    updatePosteriorCurve();
  }

  const slider_svg = d3.select("#bin_bayes_plt")
    .append("svg")
    .attr("width", width + 20)
    .attr("height", 100)
    .append("g")
    .attr("transform", `translate(25,20)`);

  const n_x = d3.scaleLinear()
    .domain([1, 10])
    .range([0, width * 0.22])
    .clamp(true);

  const a_x = d3.scaleLinear()
    .domain([0.1, 3])
    .range([0, width * 0.22])
    .clamp(true);

  const b_x = d3.scaleLinear()
    .domain([0.1, 3])
    .range([0, width * 0.22])
    .clamp(true);

  function roundN(x) { return Math.round(x - 0.5); }
  function roundAB(x) { return 0.1 * Math.round(10 * x - 0.5); }

  function updateA(x) { updatePrior(10 * x, b_key.toString()); }
  function updateB(x) { updatePrior(a_key.toString(), 10 * x); }

  createSlider(slider_svg, updateN, n_x, 260, 0.05 * height, "n", COLORS2.mu, n, roundN);
  const handleA = createSlider(slider_svg, updateA, a_x, 10, 0.05 * height, "a", COLORS2.prior, a, roundAB);
  const handleB = createSlider(slider_svg, updateB, b_x, 10, 0.3 * height, "b", COLORS2.prior, b, roundAB);

  const minimaxButton = d3.select("#minimax-button");

  minimaxButton
    .on("click", function() {
      const value = Math.round(10 * Math.sqrt(n) / 2);
      handleA.attr("cx", a_x(0.1 * value));
      handleB.attr("cx", b_x(0.1 * value));
      updatePrior(value, value);
    });

});

</script>

![](.)
*Fig. 2. Bayesian inference for binomial distribution. Note that when least favorable prior is chosen, Bayes and minimax estimators coincide regardless of the sample value.*

## Least favorable sequence of priors

Let 

$$r_\pi = \inf_{g \in \mathcal{K}} R(\pi, g), \quad \pi \in \mathcal{M}.$$

Then sequence $(\pi_m)_{m \in \mathbb{N}}$ in $\mathcal{M}$ is called **least favorable sequence of priors** if

* $\lim_{m \rightarrow \infty} r_{\pi_m} = r$,
* $r_\pi \leq r\ $ $\ \forall \pi \in \mathcal{M}$.

Let $(\pi_m)$ in $\mathcal{M}$ be a sequence, such that $r_{\pi_m} \rightarrow r \in \mathbb{R}$. Also let there be an estimator $g^* \in \mathcal{K}$, such that

$$\sup_{\vartheta \in \Theta}R(\theta, g^*) = r.$$

Then 

$$\sup_{\vartheta \in \Theta} R(\vartheta, g) \geq \int_{\Theta} R(\vartheta, g) \pi_m(d \vartheta) \geq r_{\pi_m} \rightarrow r = \sup_{\vartheta \in \Theta}R(\theta, g^*)$$

and therefore $g^*$ is minimax. Also for any $\pi \in \mathcal{M}$

$$r_\pi \leq R(\pi, g^*) = \int_\Theta R(\vartheta, g^*) \pi (d\vartheta) \leq \sup_{\vartheta \in \Theta} R(\vartheta, g^*) = r,$$

hence $(\pi_m)$ is a least favorable sequence of priors.

Let's get back to our previous example of estimating mean for normal distribution with known $\sigma^2$. Say, we have prior distribution 

$$h_m(\mu)=\frac{1}{\sqrt{2 \pi m}} \exp \Big \{ -\frac{(\mu-\nu)^2}{2m}\Big \}.$$ 

with $m \in \mathbb{N}$. Recall that Bayes estimator is

$$ g_{\nu, m}(x)=\Big( 1 + \frac{\sigma^2}{n m} \Big)^{-1} \overline{x}_n+\Big( \frac{n m}{\sigma^2}+1 \Big)^{-1} \nu. $$

For any $\mu \in \mathbb{R}$

$$
	\begin{aligned}
	R(\mu, g_{\nu, m}) & = \mathbb{E}[(g_{\nu, m}(X)-\mu)^2] \\
	& = \mathbb{E}\Bigg[\bigg(\Big( 1 + \frac{\sigma^2}{n m} \Big)^{-1} (\overline{X}_n-\mu)+\Big( \frac{n m}{\sigma^2}+1 \Big)^{-1} (\nu-\mu)\bigg)^2\Bigg] \\
	& = \Big(1 + \frac{\sigma^2}{nm}\Big)^{-2} \frac{\sigma^2}{n} + \Big( 1+\frac{nm}{\sigma^2} \Big)^{-2}(\nu-\mu)^2 \xrightarrow[m \ \rightarrow \infty]{} \frac{\sigma^2}{n}
	\end{aligned}
$$

Since the risk is bounded from above:

$$R(\mu, g_{\nu, m}) \leq \frac{\sigma^2}{n} + (\mu - \nu)^2,$$

by Lebesgue Dominated Convergence Theorem [^LDCT] we have

$$r_{\pi_{m}}=R(\pi_{m}, g_{\nu, m})=\int_{\mathbb{R}}R(\mu, g_{\nu, m})\pi_{m}(d\mu) \longrightarrow \frac{\sigma^2}{n}.$$

Since for estimator $g^*(x)=\overline{x}_n$ the equality

$$R(\mu, g^*)=\mathbb{E}[(\overline{X}_n-\mu)^2]=\frac{\sigma^2}{n},$$

holds, $g^*(x)$ is minimax and $\pi_{m}$ is sequence of least favorable priors.

---
   
[^LDCT]: Suppose there is measurable space $X$ with measure $\mu$. Also let $\lbrace f_n \rbrace_{n=1}^\infty$ and $f$ be measurable functions on $X$ and $f_n(x) \rightarrow f(x)$ almost everywhere. Then if there exists an integrable function $g$ defined on the same space such that

    $$|f_n(x)| \leq g(x) \quad \forall n \in \mathbb{N}$$ 

    almost everywhere, then $f_n$ and $f$ are integrable and 

    $$\lim\limits_{n \rightarrow \infty} \int_X f_n(x) \mu(dx) = \int_X f(x) \mu(dx).$$
