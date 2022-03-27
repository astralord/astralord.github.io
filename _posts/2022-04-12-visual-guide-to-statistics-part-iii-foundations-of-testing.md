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

### Introductory example

Let's discuss a simplified clinical study, in which we want to decide whether a newly invented drug $B$ is better than a well-known drug $A$ or not. Suppose that you know from previous years that $A$ has a chance of healing about $p_a$. The new drug $B$ was tested on $n$ persons and $m$ became healthy. Do we choose $A$ or $B$? In terms of mathematics we test

$$H\colon p_b \leq p_a \quad \text{vs} \quad K\colon p_b > p_a, $$

where $p_b$ is the unknown chance of healing with $B$.

Let $\Theta = \Theta_H \cup \Theta_K$ be a partition of $\Theta$.

* $\Theta_H$ is called **(null) hypothesis**, $\Theta_K$ is called the **alternative**.
* A **randomized test** is a measurable map $\varphi: \mathcal{X} \rightarrow [0, 1]$. Here $\varphi(x)$ is the probability of a decision for $\Theta_K$ when $x=X(\omega)$ is observed.
* For a test $\varphi$ we call $\mathcal{K}= \{x \mid \varphi(x)=1 \}$ the **critical region** and $\mathcal{R}= \{x \mid \varphi(x) \in (0,1) \}$ - the **region of randomization**. A test $\varphi$ is called **non-randomized** if $\mathcal{R} = \emptyset$.

In our example we know that the statistic $\overline{X}_n$ is the UMVU estimator for $p$. A reasonable decision rule is to decide for $K$ if $\overline{X}_n$ is "large". For example,

$$\varphi(x) =
	\left \{
	\begin{array}{cl}
	1, & \overline{X}_n > c, \\
	0, & \overline{X}_n \leq c 
	\end{array}
	\right.
$$

with some constant $c$ is a reasonable test. But how "large" must $c$ be? 

When deciding for $H$ or $K$ using $\varphi$, two errors can occur:

* **Error of the 1st kind**: decide for $K$ when $H$ is true.
* **Error of the 2nd kind**: decide for $H$ when $K$ is true.

Both errors occur with certain probabilities. In our example the probability of a decision for $K$ is

$$P(\varphi(X)=1)=P(\overline{X}_n > c).$$

In practice, we can use approximation by normal distribution

$$
\begin{aligned}
	P(\overline{X}_n > c) & = P\bigg(\frac{\sqrt{n}(\overline{X}_n - p_b)}{\sqrt{p_b(1-p_b)}} > \frac{\sqrt{n}(c - p_b)}{\sqrt{p_b(1-p_b)}}\bigg) \\
	\color{Salmon}{\text{Central Limit Theorem} \rightarrow} & \approx P\bigg(\mathcal{N}(0,1) > \frac{\sqrt{n}(c - p_b)}{\sqrt{p_b(1-p_b)}}\bigg) \\& = \Phi\bigg(\frac{\sqrt{n}(p_b - c)}{\sqrt{p_b(1-p_b)}}\bigg),
	\end{aligned}	
$$

where $\Phi$ is the distribution function of $\mathcal{N}(0, 1)$. The probability of error of the 1st kind is bounded from above:

$$
\begin{aligned}
P(\text{reject } H \mid H \text{ is true}) &= P(\overline{X}_n > c \mid p_b \leq p_a) \\ &\leq P(\overline{X}_n > c \mid p_b = p_a) \\ & =\Phi\bigg(\frac{\sqrt{n}(p_a - c)}{\sqrt{p_a(1-p_a)}}\bigg).
\end{aligned}	$$

By symmetry,

$$ P(\text{accept } H \mid K \text{ is true}) \leq 1 - \Phi\bigg(\frac{\sqrt{n}(p_a - c)}{\sqrt{p_a(1-p_a)}}\bigg).$$


<script src="https://d3js.org/d3.v4.min.js"></script>
<link href="https://fonts.googleapis.com/css?family=Arvo" rel="stylesheet">

<div id="basic_test"></div> 

<script>
function basic_test() {
var n = 100;
var c = 0.55;
var p_a = 0.5;

var margin = {top: 30, right: 0, bottom: 20, left: 30},
    width = 700 - margin.left - margin.right,
    height = 300 - margin.top - margin.bottom,
    fig_height = 200 - margin.top - margin.bottom,
    fig_width = 450;
    
var svg = d3.select("div#basic_test")
  .append("svg")
    .attr("width", width + margin.left + margin.right)
    .attr("height", height + margin.top + margin.bottom)
  .append("g")
    .attr("transform",
          "translate(" + margin.left + "," + margin.top + ")");

var x = d3.scaleLinear()
          .domain([0, 1])
          .range([0, fig_width]);
            
var xAxisTop = svg.append("g")
   .attr("transform", "translate(0," + fig_height + ")")
   .call(d3.axisBottom(x));
  
xAxisTop.selectAll(".tick text")
   .attr("font-family", "Arvo");
   
var xBtm = d3.scaleLinear()
          .domain([0, 1])
          .range([0, fig_width]);
            
var xAxisBtm = svg.append("g")
   .attr("transform", "translate(0," + 1.6 * fig_height + ")")
   .call(d3.axisBottom(xBtm));
  
xAxisBtm.selectAll(".tick text")
   .attr("font-family", "Arvo");

var y = d3.scaleLinear()
          .range([fig_height, 0])
          .domain([0, 1]);
            
var yAxisTop = svg.append("g")
    .call(d3.axisLeft(y).ticks(4));
  
yAxisTop.selectAll(".tick text")
    .attr("font-family", "Arvo");

var yBtm = d3.scaleLinear()
          .range([1.6 * fig_height, 1.3 * fig_height])
          .domain([0, 1]);
            
var yAxisBtm = svg.append("g")
    .call(d3.axisLeft(yBtm).ticks(1));
  
yAxisBtm.selectAll(".tick text")
    .attr("font-family", "Arvo");
    
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
    return 2 * sum / Math.sqrt(3.14159265358979);
}

function Phi(x) {
    return 0.5 * (1 + erf(x / Math.sqrt(2)));
}

var err1_data = [{'x': p_a, 'y': 0}];
for (var p = p_a; p > 0; p -= 0.001) {
  var pp = Math.sqrt(n / (p * (1 - p))) * (p - c);
  err1_data.push({'x': p, 'y': Phi(pp)});
}

var err2_data = [{'x': p_a, 'y': 0}];
for (var p = p_a; p < 1; p += 0.001) {
  var pp = Math.sqrt(n / (p * (1 - p))) * (p - c);
  err2_data.push({'x': p, 'y': 1 - Phi(pp)});
}

var err1_curve = svg
  .append('g')
  .append("path")
      .datum(err1_data)
      .attr("fill", "#65AD69")
      .attr("border", 0)
      .attr("opacity", ".8")
      .attr("stroke", "#000")
      .attr("stroke-width", 1)
      .attr("stroke-linejoin", "round")
      .attr("d",  d3.line()
        .curve(d3.curveBasis)
          .x(function(d) { return x(d['x']); })
          .y(function(d) { return y(d['y']); })
   );
  
var err2_curve = svg
  .append('g')
  .append("path")
      .datum(err2_data)
      .attr("fill", "#EDA137")
      .attr("border", 0)
      .attr("opacity", ".8")
      .attr("stroke", "#000")
      .attr("stroke-width", 1)
      .attr("stroke-linejoin", "round")
      .attr("d",  d3.line()
        .curve(d3.curveBasis)
          .x(function(d) { return x(d['x']); })
          .y(function(d) { return y(d['y']); })
   );
   
var phi_data_0 = [{'x': 0, 'y': 0}, {'x': c, 'y': 0}];
var phi_data_1 = [{'x': c, 'y': 1}, {'x': 1, 'y': 1}];
var phi_data_dash = [{'x': c, 'y': 0}, {'x': c, 'y': 1}];

var phi_curve_0 = svg
  .append('g')
  .append("path")
      .datum(phi_data_0)
      .attr("border", 1)
      .attr("opacity", "1")
      .attr("stroke", "#348ABD")
      .attr("stroke-width", 2.5)
      .attr("stroke-linejoin", "round")
      .attr("d",  d3.line()
          .x(function(d) { return xBtm(d['x']); })
          .y(function(d) { return yBtm(d['y']); })
   );
   
var phi_curve_1 = svg
  .append('g')
  .append("path")
      .datum(phi_data_1)
      .attr("border", 1)
      .attr("opacity", "1")
      .attr("stroke", "#348ABD")
      .attr("stroke-width", 2.5)
      .attr("stroke-linejoin", "round")
      .attr("d",  d3.line()
          .x(function(d) { return xBtm(d['x']); })
          .y(function(d) { return yBtm(d['y']); })
   );
   
var phi_curve_dash = svg
  .append('g')
  .append("path")
      .datum(phi_data_dash)
      .attr("border", 1)
      .attr("opacity", "1")
      .attr("stroke", "#348ABD")
      .attr("stroke-width", 1)
      .style("stroke-dasharray", ("3, 3"))
      .attr("stroke-linejoin", "round")
      .attr("d",  d3.line()
          .x(function(d) { return xBtm(d['x']); })
          .y(function(d) { return yBtm(d['y']); })
   );

var phi_dot = svg.append('g')
   .selectAll("dot")
   .data([{'x': xBtm(c), 'y': yBtm(1)}])
   .enter()
   .append("circle")
     .attr("cx", function (d) { return d.x; } )
     .attr("cy", function (d) { return d.y; } )
     .attr("r", 4)
     .style("fill", "#fff")
     .attr("stroke", "#348ABD")
     .attr("stroke-width", 2)
     .on("mouseover", function (d) {d3.select(this).style("cursor", "pointer");})
     .on("mouseout", function (d) {})
     .call(d3.drag()
       .on("drag", dragged_c)
     );

svg.append('g')
   .selectAll("dot")
   .data([{'x': x(p_a), 'y': y(0)}])
   .enter()
   .append("circle")
     .attr("cx", function (d) { return d.x; } )
     .attr("cy", function (d) { return d.y; } )
     .attr("r", 4)
     .style("fill", "#E86456")
     .attr("stroke", "#000")
     .attr("stroke-width", 1)
     .on("mouseover", function (d) {d3.select(this).style("cursor", "pointer");})
     .on("mouseout", function (d) {})
     .call(d3.drag()
       .on("drag", dragged_pa)
     );

function updatePhiLine() {
  var phi_data_0 = [{'x': 0, 'y': 0}, {'x': c, 'y': 0}];
  var phi_data_1 = [{'x': c, 'y': 1}, {'x': 1, 'y': 1}];
  var phi_data_dash = [{'x': c, 'y': 0}, {'x': c, 'y': 1}];
  
  phi_curve_0
      .datum(phi_data_0)
      .transition()
      .duration(0)
      .attr("d",  d3.line()
          .x(function(d) { return xBtm(d['x']); })
          .y(function(d) { return yBtm(d['y']); })
      );
      
  phi_curve_1
      .datum(phi_data_1)
      .transition()
      .duration(0)
      .attr("d",  d3.line()
          .x(function(d) { return xBtm(d['x']); })
          .y(function(d) { return yBtm(d['y']); })
      );
      
  phi_curve_dash
      .datum(phi_data_dash)
      .transition()
      .duration(0)
      .attr("d",  d3.line()
          .x(function(d) { return xBtm(d['x']); })
          .y(function(d) { return yBtm(d['y']); })
      );
}

function updateErrCurves() {
  var err1_data = [{'x': p_a, 'y': 0}];
  for (var p = p_a; p > 0; p -= 0.001) {
    var pp = Math.sqrt(n / (p * (1 - p))) * (p - c);
    err1_data.push({'x': p, 'y': Phi(pp)});
  }
  err1_data.push({'x': 0, 'y': 0});
  
  var err2_data = [{'x': p_a, 'y': 0}];
  for (var p = p_a; p < 1; p += 0.001) {
    var pp = Math.sqrt(n / (p * (1 - p))) * (p - c);
    err2_data.push({'x': p, 'y': 1 - Phi(pp)});
  }
  err2_data.push({'x': 1, 'y': 0});
  
  err1_curve
      .datum(err1_data)
      .transition()
      .duration(0)
      .attr("d",  d3.line()
        .curve(d3.curveBasis)
          .x(function(d) { return x(d['x']); })
          .y(function(d) { return y(d['y']); })
      );
  
  err2_curve
      .datum(err2_data)
      .transition()
      .duration(0)
      .attr("d",  d3.line()
        .curve(d3.curveBasis)
          .x(function(d) { return x(d['x']); })
          .y(function(d) { return y(d['y']); })
      );

}

function dragged_pa(d) {
  d3.select(this).attr("cx", d.x = Math.min(x(0.9999), 
                                   Math.max(d3.event.x, x(0.0001))));
  p_a = x.invert(d.x);
  updateErrCurves();
}

function dragged_c(d) {
  d3.select(this).attr("cx", d.x = Math.min(xBtm(0.9999), 
                                   Math.max(d3.event.x, xBtm(0.0001))));
  c = xBtm.invert(d.x);
  updatePhiLine();
  updateErrCurves();
}

var labels_x = 470;

svg.append("path")
   .attr("stroke", "#65AD69")
   .attr("stroke-width", 4)
   .attr("opacity", ".8")
   .datum([{x: labels_x, y: -5}, {x: labels_x + 25, y: -5}])
   .attr("d",  d3.line()
       .x(function(d) { return d.x; })
       .y(function(d) { return d.y; }));
       
svg.append("path")
   .attr("stroke", "#000")
   .attr("stroke-width", 1)
   .datum([{x: labels_x, y: -7}, {x: labels_x + 25, y: -7}])
   .attr("d",  d3.line()
       .x(function(d) { return d.x; })
       .y(function(d) { return d.y; }));
       
svg
  .append("text")
  .attr("text-anchor", "start")
  .attr("y", 0)
  .attr("x", labels_x + 30)
  .attr("font-family", "Arvo")
  .attr("font-weight", 700)
  .text("P(reject H | H is true)")
  .style("fill", "#65AD69");
  
svg.append("path")
   .attr("stroke", "#EDA137")
   .attr("stroke-width", 4)
   .attr("opacity", ".8")
   .datum([{x: labels_x, y: 15}, {x: labels_x + 25, y: 15}])
   .attr("d",  d3.line()
       .x(function(d) { return d.x; })
       .y(function(d) { return d.y; }));
       
svg.append("path")
   .attr("stroke", "#000")
   .attr("stroke-width", 1)
   .datum([{x: labels_x, y: 13}, {x: labels_x + 25, y: 13}])
   .attr("d",  d3.line()
       .x(function(d) { return d.x; })
       .y(function(d) { return d.y; }));
       
svg
  .append("text")
  .attr("text-anchor", "start")
  .attr("y", 20)
  .attr("x", labels_x + 30)
  .attr("font-family", "Arvo")
  .attr("font-weight", 700)
  .text("P(accept H | K is true)")
  .style("fill", "#EDA137");
 
svg.append('g')
   .selectAll("dot")
   .data([{'x': labels_x + 14, 'y': 35}])
   .enter()
   .append("circle")
     .attr("cx", function (d) { return d.x; } )
     .attr("cy", function (d) { return d.y; } )
     .attr("r", 4)
     .style("fill", "#E86456")
     .attr("stroke", "#000")
     .attr("stroke-width", 1);
       
svg
  .append("text")
  .attr("text-anchor", "start")
  .attr("y", 40)
  .attr("x", labels_x + 30)
  .attr("font-family", "Arvo")
  .attr("font-weight", 700)
  .text("pₐ")
  .style("fill", "#E86456");
       
svg.append("path")
   .attr("stroke", "#348ABD")
   .attr("stroke-width", 3)
   .attr("opacity", "1")
   .datum([{x: labels_x + 5, y: 55}, {x: labels_x + 25, y: 55}])
   .attr("d",  d3.line()
       .x(function(d) { return d.x; })
       .y(function(d) { return d.y; }));
 
svg.append('g')
   .selectAll("dot")
   .data([{'x': labels_x + 7, 'y': 55}])
   .enter()
   .append("circle")
     .attr("cx", function (d) { return d.x; } )
     .attr("cy", function (d) { return d.y; } )
     .attr("r", 4)
     .style("fill", "#fff")
     .attr("stroke", "#348ABD")
     .attr("stroke-width", 2);
       
svg
  .append("text")
  .attr("text-anchor", "start")
  .attr("y", 60)
  .attr("x", labels_x + 30)
  .attr("font-family", "Arvo")
  .attr("font-weight", 700)
  .text("φ(x)")
  .style("fill", "#348ABD");
  
svg
  .append("text")
  .attr("text-anchor", "start")
  .attr("y", fig_height + 5)
  .attr("x", labels_x - 5)
  .attr("font-family", "Arvo")
  .text("p")
  .style("fill", "#000")
  .append('tspan')
    .text('b')
    .style('font-size', '.5rem')
    .attr('dx', '-.1em')
    .attr('dy', '.8em');
    
svg
  .append("text")
  .attr("text-anchor", "start")
  .attr("y", 1.6 * fig_height + 5)
  .attr("x", labels_x - 5)
  .attr("font-family", "Arvo")
  .text("x")
  .style("fill", "#000");
}


basic_test();

</script>

![](.)
*Fig. 1. Visualization of basic test experiment. Parameters $p_a$ and $c$ are draggable.*

### Power of a test

Ideally we want to minimize both errors simulaneously and pick the optimal test. The problem is that criterias $\varphi_0(x) \equiv 0$ and $\varphi_1(x) \equiv 1$ are optimal if one needs to minimize one of the errors, but they don't minimize both errors at the same time. In practice, the upper bound $\alpha$ is taken for the probability of error of the 1st kind and probability of error of the 2nd kind is minimized. Typically, $0.01 \leq \alpha \leq 0.1$ (the set belonging to the more severe consequences is the alternative).

Now suppose $\varphi$ is a test for $H \colon \vartheta \in \Theta_H$ vs $K \colon \vartheta \in \Theta_K$. Let's define function

$$\beta_\varphi(\vartheta) = 1 - \mathbb{E}_\vartheta[\varphi(X)].$$

The function $1 - \beta_\varphi(\vartheta)$ is called **power of a test** $\varphi$. Note that for non-randomized test $\varphi$ we have

$$1-\beta_\varphi(\vartheta) = P_\vartheta(\varphi(x) = 1),$$

which is the probability to decide for $K$. In particular, 

* $\vartheta \in \Theta_H$: $1 - \beta_\varphi(\vartheta)$ is the probability of an error of the 1st kind,
* $\vartheta \in \Theta_K$: $\beta_\varphi(\vartheta)$ is the probability of an error of the 2nd kind.

The same intuition holds for randomized tests. Test $\varphi$ is called a test with **significance level** $\alpha \in [0, 1]$ if 

$$1 - \beta_\varphi(\vartheta) \leq \alpha \quad \forall \vartheta \in \Theta_H.$$

A test with significance level $\alpha$ has a probability of an error of the 1st kind, which is bounded by $\alpha$. We will denote set of all tests with significance level $\alpha$ as $\Phi_\alpha$. Test $\varphi$ is also called **unbiased with significance level** $\alpha$ if $\varphi \in \Phi_\alpha$ and 

$$1-\beta_\varphi(\vartheta) \geq \alpha \quad \forall \vartheta \in \Theta_K.$$

For an unbiased test with significance level $\alpha$ the probability of deciding for $K$ for every $\vartheta \in \Theta_K$ is not smaller than for $\vartheta \in \Theta_H$. The set of all unbiased tests with level $\alpha$ we will call $\Phi_{\alpha \alpha}$.

Test $\varphi^* \in \Phi_\alpha$ is called **uniformly most powerful (UMP)** test with significance level $\alpha$ if

$$\beta_{\varphi^*}(\vartheta) = \inf_{\varphi \in \Phi_\alpha} \beta_\varphi(\vartheta) \quad \forall \vartheta \in \Theta_K.$$

Test $\varphi^* \in \Phi_{\alpha\alpha}$ is called **uniformly most powerful unbiased (UMPU)** test with significance level $\alpha$ if

$$\beta_{\varphi^*}(\vartheta) = \inf_{\varphi \in \Phi_{\alpha\alpha}} \beta_\varphi(\vartheta) \quad \forall \vartheta \in \Theta_K.$$

### Neyman-Pearson lemma

Let's consider *simple hypothesis*:

$$H\colon \vartheta \in \lbrace \vartheta_0 \rbrace \ \ \text{vs} \ \ K\colon \vartheta \in \lbrace \vartheta_1 \rbrace , \quad \vartheta_0 \neq \vartheta_1.$$

Corresponding densities: $p_i = \frac{dP_{\vartheta_i}}{dx}$. UMP-test with level $\alpha$ maximizes

$$1-\beta_\varphi(\vartheta_1) = \mathbb{E}_{\vartheta_1}[\varphi(X)] = \int_{\mathcal{X}} \varphi(x)p_1(x)dx$$

under the constraint 

$$1-\beta_\varphi(\vartheta_0) = \mathbb{E}_{\vartheta_0}[\varphi(X)] = \int_{\mathcal{X}} \varphi(x)p_0(x)dx \leq \alpha.$$

In the situation of simple hypotheses a test $\varphi$ is called **a Neyman-Pearson test (NP test)** if $c\in[0, \infty)$ exists such that

$$\varphi(x):
	\left \{
	\begin{array}{cl}
	1, & p_1(x) > cp_0(x), \\
	0, & p_1(x) < cp_0(x).
	\end{array}
	\right.$$

Let $\varphi^*$ be an NP-test with constant $c^*$ and let $\varphi$ be some other test with $\beta_\varphi(\vartheta_0) \geq \beta_{\varphi^*}(\vartheta_0)$. Then we have

$$\begin{aligned} \beta_\varphi(\vartheta_1) - \beta_{\varphi^*}(\vartheta_1) &= (1 - \beta_{\varphi^*}(\vartheta_1) ) - (1 - \beta_\varphi(\vartheta_1) ) \\&=\int (\varphi^* - \varphi) p_1 dx \\&= \int (\varphi^* - \varphi)(p_1 - c^*p_0)dx + \int c^* p_0 (\varphi^* - \varphi) dx.
\end{aligned}$$

For the first integral note that

$$\varphi^* - \varphi > 0 \Longrightarrow \varphi^* > 0 \Longrightarrow p_1 \geq c^*p_0, \\
\varphi^* - \varphi < 0 \Longrightarrow \varphi^* < 1 \Longrightarrow p_1 \leq c^*p_0. $$

$\Longrightarrow (\varphi^* - \varphi)(p_1 - c^*p_0) \geq 0$ always. The second integral is $c^*(\beta_{\varphi^*}(\vartheta_0) - \beta_\varphi(\vartheta_0)) \geq 0$. 

Therefore we have $\beta_\varphi(\vartheta_1) \geq \beta_{\varphi^*}(\vartheta_1)$ and NP-test $\varphi^*$ is an UMP test with level $\alpha = \mathbb{E}_{\vartheta_0}[\varphi^*(X)]$. This statement is called **NP lemma**.

There are also other parts of this lemma which I will state here without proof:

* For any $\alpha \in [0, 1]$ there is an NP-test $\varphi$ with $\mathbb{E}_{\vartheta_0}[\varphi(X)] = \alpha$.
* If $\varphi'$ is UMP with level $\alpha$, then $\varphi'$ is (a.s.) an NP-test. If $\mathbb{E}_{\vartheta_0}[\varphi'(X)] < \alpha$, then $\mathbb{E}_{\vartheta_1}[\varphi'(X)]=1$.

An NP-test $\varphi^*$ for $H \colon \vartheta = \vartheta_0$ vs $K \colon \vartheta = \vartheta_1$ is uniquely defined outside of $S_= =\{x\ |\ p_1(x) = c^*p_0(x) \}$. On $S_=$ set the test can be chosen such that $\beta_{\varphi^*}(\vartheta_0) = \alpha$.

Is must also be noted that every NP-test $\varphi^*$ with $\beta_{\varphi^*}(\vartheta_0) \in (0, 1)$ is unbiased. In particular

$$\alpha := 1 - \beta_{\varphi^*}(\vartheta_0) < 1 - \beta_{\varphi^*}(\vartheta_1).$$

<details>
<summary>Proof</summary>
Take test $\varphi \equiv \alpha$. It has significance level $\alpha$ and since $\varphi^*$ is UMP, we have $1-\beta_\varphi(\vartheta_1) \leq 1-\beta_{\varphi^*}(\vartheta_1)$. If $\alpha = 1-\beta_{\varphi^*}(\vartheta_1) < 1$, then $\varphi \equiv \alpha$ is UMP. Since every UMP test is an NP test, we know that $p_1(x) = c^*p_0(x)$ for almost all $x$. Therefore, $c^*=1$ and $p_1 = p_0$ a.s. and also $P_{\vartheta_0} = P_{\vartheta_1}$, which is contradictory.
</details>

Let $X_1, \dots X_n$ i.i.d. $\sim \mathcal{N}(\mu,\sigma^2)$ with $\sigma^2$ known. We test

$$H \colon \mu = \mu_0 \quad \text{vs} \quad K \colon \mu = \mu_1$$

with $\mu_0 < \mu_1$. For the density of $X_1, \dots X_n$ it holds

$$p_j(x) = (2 \pi \sigma^2)^{-n/2} \exp \Big \{ -\frac{1}{2\sigma^2} \Big( \sum_{i=1}^{n} X_i^2 - 2 \mu_j \sum_{i=1}^{n}X_i + n\mu_j^2  \Big)\Big \}, \quad j = 0, 1.$$

As the inequality for the likelihood ratio which we need for the construction of the NP test, we get

$$\frac{p_1(x)}{p_0(x)} = \exp \Big \{ \frac{1}{\sigma^2} \sum_{i=1}^{n} x_i(\mu_1 - \mu_0) \Big \} \cdot f(\sigma^2, \mu_1, \mu_0) > c^*,$$

where the known constant $f(\sigma^2, \mu_1, \mu_0)$ is positive. This inequality is equivalent to

$$\overline{X}_n = \frac{1}{n} \sum_{i=1}^{n}X_i > c,$$

for some appropriate $c$ (because of $\mu_1 > \mu_0$). Therefore it is equally well possible to determine $c$ such that

$$P_{\mu_0}(\overline{X}_n > c) = \alpha$$

or equivalently

$$\begin{aligned}
	P_{\mu_0}\Big( &\underbrace{\frac{\sqrt{n}(\overline{X}_n - \mu_0)}{\sigma}} > \frac{\sqrt{n}(c-\mu_0)}{\sigma}\Big) = 1 - \Phi\Big(\frac{\sqrt{n}(c - \mu_0)}{\sigma}\Big) = \alpha. \\
	&\quad \sim \mathcal{N}(0, 1)
	\end{aligned}$$

If we call $u_p$ the **p-quantile** of $\mathcal{N}(0, 1)$, which is the value such that $\Phi(u_q)=q$, then we get

$$\frac{\sqrt{n}(c - \mu_0)}{\sigma} = u_{1-\alpha} \quad \Longleftrightarrow \quad c = \mu_0 + u_{1-\alpha}\frac{\sigma}{\sqrt{n}}.$$

The NP-test becomes

$$\varphi^*(x) = 1_{\{\overline{X}_n > \mu_0 + u_{1-\alpha} \frac{\sigma}{\sqrt{n}}  \} }.$$




TODO: p-value