---
layout: post
title: 'Visual Guide to Statistics. Part III: Foundations of Testing'
date: 2022-04-12 03:13 +0800
categories: [Statistics]
tags: [statistics, hypothesis, significance-level, power-of-a-test, neyman-pearson-test, ump-test, confidence-interval, monotone-likelihood-ratio, one-tailed-gauss-test, one-tailed-t-test]
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
* For a test $\varphi$ we call $\mathcal{K}= \lbrace x \mid \varphi(x)=1 \rbrace$ the **critical region** and $\mathcal{R}= \lbrace x \mid \varphi(x) \in (0,1) \rbrace$ - the **region of randomization**. A test $\varphi$ is called **non-randomized** if $\mathcal{R} = \emptyset$.

In our example we know that the statistic $\overline{X}_n$ is the UMVU estimator for $p$. A reasonable decision rule is to decide for $K$ if $\overline{X}_n$ is "large". For example,

$$\varphi(x) =
	\left \lbrace
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

#sample-button-h {
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

#sample-button-h:hover {
  background-color: #696969;
  
}#sample-button-k {
  top: 15px;
  left: 15px;
  background: #EDA137;
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

#sample-button-k:hover {
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

#n-text {
  font-family: Arvo;
  font-size: 11px;
}

#n-num {
  font-family: Arvo;
  font-size: 11px;
}
  		
}

</style>

<div id="basic_test"></div> 

<script>

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

function erfinv(x){
    var z;
    var a = 0.147;                                                   
    var the_sign_of_x;
    if (0 == x) {
        the_sign_of_x = 0;
    } else if (x > 0) {
        the_sign_of_x = 1;
    } else {
        the_sign_of_x = -1;
    }

    if (0 != x) {
        var ln_1minus_x_sqrd = Math.log(1 - x * x);
        var ln_1minusxx_by_a = ln_1minus_x_sqrd / a;
        var ln_1minusxx_by_2 = ln_1minus_x_sqrd / 2;
        var ln_etc_by2_plus2 = ln_1minusxx_by_2 + (2/(Math.PI * a));
        var first_sqrt = Math.sqrt((ln_etc_by2_plus2 * ln_etc_by2_plus2) - ln_1minusxx_by_a);
        var second_sqrt = Math.sqrt(first_sqrt - ln_etc_by2_plus2);
        z = second_sqrt * the_sign_of_x;
    } else {
        z = 0;
    }
    return z;
}

function PhiInv(y) {
    return 1.41421356237 * erfinv(2 * y - 1);
}

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

var labels_x = 450;
var labels_y = 0;
var labels_v = 18;

svg.append("path")
   .attr("stroke", "#65AD69")
   .attr("stroke-width", 4)
   .attr("opacity", ".8")
   .datum([{x: labels_x, y: labels_y}, {x: labels_x + 25, y: labels_y}])
   .attr("d",  d3.line()
       .x(function(d) { return d.x; })
       .y(function(d) { return d.y; }));
       
svg.append("path")
   .attr("stroke", "#000")
   .attr("stroke-width", 1)
   .datum([{x: labels_x, y: labels_y - 2}, {x: labels_x + 25, y: labels_y - 2}])
   .attr("d",  d3.line()
       .x(function(d) { return d.x; })
       .y(function(d) { return d.y; }));
       
svg
  .append("text")
  .attr("text-anchor", "start")
  .attr("y", labels_y + 5)
  .attr("x", labels_x + 30)
  .attr("font-family", "Arvo")
  .attr("font-weight", 700)
  .text("P( reject H | H is true)")
  .style("fill", "#65AD69");
  
svg.append("path")
   .attr("stroke", "#EDA137")
   .attr("stroke-width", 4)
   .attr("opacity", ".8")
   .datum([{x: labels_x, y: labels_y + labels_v}, {x: labels_x + 25, y: labels_y + labels_v}])
   .attr("d",  d3.line()
       .x(function(d) { return d.x; })
       .y(function(d) { return d.y; }));
       
svg.append("path")
   .attr("stroke", "#000")
   .attr("stroke-width", 1)
   .datum([{x: labels_x, y: labels_y + labels_v - 2}, {x: labels_x + 25, y: labels_y + labels_v - 2}])
   .attr("d",  d3.line()
       .x(function(d) { return d.x; })
       .y(function(d) { return d.y; }));
       
svg
  .append("text")
  .attr("text-anchor", "start")
  .attr("y", labels_y + labels_v + 5)
  .attr("x", labels_x + 30)
  .attr("font-family", "Arvo")
  .attr("font-weight", 700)
  .text("P(accept H | K is true)")
  .style("fill", "#EDA137");
 
svg.append('g')
   .selectAll("dot")
   .data([{'x': labels_x + 14, 'y': labels_y + 2 * labels_v - 3}])
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
  .attr("y", labels_y + 2 * labels_v)
  .attr("x", labels_x + 30)
  .attr("font-family", "Arvo")
  .attr("font-weight", 700)
  .text("pₐ")
  .style("fill", "#E86456");
       
svg.append("path")
   .attr("stroke", "#348ABD")
   .attr("stroke-width", 3)
   .attr("opacity", "1")
   .datum([{x: labels_x + 5, y: labels_y + 3 * labels_v - 5}, {x: labels_x + 25, y: labels_y + 3 * labels_v - 5}])
   .attr("d",  d3.line()
       .x(function(d) { return d.x; })
       .y(function(d) { return d.y; }));
 
svg.append('g')
   .selectAll("dot")
   .data([{'x': labels_x + 7, 'y': labels_y + 3 * labels_v - 5}])
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
  .attr("y", labels_y + 3 * labels_v)
  .attr("x", labels_x + 30)
  .attr("font-family", "Arvo")
  .attr("font-weight", 700)
  .text("φ(x)")
  .style("fill", "#348ABD");
  
svg
  .append("text")
  .attr("text-anchor", "start")
  .attr("y", fig_height + 5)
  .attr("x", labels_x + 15)
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
  .attr("x", labels_x + 15)
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

Note that for non-randomized test $\varphi$ we have

$$\beta_\varphi(\vartheta) = P_\vartheta(\varphi(x) = 0),$$

which is the probability to decide for $H$. In particular, 

* $\vartheta \in \Theta_H$: $1 - \beta_\varphi(\vartheta)$ is the probability of an error of the 1st kind,
* $\vartheta \in \Theta_K$: $\beta_\varphi(\vartheta)$ is the probability of an error of the 2nd kind.

The function $1 - \beta_\varphi(\vartheta)$ for $\vartheta \in \Theta_K$, which is the probability of correctly rejecting hypothesis $H$, when alterntative $K$ is true, is called **power of a test** $\varphi$. The same intuition holds for randomized tests. Test $\varphi$ is called a test with **significance level** $\alpha \in [0, 1]$ if 

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
	\left \lbrace
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

An NP-test $\varphi^*$ for $H \colon \vartheta = \vartheta_0$ vs $K \colon \vartheta = \vartheta_1$ is uniquely defined outside of $S_= =\lbrace x\ |\ p_1(x) = c^*p_0(x) \rbrace$. On $S_=$ set the test can be chosen such that $\beta_{\varphi^*}(\vartheta_0) = \alpha$.

Is must also be noted that every NP-test $\varphi^*$ with $\beta_{\varphi^*}(\vartheta_0) \in (0, 1)$ is unbiased. In particular

$$\alpha := 1 - \beta_{\varphi^*}(\vartheta_0) < 1 - \beta_{\varphi^*}(\vartheta_1).$$

<details>
<summary>Proof</summary>
Take test $\varphi \equiv \alpha$. It has significance level $\alpha$ and since $\varphi^*$ is UMP, we have $1-\beta_\varphi(\vartheta_1) \leq 1-\beta_{\varphi^*}(\vartheta_1)$. If $\alpha = 1-\beta_{\varphi^*}(\vartheta_1) < 1$, then $\varphi \equiv \alpha$ is UMP. Since every UMP test is an NP test, we know that $p_1(x) = c^*p_0(x)$ for almost all $x$. Therefore, $c^*=1$ and $p_1 = p_0$ a.s. and also $P_{\vartheta_0} = P_{\vartheta_1}$, which is contradictory.
</details>

### Confidence interval

Let $X_1, \dots X_n$ i.i.d. $\sim \mathcal{N}(\mu,\sigma^2)$ with $\sigma^2$ known. We test

$$H \colon \mu = \mu_0 \quad \text{vs} \quad K \colon \mu = \mu_1$$

with $\mu_0 < \mu_1$. For the density of $X_1, \dots X_n$ it holds

$$p_j(x) = (2 \pi \sigma^2)^{-n/2} \exp \Big( -\frac{1}{2\sigma^2} \Big( \sum_{i=1}^{n} X_i^2 - 2 \mu_j \sum_{i=1}^{n}X_i + n\mu_j^2  \Big)\Big), \quad j = 0, 1.$$

As the inequality for the likelihood ratio which we need for the construction of the NP test, we get

$$\frac{p_1(x)}{p_0(x)} = \exp \Big( \frac{1}{\sigma^2} \sum_{i=1}^{n} x_i(\mu_1 - \mu_0) \Big) \cdot f(\sigma^2, \mu_1, \mu_0) > c^*,$$

where the known constant $f(\sigma^2, \mu_1, \mu_0)$ is positive. This inequality is equivalent to

$$\overline{X}_n = \frac{1}{n} \sum_{i=1}^{n}X_i > c,$$

for some appropriate $c$ (because of $\mu_1 > \mu_0$). Therefore it is equally well possible to determine $c$ such that

$$P_{\mu_0}(\overline{X}_n > c) = \alpha$$

or equivalently

$$\begin{aligned}
	P_{\mu_0}\Big( &\underbrace{\frac{\sqrt{n}(\overline{X}_n - \mu_0)}{\sigma}} > \frac{\sqrt{n}(c-\mu_0)}{\sigma}\Big) = 1 - \Phi\Big(\frac{\sqrt{n}(c - \mu_0)}{\sigma}\Big) = \alpha. \\
	&\quad \sim \mathcal{N}(0, 1)
	\end{aligned}$$

If we call $u_p$ the **p-quantile** of $\mathcal{N}(0, 1)$, which is the value such that $\Phi(u_p)=p$, then we get

$$\frac{\sqrt{n}(c - \mu_0)}{\sigma} = u_{1-\alpha} \quad \Longleftrightarrow \quad c = \mu_0 + u_{1-\alpha}\frac{\sigma}{\sqrt{n}}.$$

The NP-test becomes

$$\varphi^*(x) = 1_{\lbrace\overline{X}_n > \mu_0 + u_{1-\alpha} \frac{\sigma}{\sqrt{n}}  \rbrace }.$$



<button id="sample-button-h">Sample H</button> <button id="sample-button-k">Sample K</button> <label id="n-text">n:</label><input type="number" min="1" max="100" step="1" value="10" id="n-num"> <button id="reset-button">Reset</button>


 
<div id="simple_hypothesis"></div> 

<script>
  
function randn_bm() {
    var u = 0, v = 0;
    while(u === 0) u = Math.random();
    while(v === 0) v = Math.random();
    return Math.sqrt( -2.0 * Math.log( u ) ) * Math.cos( 2.0 * Math.PI * v );
}

function gauss_data(mu, sigma) {
  var data = [{x: -4, y: 0}];
  for (var i = -4; i < 4; i += 0.01) {
      data.push({x: i, y: Math.exp(-0.5 * ((i - mu) / sigma) ** 2) / (sigma * Math.sqrt(2 * Math.PI)) });
  }
  data.push({x: 4, y: 0});
  return data;
}

function quantile_data() {
  var data = [{x: 0, y: 0}];
  for (var i = 0; i < 3.5; i += 0.01) {
      data.push({x: i, y: 1 - Phi(i) });
  }
  data.push({x: 3.5, y: 0});
  return data;
}

    
function createSlider(svg_, parameter_update, x, loc_x, loc_y, letter, color, init_val, round_fun) {
    var slider = svg_.append("g")
      .attr("class", "slider")
      .attr("transform", "translate(" + loc_x + "," + loc_y + ")");
    
    var drag = d3.drag()
	        .on("start.interrupt", function() { slider.interrupt(); })
	        .on("start drag", function() { 
	          handle.attr("cx", x(round_fun(x.invert(d3.event.x))));  
	          parameter_update(x.invert(d3.event.x));	         });
	         
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

function simple_hypothesis() {

var mu0 = -1,
    mu1 = 1,
    sigma = 1,
    alpha = 0.05,
    n = 10;

var u_q = PhiInv(1 - alpha);
var power = 1 - Phi(Math.sqrt(n) * (mu0 - mu1) / sigma + u_q);

var margin = {top: 30, right: 0, bottom: 20, left: 30},
    width = 750 - margin.left - margin.right,
    height = 500 - margin.top - margin.bottom,
    fig_height = 200 - margin.top - margin.bottom,
    fig_width = 350;
    
var svg = d3.select("div#simple_hypothesis")
  .append("svg")
    .attr("width", width + margin.left + margin.right)
    .attr("height", height + margin.top + margin.bottom)
  .append("g")
    .attr("transform",
          "translate(" + margin.left + "," + margin.top + ")");

var x = d3.scaleLinear()
          .domain([-4, 4])
          .range([0, fig_width]);
            
var xAxisTop = svg.append("g")
   .attr("transform", "translate(0," + fig_height + ")")
   .call(d3.axisBottom(x));
  
xAxisTop.selectAll(".tick text")
   .attr("font-family", "Arvo");

   
var xBtm = d3.scaleLinear()
          .domain([-4, 4])
          .range([0, fig_width]);
            
var xAxisBtm = svg.append("g")
   .attr("transform", "translate(0," + 1.6 * fig_height + ")")
   .call(d3.axisBottom(xBtm));
  
xAxisBtm.selectAll(".tick text")
   .attr("font-family", "Arvo");


var xRight = d3.scaleLinear()
          .domain([0, 3.5])
          .range([1.2 * fig_width, 1.9 * fig_width]);
          
var xAxisRight = svg.append("g")
   .attr("transform", "translate(0," + 2 * fig_height + ")")
   .call(d3.axisBottom(xRight).ticks(5));
  
xAxisRight.selectAll(".tick text")
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
            

var yRight = d3.scaleLinear()
          .range([2 * fig_height, 0])
          .domain([0, 0.5]);
            
var yAxisRight = svg.append("g")
   .attr("transform", "translate(" + 1.2 * fig_width + ",0)")
    .call(d3.axisLeft(yRight).ticks(4));
  
yAxisRight.selectAll(".tick text")
    .attr("font-family", "Arvo");
    
    
    
var mu0_data = gauss_data(mu0, sigma);

var mu0_curve = svg
  .append('g')
  .append("path")
      .datum(mu0_data)
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
   
var mu1_data = gauss_data(mu1, sigma);

var mu1_curve = svg
  .append('g')
  .append("path")
      .datum(mu1_data)
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

var q_data = quantile_data();
var quantile_curve = svg
  .append('g')
  .append("path")
      .datum(q_data)
      .attr("fill", "#348ABD")
      .attr("border", 0)
      .attr("opacity", ".8")
      .attr("stroke", "#000")
      .attr("stroke-width", 1)
      .attr("stroke-linejoin", "round")
      .attr("d",  d3.line()
        .curve(d3.curveBasis)
          .x(function(d) { return xRight(d.x); })
          .y(function(d) { return yRight(d.y); })
   );

function updatePower() {
  var rc = 1000;
  if (power > 0.999) {
      rc = 10000;
  }
  power = 1 - Phi(Math.sqrt(n) * (mu0 - mu1) / sigma + u_q);
  power_text
    .transition()
    .duration(500)
    .text('Power: ' + Math.round(rc * power) / rc);
}

function dragged_u(d) {
  var u_x = Math.min(xRight(3.5), Math.max(d3.event.x, xRight(0)));
  u_q = xRight.invert(u_x);
  alpha = 1 - Phi(u_q);
  var u_y = yRight(alpha);
  d3.select(this).attr("cx", d.x = u_x).attr("cy", d.y = u_y);
  updatePhiLine();
  var rc = 1000;
  if (alpha < 0.001) {
    rc = 10000;
  }
  alpha_text
    .transition()
    .duration(500)
    .text('Significance level: ' + Math.round(rc * alpha) / rc);
  updatePower();
}

var u_dot = svg.append('g')
   .selectAll("dot")
   .data([{'x': xRight(u_q), 'y': yRight(alpha)}])
   .enter()
   .append("circle")
     .attr("cx", function (d) { return d.x; } )
     .attr("cy", function (d) { return d.y; } )
     .attr("r", 4)
     .style("fill", "#fff")
     .attr("stroke", "#348ABD")
     .attr("stroke-width", 2)
     .on("mouseover", function(d) { d3.select(this)
                                      .style("cursor", "pointer");})
     .on("mousemove", function (d) {})
     .call(d3.drag()
       .on("drag", dragged_u)
     );
     
function updatePhiLine() {
  reset();
  c = mu0 + u_q * sigma / Math.sqrt(n);
  var phi_data_0 = [{'x': -4, 'y': 0}, {'x': c, 'y': 0}];
  var phi_data_1 = [{'x': c, 'y': 1}, {'x': 4, 'y': 1}];
  var phi_data_dash = [{'x': c, 'y': 0}, {'x': c, 'y': 1}];
      
  phi_dot
      .transition()
      .duration(0)
      .attr("cx", xBtm(c) );
       
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

var c = mu0 + u_q * sigma / Math.sqrt(n);
var phi_data_0 = [{'x': -4, 'y': 0}, {'x': c, 'y': 0}];
var phi_data_1 = [{'x': c, 'y': 1}, {'x': 4, 'y': 1}];
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
     .attr("stroke-width", 2);
     
function updateCurves(delay) {
    mu0_data = gauss_data(mu0, sigma);
    mu1_data = gauss_data(mu1, sigma);
    
    mu0_curve
      .datum(mu0_data)
      .transition()
      .delay(delay) 
      .duration(0)
      .attr("d",  d3.line()
        .curve(d3.curveBasis)
        .x(function(d) { return x(d.x); })
        .y(function(d) { return y(d.y); })
      );
      
    mu1_curve
      .datum(mu1_data)
      .transition()
      .delay(delay) 
      .duration(0)
      .attr("d",  d3.line()
        .curve(d3.curveBasis)
        .x(function(d) { return x(d.x); })
        .y(function(d) { return y(d.y); })
      );
}

function updateSigma(x) {
    reset();
    sigma = x;
    updatePower();
    updateCurves(0); 
    updatePhiLine();
}

function trivialRound(x) { return x; }

var sigma_x = d3.scaleLinear()
    .domain([0.4, 1.4])
    .range([0, width * 0.4])
    .clamp(true);
    
createSlider(svg, updateSigma, sigma_x, margin.left, 2 * fig_height, "σ", "#A9A750", sigma, trivialRound);

d3.select("#n-num").on("input", function() {
    n = this.value;
    updatePhiLine();
    updatePower();
});

var table_nums = [0, 0, 0, 0];

function updateTableText() {

  table_text_hh
    .transition()
    .duration(500)
    .text(table_nums[0]);

  table_text_hk
    .transition()
    .duration(500)
    .text(table_nums[1]);

  table_text_kh
    .transition()
    .duration(500)
    .text(table_nums[2]);

  table_text_kk
    .transition()
    .duration(500)
    .text(table_nums[3]);
}

var avg_dots = [];
function sampleGauss(mu, color) {
      var avg_dur = 1200;
      var random_samples = [];
      var smpl_dots = [];
      var avg = 0;
      for (var i = 0; i < n; i += 1) {
          random_samples.push(mu + sigma * randn_bm());
          smpl_dots.push(svg.append('g')
            .selectAll("dot")
            .data([{x: random_samples[i], y: 1}])
            .enter()
            .append("circle")
              .attr("cx", function (d) { return x(d.x); } )
              .attr("cy", function (d) { return y(d.y); } )
              .attr("r", 3)
              .style("fill", color)
              .attr("stroke", "#000")
              .attr("stroke-width", 1));
          
          smpl_dots[i].transition()
            .duration(avg_dur)
            .attr("cx", function (d) { return x(random_samples[i]); } )
            .attr("cy", function (d) { return y(0); } );
      
          avg += random_samples[i];
      }
      avg /= n;  
      
      for (var i = 0; i < n; i += 1) {
          smpl_dots[i]
            .transition()
            .delay(avg_dur) 
            .duration(avg_dur)
            .attr("cx", function (d) { return x(avg); } )
            .attr("cy", function (d) { return y(0); } );
            
          if (i > 0) {
            smpl_dots[i].transition().delay(2 * avg_dur).remove();
          }
          else {
            avg_dots.push(smpl_dots[0]);
          }
      }
      
      if (avg > c) {
          smpl_dots[0]
            .transition()
            .delay(2 * avg_dur) 
            .duration(avg_dur)
            .attr("cy", function (d) { return yBtm(1); } );
            
         if (mu == mu0) {
           table_nums[2] += 1;
         }
         else {
           table_nums[3] += 1;
         }
      }
      else {
          smpl_dots[0]
            .transition()
            .delay(2 * avg_dur) 
            .duration(avg_dur)
            .attr("cy", function (d) { return yBtm(0); } );
            
         if (mu == mu0) {
           table_nums[0] += 1;
         }
         else {
           table_nums[1] += 1;
         }
      }
      
      updateTableText();
}

d3.select("#sample-button-h").on("click", function() {sampleGauss(mu0, "#65AD69")});
d3.select("#sample-button-k").on("click", function() {sampleGauss(mu1, "#EDA137")});

function reset() {
  for (var i = 0; i < avg_dots.length; i += 1) {
      avg_dots[i].remove();
  }
  avg_dots = [];
  table_nums = [0, 0, 0, 0];
  updateTableText();
}

d3.select("#reset-button").on("click", function() { reset(); });

var alpha_text = svg
  .append("text")
  .attr("text-anchor", "start")
  .attr("y", 2.6 * fig_height)
  .attr("x", 1.2 * fig_width + 5)
  .attr("font-family", "Arvo")
  .text("Significance level: " + alpha)
  .style("fill", "#348ABD");
  
var power_text = svg
  .append("text")
  .attr("text-anchor", "start")
  .attr("y", 2.9 * fig_height)
  .attr("x", 1.2 * fig_width + 5)
  .attr("font-family", "Arvo")
  .text("Power: " + power)
  .style("fill", "#348ABD");

svg
  .append("text")
  .attr("text-anchor", "start")
  .attr("y", 0)
  .attr("x", 1.2 * fig_width + 5)
  .attr("font-family", "Arvo")
  .text("α")
  .style("fill", "#000");
  
svg
  .append("text")
  .attr("text-anchor", "start")
  .attr("y", 2 * fig_height)
  .attr("x", 1.9 * fig_width + 5)
  .attr("font-family", "Arvo")
  .text("u")
  .style("fill", "#000")
  .append('tspan')
    .text('1-α')
    .style('font-size', '.5rem')
    .attr('dx', '-.1em')
    .attr('dy', '.8em');

var labels_x = 250;
var labels_y = 0;

svg.append("path")
   .attr("stroke", "#65AD69")
   .attr("stroke-width", 4)
   .attr("opacity", ".8")
   .datum([{x: labels_x, y: labels_y}, {x: labels_x + 25, y: labels_y}])
   .attr("d",  d3.line()
       .x(function(d) { return d.x; })
       .y(function(d) { return d.y; }));
       
svg.append("path")
   .attr("stroke", "#000")
   .attr("stroke-width", 1)
   .datum([{x: labels_x, y: labels_y - 2}, {x: labels_x + 25, y: labels_y - 2}])
   .attr("d",  d3.line()
       .x(function(d) { return d.x; })
       .y(function(d) { return d.y; }));
       
svg
  .append("text")
  .attr("text-anchor", "start")
  .attr("y", labels_y + 5)
  .attr("x", labels_x + 30)
  .attr("font-family", "Arvo")
  .attr("font-weight", 700)
  .text("H distribution")
  .style("fill", "#65AD69");
  
svg.append("path")
   .attr("stroke", "#EDA137")
   .attr("stroke-width", 4)
   .attr("opacity", ".8")
   .datum([{x: labels_x, y: labels_y + 15}, {x: labels_x + 25, y: labels_y + 15}])
   .attr("d",  d3.line()
       .x(function(d) { return d.x; })
       .y(function(d) { return d.y; }));
       
svg.append("path")
   .attr("stroke", "#000")
   .attr("stroke-width", 1)
   .datum([{x: labels_x, y: labels_y + 13}, {x: labels_x + 25, y: labels_y + 13}])
   .attr("d",  d3.line()
       .x(function(d) { return d.x; })
       .y(function(d) { return d.y; }));
       
svg
  .append("text")
  .attr("text-anchor", "start")
  .attr("y", labels_y + 20)
  .attr("x", labels_x + 30)
  .attr("font-family", "Arvo")
  .attr("font-weight", 700)
  .text("K distribution")
  .style("fill", "#EDA137");
       
svg.append("path")
   .attr("stroke", "#348ABD")
   .attr("stroke-width", 3)
   .attr("opacity", "1")
   .datum([{x: labels_x + 5, y: labels_y + 30}, {x: labels_x + 25, y: labels_y + 30}])
   .attr("d",  d3.line()
       .x(function(d) { return d.x; })
       .y(function(d) { return d.y; }));
 
svg.append('g')
   .selectAll("dot")
   .data([{'x': labels_x + 7, 'y': labels_y + 30}])
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
  .attr("y", labels_y + 35)
  .attr("x", labels_x + 30)
  .attr("font-family", "Arvo")
  .attr("font-weight", 700)
  .text("φ(x)")
  .style("fill", "#348ABD");

svg.append("path")
   .attr("stroke", "#348ABD")
   .attr("stroke-width", 4)
   .attr("opacity", ".8")
   .datum([{x: labels_x + fig_width, y: labels_y}, {x: labels_x + fig_width + 25, y: labels_y}])
   .attr("d",  d3.line()
       .x(function(d) { return d.x; })
       .y(function(d) { return d.y; }));
       
svg.append("path")
   .attr("stroke", "#000")
   .attr("stroke-width", 1)
   .datum([{x: labels_x + fig_width, y: labels_y - 2}, {x: labels_x + fig_width + 25, y: labels_y - 2}])
   .attr("d",  d3.line()
       .x(function(d) { return d.x; })
       .y(function(d) { return d.y; }));
       
svg
  .append("text")
  .attr("text-anchor", "start")
  .attr("y", labels_y + 5)
  .attr("x", labels_x + fig_width + 30)
  .attr("font-family", "Arvo")
  .attr("font-weight", 700)
  .text("1-Φ(x)")
  .style("fill", "#348ABD");
  
var table_text_acc_h = svg
  .append("text")
  .attr("text-anchor", "start")
  .attr("y", 2.35 * fig_height)
  .attr("x", 0.5 * fig_width)
  .attr("font-family", "Arvo")
  .text("Accepted H")
  .style("fill", "#65AD69");
  
var table_text_rej_h = svg
  .append("text")
  .attr("text-anchor", "start")
  .attr("y", 2.35 * fig_height)
  .attr("x", 0.85 * fig_width)
  .attr("font-family", "Arvo")
  .text("Rejected H")
  .style("fill", "#EDA137");
  
var table_text_true_h = svg
  .append("text")
  .attr("text-anchor", "start")
  .attr("y", 2.6 * fig_height)
  .attr("x", 0.2 * fig_width)
  .attr("font-family", "Arvo")
  .text("H is true")
  .style("fill", "#65AD69");
  
var table_text_true_k = svg
  .append("text")
  .attr("text-anchor", "start")
  .attr("y", 2.9 * fig_height)
  .attr("x", 0.2 * fig_width)
  .attr("font-family", "Arvo")
  .text("K is true")
  .style("fill", "#EDA137");
  
var table_text_hh = svg
  .append("text")
  .attr("text-anchor", "start")
  .attr("y", 2.6 * fig_height)
  .attr("x", 0.6 * fig_width)
  .attr("font-family", "Arvo")
  .text("0")
  .style("fill", "#65AD69");
  
var table_text_hk = svg
  .append("text")
  .attr("text-anchor", "start")
  .attr("y", 2.9 * fig_height)
  .attr("x", 0.6 * fig_width)
  .attr("font-family", "Arvo")
  .text("0")
  .style("fill", "#E86456");
  
var table_text_kh = svg
  .append("text")
  .attr("text-anchor", "start")
  .attr("y", 2.6 * fig_height)
  .attr("x", 0.95 * fig_width)
  .attr("font-family", "Arvo")
  .text("0")
  .style("fill", "#E86456");
  
var table_text_kk = svg
  .append("text")
  .attr("text-anchor", "start")
  .attr("y", 2.9 * fig_height)
  .attr("x", 0.95 * fig_width)
  .attr("font-family", "Arvo")
  .text("0")
  .style("fill", "#EDA137");
  
}

simple_hypothesis();

</script>

Simple hypotheses like that are not relevant in practice, <ins>but</ins>:

* They explain intuitively how to construct a test. One needs a so called **confidence interval** $c(X) \subset \Theta$ in which the unknown parameter lies with probability $1-\alpha$. In example above we used that for $c(X) = [\overline{X}_n - u_{1-\alpha} \frac{\sigma}{\sqrt{n}}, \infty)$:
$$P_{\mu_0}(\mu_0 \in c(X)) = P_{\mu_0}(\overline{X}_n \leq \mu_0 + \frac{\sigma}{\sqrt{n}} u_{1-\alpha}) = 1-\alpha.$$
Any such $c(X)$ can be used to construct a test, for example,
$$c'(X) =\Big[\overline{X}_n -u_{1-\frac{\alpha}{2}} \frac{\sigma}{\sqrt{n}}, \overline{X}_n + u_{1-\frac{\alpha}{2}} \frac{\sigma}{\sqrt{n}} \Big].$$
In addition, simple hypotheses tell you on which side the alternative lies.

* Formal results like the NP lemma are useful to derive more relevant results. 


### Monotone likelihood ratio

Let $\Theta = \mathbb{R}$, $\mathcal{P} = \lbrace P_\vartheta \mid \vartheta \in \Theta \rbrace$ and $T\colon \mathcal{X} \rightarrow \mathbb{R}$ be some statistic. Family $\mathcal{P}$ is called **class with monotone (isotonic) likelihood ratio** if for every $\vartheta < \vartheta_1$ there exists monotonically increasing function $H_{\vartheta_0, \vartheta_1} \colon \mathbb{R} \rightarrow [0, \infty)$, such that

$$\frac{p_{\vartheta_1}(x)}{p_{\vartheta_0}(x)} =H_{\vartheta_0, \vartheta_1}(T(x)) \quad P_{\vartheta_0} + P_{\vartheta_1}\text{-a.s.}$$

In our example above we had

$$\frac{p_{\mu_1}(x)}{p_{\mu_0}(x)} = \exp \Big( \frac{1}{\sigma^2} \sum_{i=1}^{n} x_i(\mu_1 - \mu_0) \Big) \cdot f(\sigma^2, \mu_1, \mu_0), $$

which is monotonically increasing in $\overline{x}_n$. This can be generalized to one-parametric exponential families.

Let $\mathcal{P} = \lbrace P_\vartheta \mid \vartheta \in \Theta \rbrace$ be class with monotone likelihood ratio in $T$, $\vartheta \in \Theta$, $\alpha \in (0, 1)$ and we consider the one-tailed hypothesis

$$H\colon\vartheta \leq \vartheta_0 \quad \text{vs} \quad K\colon\vartheta > \vartheta_0.$$

Let also

$$\varphi^*(x) = 1_{\lbrace t(x) > c\rbrace} + \gamma 1_{\lbrace T(x) = c\rbrace},$$

where $c := \inf \lbrace t\ |\ P_{\vartheta_0}(T(X) > t) \leq \alpha \rbrace$ and

$$\gamma = 
				\left \lbrace
				\begin{array}{cl}
				\frac{\alpha - P_{\vartheta_0}(T(X) > c) }{ P_{\vartheta_0}(T(X) = c) }, & \text{if } P_{\vartheta_0}(T(X) = c) \neq 0  \\
				0, & \text{otherwise}.
				\end{array}
				\right.$$

Then $1-\beta_{\varphi^*}(\vartheta_0) = \alpha$ and $\varphi^*$ is UMP test with significance level $\alpha$. Also for any $\vartheta < \vartheta_0$ we have 

$$\beta_{\varphi^*}(\vartheta) = \sup \lbrace \beta_\varphi(\vartheta)\ |\ 1 - \beta_\varphi(\vartheta_0) = \alpha \rbrace.$$

<details>
<summary>Proof</summary>
</details>

Back to our previous example with $X_1, \dots, X_n$ with known $\sigma^2$, we know that 

$$ p_\mu(x) = (2 \pi \sigma^2)^{-\frac{n}{2}} \exp \Big( -\frac{1}{2\sigma^2}\sum_{i=1}^{n}(x_i - \mu)^2 \Big) $$

has a monotone likelihood ratio in $T(X) = \overline{X}_n$. An UMP test with level $\alpha$ is given by

$$\varphi^*(x) = 1_{\lbrace\overline{x}_n > c\rbrace } + \gamma 1_{\lbrace\overline{x}_n = c\rbrace}.$$

Since $P_{\mu_0}(T(X) = c) = 0$, then $\gamma = 0$ and we choose $c$ such that 

$$P_{\mu_0}(\overline{X}_n > c) = \alpha \Longleftrightarrow c = \mu_0 + \frac{\sigma}{\sqrt{n}} u_{1-\alpha}.$$

This UMP test $\varphi^*(x) = 1_{\lbrace \overline{X}_n > \mu_0 + \frac{\sigma}{\sqrt{n}}u_{1-\alpha} \rbrace }$ is called **the one-tailed Gauss test**.

There is a heuristic how to get to the one-tailed Gauss test: since $\overline{X}_n$ is UMVU for $\mu$, a reasonable strategy is to decide for $K$ if $\overline{X}_n$ is "large enough", so the test shoud be of the form $\varphi(x) = 1_{\lbrace \overline{X}_n > c \rbrace }$. Choosing $c$ happens by controlling the error of the 1st kind. For all $\mu \leq \mu_0$ we have

$$ \begin{aligned}
\beta_\varphi(\mu) &= P_\mu(\overline{X}_n > c) \\ &= P_\mu \Big( \frac{\sqrt{n}(\overline{X}_n - \mu) }{\sigma} > \frac{\sqrt{n}(c-\mu)}{\sigma}\Big) \\ &= 1 - \Phi\Big(\frac{\sqrt{n}(c-\mu)}{\sigma}\Big) \\&\leq 1 - \Phi\Big(\frac{\sqrt{n}(c-\mu_0)}{\sigma}\Big).
\end{aligned}$$

So we need to secure that 

$$1- \Phi\Big(\frac{\sqrt{n}(c-\mu_0)}{\sigma}\Big) \leq \alpha \Longleftrightarrow c \geq \mu_0 + \frac{\sigma}{\sqrt{n}} u_{1-\alpha}.$$

We take $c = \mu_0 + \frac{\sigma}{\sqrt{n}} u_{1-\alpha}$ for an error of the 1st kind to be $\alpha$.

This method doesn't tell you anything about optimality, but at least provides a test. Most importantly, it can be applied in more general situations like unknown $\sigma^2$. In this case one can use

$$\hat{\sigma}_n^2 = \frac{1}{n-1}\sum_{i=1}^{n}(X_i - \overline{X}_n)^2.$$

As above we obtain

$$\beta_\varphi(\mu) = P_\mu\Big( \frac{\sqrt{n}(\overline{X}_n - \mu) }{\hat{\sigma}_n} > \frac{\sqrt{n}(c-\mu)}{\hat{\sigma}_n}\Big) = 1 - F_{t_{n-1}}\bigg( \frac{c - \mu}{\sqrt{\hat{\sigma}_n^2 / n}} \bigg),$$

where $F_{t_{n-1}}$ denotes the distribution function of $t_{n-1}$. A reasonable choice is 

$$c = \mu_0 + \frac{\hat{\sigma}_n}{\sqrt{n}}t_{n-1,1-\alpha}, $$

with the corresponding quantile of a $t_{n-1}$ distribution. The test

$$\phi(x) = 1_{\lbrace \overline{x}_n > \mu_0 + \frac{\hat{\sigma}_n}{\sqrt{n}}t_{n-1,1-\alpha} \rbrace} $$

is called **the one-tailed t-test**.

### Two-tailed tests

There are in general no UMP tests for

$$H\colon\vartheta = \vartheta_0 \quad \text{vs} \quad K\colon\vartheta \neq \vartheta_0,$$

because these have to be optimal for all

$$H'\colon\vartheta = \vartheta_0 \quad \text{vs} \quad K'\colon\vartheta = \vartheta_1$$

with $\vartheta_0 \neq \vartheta_1$. In case of monotone likelihood-ratio, the optimal test in this case is 
$$\varphi(x) = 1_{\lbrace T(x) > c \}} + \gamma(x) 1_{\{T(x) = c\rbrace}$$
for $\vartheta_1 > \vartheta_0$ and
 $$\varphi'(x) = 1_{\lbrace T(x) < c'\}} + \gamma'(x) 1_{\{T(x) = c'\rbrace} $$ for $\vartheta_1 < \vartheta_0$. This is not possible.

There exists a theorem for one-parametric exponential family with density

$$p_\vartheta(x) = c(\vartheta)h(x)\exp(Q(\vartheta) T(x))$$

with increasing $Q$: UMPU test for

$$H \colon \vartheta \in [\vartheta_1, \vartheta_2] \quad \text{vs} \quad K\colon\vartheta \notin [\vartheta_1, \vartheta_2]$$

is

$$\varphi^*(x) = 
	\left \{
	\begin{array}{cl}
	1, & \text{if } T(x) \notin [c_1, c_2], \\
	\gamma_i, & \text{if } T(x) = c_i, \\
	0, & \text{if } T(x) \in (c_1, c_2),
	\end{array}
	\right.$$

where the constants $c_i, \gamma_i$ determined from

$$\beta_\varphi(\vartheta_1) = \beta_\varphi(\vartheta_2) = \alpha.$$

Similar results hold for $k$-parametric exponential families.

HERE: visuaization for exponential distribution

### Asymptotic properties of tests

TODO: p-value