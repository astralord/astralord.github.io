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

Let's discuss a simplified clinical study, in which we want to decide whether a newly invented drug $B$ is better than a well-known drug $A$ or not. Suppose that you know from previous years that $A$ has a chance of healing about $p_a$. The new drug $B$ was tested on $n$ persons and $m$ became healthy. Do we choose $A$ or $B$? In terms of mathematics we test

$$H: p_b \leq p_a \quad \text{vs} \quad K: p_b > p_a, $$

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
	P(\overline{X}_n > c) & = P\bigg(\frac{\sqrt{n}(\overline{X}_n - p)}{\sqrt{p(1-p)}} > \frac{\sqrt{n}(c - p)}{\sqrt{p(1-p)}}\bigg) \\
	\color{Salmon}{\text{Central Limit Theorem} \rightarrow} & \approx P\bigg(\mathcal{N}(0,1) > \frac{\sqrt{n}(c - p)}{\sqrt{p(1-p)}}\bigg) \\& = \Phi\bigg(\frac{\sqrt{n}(p - c)}{\sqrt{p(1-p)}}\bigg),
	\end{aligned}	
$$

where $\Phi$ is the distribution function of $\mathcal{N}(0, 1)$. The probability of error is bounded from above:

$$
\begin{aligned}
P(\text{Error of the 1st kind}) &= P(\overline{X}_n > c \mid p_b \leq p_a) \\ &\leq P(\overline{X}_n > c \mid p_b = p_a) \\ & =\Phi\bigg(\frac{\sqrt{n}(p_a - c)}{\sqrt{p_a(1-p_a)}}\bigg).
\end{aligned}	$$

By symmetry,

$$ P(\text{Error of 2nd kind}) \leq 1 - \Phi\bigg(\frac{\sqrt{n}(p_a - c)}{\sqrt{p_a(1-p_a)}}\bigg).$$


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
  .text("P(Error of the 1st kind)")
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
  .text("P(Error of the 2nd kind)")
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
}

basic_test();

</script>

![](.)
*Fig. 1. Visualization of basic test experiment. Parameters $p_a$ and $c$ are draggable.*

Ideally we want to minimize both errors simulaneously and pick the optimal test. The problem is