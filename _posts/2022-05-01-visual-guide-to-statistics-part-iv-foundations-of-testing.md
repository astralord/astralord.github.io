---
layout: post
title: 'Visual Guide to Statistics. Part IV: Foundations of Testing'
date: 2022-05-01 11:00 +0800
categories: [Statistics, Visual Guide]
tags: [statistics, hypothesis, significance level, power of a test, neyman-pearson test, ump-test, confidence interval, one-sided gauss test, one-sided t-test, two-sample t-test, likelihood-ratio test, wilks theorem, bartlett test, chi-square independence test]
math: true
---
  
> This is the fourth and the last part of a 'Visual Guide to Statistics' cycle. All the previous parts and other topics related to statistics could be found [here](https://astralord.github.io/categories/statistics/). 
> In this post we will test hypotheses about the unknown parameter $\vartheta$. As before, we have a statistical experiment with sample space $\mathcal{X}$ and family of probability measures $\mathcal{P} = \lbrace P_\vartheta \mid \vartheta \in \Theta \rbrace$.

## Introductory example

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
  stroke: #ddd;
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
}

#sample-button-k {
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

#sample-button-brt {
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

#sample-button-brt:hover {
  background-color: #696969;
}

#add-button {
  top: 15px;
  left: 15px;
  background: #348ABD;
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

#add-button:hover {
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

#delete-button {
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

#delete-button:hover {
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

d3.select("#basic_test")
  .style("position", "relative");

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
    height = 290 - margin.top - margin.bottom,
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
      .attr("stroke", "currentColor")
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
      .attr("stroke", "currentColor")
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
     .attr("stroke", "black")
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

function dragged_pa(event, d) {
  d3.select(this).attr("cx", d.x = Math.min(x(0.9999), 
                                   Math.max(event.x, x(0.0001))));
  p_a = x.invert(d.x);
  updateErrCurves();
}

function dragged_c(event, d) {
  d3.select(this).attr("cx", d.x = Math.min(xBtm(0.9999), 
                                   Math.max(event.x, xBtm(0.0001))));
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
   .attr("stroke", "currentColor")
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
  .style("font-size", "14px")
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
   .attr("stroke", "currentColor")
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
  .style("font-size", "14px")
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
     .attr("stroke", "black")
     .attr("stroke-width", 1);
   
       
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
        
d3.select("#basic_test")
  .append("div")
  .text("\\(p_b\\)")
  .style("font-size", "15px")
  .style("font-weight", "700")
  .attr("font-family", "Arvo")
  .style("position", "absolute")
  .style("left", labels_x + 15 + margin.left +  "px")
  .style("top", fig_height + 20 + "px");
  
d3.select("#basic_test")
  .append("div")
  .text("\\(p_a\\)")
  .style('color', '#E86456')
  .style("font-size", "15px")
  .style("font-weight", "700")
  .attr("font-family", "Arvo")
  .style("position", "absolute")
  .style("left", fig_width + margin.left + 30 + "px")
  .style("top", labels_y + 2 * labels_v + 12 + "px");
  
d3.select("#basic_test")
  .append("div")
  .text("\\(\\varphi(x) \\)")
  .style('color', '#348ABD')
  .style("font-size", "15px")
  .style("font-weight", "700")
  .attr("font-family", "Arvo")
  .style("position", "absolute")
  .style("left", fig_width + margin.left + 30 + "px")
  .style("top", labels_y + 3 * labels_v + 12 + "px");
    
d3.select("#basic_test")
  .append("div")
  .text("\\(x\\)")
  .style("font-size", "15px")
  .style("font-weight", "700")
  .attr("font-family", "Arvo")
  .attr("font-weight", 700)
  .style("position", "absolute")
  .style("left", labels_x + 15 + margin.left +  "px")
  .style("top", 1.6 * fig_height + 20 + "px");
}


basic_test();

</script>

![](.)
*Fig. 1. Visualization of basic test experiment. Parameters $p_a$ and $c$ are draggable.*

## Power of a test

Ideally we want to minimize both errors simulaneously and pick the optimal test. The problem is that criterias $\varphi_0(x) \equiv 0$ and $\varphi_1(x) \equiv 1$ are optimal if one needs to minimize one of the errors, but they don't minimize both errors at the same time. In practice, the upper bound $\alpha$ is taken for the probability of error of the 1st kind and probability of error of the 2nd kind is minimized. Typically, $0.01 \leq \alpha \leq 0.1$ (the set belonging to the more severe consequences is the alternative).

Now suppose $\varphi$ is a test for $H \colon \vartheta \in \Theta_H$ vs $K \colon \vartheta \in \Theta_K$. Let's define function

$$\beta_\varphi(\vartheta) = 1 - \mathbb{E}_\vartheta[\varphi(X)].$$

Note that for non-randomized test $\varphi$ we have

$$\beta_\varphi(\vartheta) = P_\vartheta(\varphi(X) = 0),$$

which is the probability to decide for $H$. In particular, 

* $\vartheta \in \Theta_H$: $1 - \beta_\varphi(\vartheta)$ is the probability of an error of the 1st kind,
* $\vartheta \in \Theta_K$: $\beta_\varphi(\vartheta)$ is the probability of an error of the 2nd kind.

The function $1 - \beta_\varphi(\vartheta)$ for $\vartheta \in \Theta_K$, which is the probability of correctly rejecting hypothesis $H$, when alterntative $K$ is true, is called **power of a test** $\varphi$. The same intuition holds for randomized tests. Test $\varphi$ is called a test with **significance level** $\alpha \in [0, 1]$ if 

$$1 - \beta_\varphi(\vartheta) \leq \alpha \quad \forall \vartheta \in \Theta_H.$$

A test with significance level $\alpha$ has a probability of an error of the 1st kind, which is bounded by $\alpha$. We will denote set of all tests with significance level $\alpha$ as $\Phi_\alpha$. Test $\varphi$ is also called **unbiased with significance level** $\alpha$ if $\varphi \in \Phi_\alpha$ and 

$$1-\beta_\varphi(\vartheta) \geq \alpha \quad \forall \vartheta \in \Theta_K.$$

For an unbiased test with significance level $\alpha$ the probability of deciding for $K$ for every $\vartheta \in \Theta_K$ is not smaller than for $\vartheta \in \Theta_H$. The set of all unbiased tests with level $\alpha$ we will call $\Phi_{\alpha \alpha}$.

Test $\tilde{\varphi} \in \Phi_\alpha$ is called **uniformly most powerful (UMP)** test with significance level $\alpha$ if

$$\beta_{\tilde{\varphi}}(\vartheta) = \inf_{\varphi \in \Phi_\alpha} \beta_\varphi(\vartheta) \quad \forall \vartheta \in \Theta_K.$$

Test $\tilde{\varphi} \in \Phi_{\alpha\alpha}$ is called **uniformly most powerful unbiased (UMPU)** test with significance level $\alpha$ if

$$\beta_{\tilde{\varphi}}(\vartheta) = \inf_{\varphi \in \Phi_{\alpha\alpha}} \beta_\varphi(\vartheta) \quad \forall \vartheta \in \Theta_K.$$

## Neyman-Pearson lemma

Let's start with *simple hypothesis*:

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

Let $\tilde{\varphi}$ be an NP-test with constant $\tilde{c}$ and let $\varphi$ be some other test with 

$$\beta_\varphi(\vartheta_0) \geq \beta_{\tilde{\varphi}}(\vartheta_0).$$ 

Then we have

$$\begin{aligned} \beta_\varphi(\vartheta_1) - \beta_{\tilde{\varphi}}(\vartheta_1) &= (1 - \beta_{\tilde{\varphi}}(\vartheta_1) ) - (1 - \beta_\varphi(\vartheta_1) ) \\&=\int (\tilde{\varphi} - \varphi) p_1 dx \\&= \int (\tilde{\varphi} - \varphi)(p_1 - \tilde{c}p_0)dx + \int \tilde{c} p_0 (\tilde{\varphi} - \varphi) dx.
\end{aligned}$$

For the first integral note that

$$\begin{aligned}\tilde{\varphi} - \varphi > 0 \Longrightarrow \tilde{\varphi} > 0 \Longrightarrow p_1 \geq \tilde{c}p_0, \\
\tilde{\varphi} - \varphi < 0 \Longrightarrow \tilde{\varphi} < 1 \Longrightarrow p_1 \leq \tilde{c}p_0.
\end{aligned} $$

Hence, $(\tilde{\varphi} - \varphi)(p_1 - \tilde{c}p_0) \geq 0$ always. The second integral is 

$$\tilde{c}(\beta_{\tilde{\varphi}}(\vartheta_0) - \beta_\varphi(\vartheta_0)) \geq 0.$$ 

Therefore we have 

$$\beta_\varphi(\vartheta_1) \geq \beta_{\tilde{\varphi}}(\vartheta_1)$$

and NP-test $\tilde{\varphi}$ is an UMP test with level $\alpha = \mathbb{E}_{\vartheta_0}[\tilde{\varphi}(X)]$. This statement is called **NP lemma**.

There are also other parts of this lemma which I will state here without proof:

* For any $\alpha \in [0, 1]$ there is an NP-test $\varphi$ with $\mathbb{E}_{\vartheta_0}[\varphi(X)] = \alpha$.
* If $\varphi'$ is UMP with level $\alpha$, then $\varphi'$ is (a.s.) an NP-test. Also

$$ \mathbb{E}_{\vartheta_0}[\varphi'(X)] < \alpha \Longrightarrow \mathbb{E}_{\vartheta_1}[\varphi'(X)]=1.$$

An NP-test $\tilde{\varphi}$ for $H \colon \vartheta = \vartheta_0$ vs $K \colon \vartheta = \vartheta_1$ is uniquely defined outside of 

$$S_= =\lbrace x\ |\ p_1(x) = \tilde{c}p_0(x) \rbrace.$$

On $S_=$ set the test can be chosen such that $\beta_{\tilde{\varphi}}(\vartheta_0) = \alpha$.

Is must also be noted that every NP-test $\tilde{\varphi}$ with $\beta_{\tilde{\varphi}}(\vartheta_0) \in (0, 1)$ is unbiased. In particular

$$\alpha := 1 - \beta_{\tilde{\varphi}}(\vartheta_0) < 1 - \beta_{\tilde{\varphi}}(\vartheta_1).$$

<details>
<summary>Proof</summary>
Take test $\varphi \equiv \alpha$. It has significance level $\alpha$ and since $\tilde{\varphi}$ is UMP, we have 

$$1-\beta_\varphi(\vartheta_1) \leq 1-\beta_{\tilde{\varphi}}(\vartheta_1).$$

If $\alpha = 1-\beta_{\tilde{\varphi}}(\vartheta_1) < 1$, then $\varphi \equiv \alpha$ is UMP. Since every UMP test is an NP test, we know that $p_1(x) = \tilde{c}p_0(x)$ for almost all $x$. Therefore, $\tilde{c}=1$ and $p_1 = p_0$ a.s. and also $P_{\vartheta_0} = P_{\vartheta_1}$, which is contradictory.
</details>

## Confidence interval

Let $X_1, \dots X_n$ i.i.d. $\sim \mathcal{N}(\mu,\sigma^2)$ with $\sigma^2$ known. We test

$$H \colon \mu = \mu_0 \quad \text{vs} \quad K \colon \mu = \mu_1$$

with $\mu_0 < \mu_1$. For the density of $X_1, \dots X_n$ it holds

$$p_j(x) = (2 \pi \sigma^2)^{-n/2} \exp \Big( -\frac{1}{2\sigma^2} \Big( \sum_{i=1}^{n} X_i^2 - 2 \mu_j \sum_{i=1}^{n}X_i + n\mu_j^2  \Big)\Big), \quad j = 0, 1.$$

As the inequality for the likelihood ratio which we need for the construction of the NP test, we get

$$\frac{p_1(x)}{p_0(x)} = \exp \Big( \frac{1}{\sigma^2} \sum_{i=1}^{n} x_i(\mu_1 - \mu_0) \Big) \cdot f(\sigma^2, \mu_1, \mu_0) > \tilde{c},$$

where the known constant $f(\sigma^2, \mu_1, \mu_0)$ is positive. This inequality is equivalent to

$$\overline{X}_n = \frac{1}{n} \sum_{i=1}^{n}X_i > c,$$

for some appropriate $c$ (because of $\mu_1 > \mu_0$). Therefore it is equally well possible to determine $c$ such that

$$P_{\mu_0}(\overline{X}_n > c) = \alpha$$

or equivalently

$$\begin{aligned}
	P_{\mu_0}\Big( &\underbrace{\frac{\sqrt{n}(\overline{X}_n - \mu_0)}{\sigma}} > \frac{\sqrt{n}(c-\mu_0)}{\sigma}\Big) = 1 - \Phi\Big(\frac{\sqrt{n}(c - \mu_0)}{\sigma}\Big) = \alpha. \\
	&\quad \color{Salmon}{\sim \mathcal{N}(0, 1)}
	\end{aligned}$$

If we call $u_p$ the **p-quantile** of $\mathcal{N}(0, 1)$, which is the value such that $\Phi(u_p)=p$, then we get

$$\frac{\sqrt{n}(c - \mu_0)}{\sigma} = u_{1-\alpha} \quad \Longleftrightarrow \quad c = \mu_0 + u_{1-\alpha}\frac{\sigma}{\sqrt{n}}.$$

The NP-test becomes

$$\tilde{\varphi}(X) = 1_{\lbrace\overline{X}_n > \mu_0 + u_{1-\alpha} \frac{\sigma}{\sqrt{n}}  \rbrace }.$$



<button id="sample-button-h">Sample H</button> <button id="sample-button-k">Sample K</button> <label id="n-text">n:</label><input type="number" min="1" max="100" step="1" value="10" id="n-num"> <button id="reset-button">Reset</button>


 
<div id="simple_hypothesis"></div> 

<script>
  
d3.select("#simple_hypothesis")
  .style("position", "relative");
  
function randn_bm() {
    var u = 0, v = 0;
    while(u === 0) u = Math.random();
    while(v === 0) v = Math.random();
    return Math.sqrt( -2.0 * Math.log( u ) ) * Math.cos( 2.0 * Math.PI * v );
}

function gauss_data(mu, sigma, min, max) {
  var data = [{x: min, y: 0}];
  for (var i = min; i < max; i += 0.01) {
      data.push({x: i, y: Math.exp(-0.5 * ((i - mu) / sigma) ** 2) / (sigma * Math.sqrt(2 * Math.PI)) });
  }
  data.push({x: max, y: 0});
  return data;
}

function quantile_data(u_q) {
  var data0 = [{x: 0, y: 0}];
  for (var i = 0; i < u_q; i += 0.01) {
      data0.push({x: i, y: 1 - Phi(i) });
  }
  var q = 1 - Phi(u_q);
  data0.push({x: u_q, y: q});
  data0.push({x: u_q, y: 0});
  
  var data1 = [{x: u_q, y: 0}];
  for (var i = u_q; i < 3.5; i += 0.01) {
      data1.push({x: i, y: 1 - Phi(i) });
  }
  data1.push({x: 3.5, y: 0});
  return [data0, data1];
}

    
function createSlider(svg_, parameter_update, x, loc_x, loc_y, letter, color, init_val, round_fun) {
    var slider = svg_.append("g")
      .attr("class", "slider")
      .attr("transform", "translate(" + loc_x + "," + loc_y + ")");
    
    var drag = d3.drag()
	        .on("start.interrupt", function() { slider.interrupt(); })
	        .on("start drag", function(event, d) { 
	          handle.attr("cx", x(round_fun(x.invert(event.x))));  
	          parameter_update(x.invert(event.x));	         });
	         
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
    .style('fill', "currentColor")
    .text(function(d) { return d; });

   var handle = slider.insert("circle", ".track-overlay")
      .attr("class", "handle")
      .attr("r", 5).attr("cx", x(init_val));
      
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
    height = 465 - margin.top - margin.bottom,
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
    
    
    
var mu0_data = gauss_data(mu0, sigma, -4, 4);

var mu0_curve = svg
  .append('g')
  .append("path")
      .datum(mu0_data)
      .attr("fill", "#65AD69")
      .attr("border", 0)
      .attr("opacity", ".8")
      .attr("stroke", "currentColor")
      .attr("stroke-width", 1)
      .attr("stroke-linejoin", "round")
      .attr("d",  d3.line()
        .curve(d3.curveBasis)
          .x(function(d) { return x(d.x); })
          .y(function(d) { return y(d.y); })
   );
   
var mu1_data = gauss_data(mu1, sigma, -4, 4);

var mu1_curve = svg
  .append('g')
  .append("path")
      .datum(mu1_data)
      .attr("fill", "#EDA137")
      .attr("border", 0)
      .attr("opacity", ".8")
      .attr("stroke", "currentColor")
      .attr("stroke-width", 1)
      .attr("stroke-linejoin", "round")
      .attr("d",  d3.line()
        .curve(d3.curveBasis)
          .x(function(d) { return x(d.x); })
          .y(function(d) { return y(d.y); })
   );

var q_data = quantile_data(u_q);
var quantile_curve0 = svg
  .append('g')
  .append("path")
      .datum(q_data[0])
      .attr("fill", "#348ABD")
      .attr("border", 0)
      .attr("opacity", ".8")
      .attr("stroke", "currentColor")
      .attr("stroke-width", 1)
      .attr("stroke-linejoin", "round")
      .attr("d",  d3.line()
        .curve(d3.curveBasis)
          .x(function(d) { return xRight(d.x); })
          .y(function(d) { return yRight(d.y); })
   );
   
var quantile_curve1 = svg
  .append('g')
  .append("path")
      .datum(q_data[1])
      .attr("fill", "#348ABD")
      .attr("border", 0)
      .attr("opacity", ".2")
      .attr("stroke", "currentColor")
      .attr("stroke-width", 1)
      .attr("stroke-linejoin", "round")
      .attr("d",  d3.line()
        .curve(d3.curveBasis)
          .x(function(d) { return xRight(d.x); })
          .y(function(d) { return yRight(d.y); })
   );
  
function updateQuantileCurve() {
  q_data = quantile_data(u_q);
  quantile_curve0
      .datum(q_data[0])
      .transition()
      .duration(0)
      .attr("d",  d3.line()
        .curve(d3.curveBasis)
          .x(function(d) { return xRight(d.x); })
          .y(function(d) { return yRight(d.y); })
      );
  quantile_curve1
      .datum(q_data[1])
      .transition()
      .duration(0)
      .attr("d",  d3.line()
        .curve(d3.curveBasis)
          .x(function(d) { return xRight(d.x); })
          .y(function(d) { return yRight(d.y); })
      );
}   
   
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

function dragged_u(event, d) {
  var u_x = Math.min(xRight(3.5), Math.max(event.x, xRight(0)));
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
  updateQuantileCurve();
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
    mu0_data = gauss_data(mu0, sigma, -4, 4);
    mu1_data = gauss_data(mu1, sigma, -4, 4);
    
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
    sigma = Math.sqrt(x);
    updatePower();
    updateCurves(0); 
    updatePhiLine();
}

function trivialRound(x) { return x; }

var sigma_x = d3.scaleLinear()
    .domain([0.15, 2])
    .range([0, width * 0.4])
    .clamp(true);
    
createSlider(svg, updateSigma, sigma_x, margin.left, 2 * fig_height, "", "#A9A750", sigma, trivialRound);


d3.select("#simple_hypothesis")
  .append("div")
  .text("\\(\\sigma^2 \\)")
  .style('color', '#A9A750')
  .style("font-size", "17px")
  .style("font-weight", "700")
  .attr("font-family", "Arvo")
  .attr("font-weight", 700)
  .attr("font-size", 20)
  .style("position", "absolute")
  .style("left", margin.left + "px")
  .style("top", 2 * fig_height + 15 + "px");
  


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
      var smpl_dots = [];
      var avg = 0;
      for (var i = 0; i < n; i += 1) {
          var random_sample = mu + sigma * randn_bm();
          smpl_dots.push(svg.append('g')
            .selectAll("dot")
            .data([{x: random_sample, y: 1}])
            .enter()
            .append("circle")
              .attr("cx", function (d) { return x(d.x); } )
              .attr("cy", function (d) { return y(d.y); } )
              .attr("r", 3)
              .style("fill", color)
              .attr("stroke", "black")
              .attr("stroke-width", 1));
          
          smpl_dots[i].transition()
            .duration(avg_dur)
            .attr("cx", function (d) { return x(random_sample); } )
            .attr("cy", function (d) { return y(0); } );
      
          avg += random_sample;
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
  .style("font-size", "14px")
  .style("fill", "#348ABD");
  
var power_text = svg
  .append("text")
  .attr("text-anchor", "start")
  .attr("y", 2.9 * fig_height)
  .attr("x", 1.2 * fig_width + 5)
  .attr("font-family", "Arvo")
  .text("Power: " + power)
  .style("font-size", "14px")
  .style("fill", "#348ABD");

   
d3.select("#simple_hypothesis")
  .append("div")
  .text("\\(\\alpha \\)")
  .style("font-size", "15px")
  .style("font-weight", "700")
  .attr("font-family", "Arvo")
  .style("position", "absolute")
  .style("left", 1.2 * fig_width + 5 + margin.left + "px")
  .style("top", 15 + "px");
  
d3.select("#simple_hypothesis")
  .append("div")
  .text("\\(u_{1-\\alpha} \\)")
  .style("font-size", "15px")
  .style("font-weight", "700")
  .attr("font-family", "Arvo")
  .style("position", "absolute")
  .style("left", 1.9 * fig_width + 5 + margin.left + "px")
  .style("top", 2 * fig_height + 15 + "px");
  
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
   .attr("stroke", "currentColor")
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
  .style("font-size", "14px")
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
   .attr("stroke", "currentColor")
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
  .style("font-size", "14px")
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


d3.select("#simple_hypothesis")
  .append("div")
  .text("\\(\\varphi(x) \\)")
  .style('color', '#348ABD')
  .style("font-size", "15px")
  .style("font-weight", "700")
  .attr("font-family", "Arvo")
  .style("position", "absolute")
  .style("left", labels_x + margin.left + 30 + "px")
  .style("top", labels_y + 45 + "px");


svg.append("path")
   .attr("stroke", "#348ABD")
   .attr("stroke-width", 4)
   .attr("opacity", ".8")
   .datum([{x: labels_x + fig_width, y: labels_y}, {x: labels_x + fig_width + 25, y: labels_y}])
   .attr("d",  d3.line()
       .x(function(d) { return d.x; })
       .y(function(d) { return d.y; }));
       
svg.append("path")
   .attr("stroke", "currentColor")
   .attr("stroke-width", 1)
   .datum([{x: labels_x + fig_width, y: labels_y - 2}, {x: labels_x + fig_width + 25, y: labels_y - 2}])
   .attr("d",  d3.line()
       .x(function(d) { return d.x; })
       .y(function(d) { return d.y; }));
       
d3.select("#simple_hypothesis")
  .append("div")
  .text("\\(1-\\Phi(x) \\)")
  .style('color', '#348ABD')
  .style("font-size", "15px")
  .style("font-weight", "700")
  .attr("font-family", "Arvo")
  .style("position", "absolute")
  .style("left", labels_x + fig_width + margin.left + 30 + "px")
  .style("top", labels_y + 15 + "px");
  
var table_text_acc_h = svg
  .append("text")
  .attr("text-anchor", "start")
  .attr("y", 2.35 * fig_height)
  .attr("x", 0.5 * fig_width)
  .attr("font-family", "Arvo")
  .text("Accepted H")
  .style("font-size", "14px")
  .style("fill", "#65AD69");
  
var table_text_rej_h = svg
  .append("text")
  .attr("text-anchor", "start")
  .attr("y", 2.35 * fig_height)
  .attr("x", 0.85 * fig_width)
  .attr("font-family", "Arvo")
  .text("Rejected H")
  .style("font-size", "14px")
  .style("fill", "#EDA137");
  
var table_text_true_h = svg
  .append("text")
  .attr("text-anchor", "start")
  .attr("y", 2.6 * fig_height)
  .attr("x", 0.2 * fig_width)
  .attr("font-family", "Arvo")
  .text("H is true")
  .style("font-size", "14px")
  .style("fill", "#65AD69");
  
var table_text_true_k = svg
  .append("text")
  .attr("text-anchor", "start")
  .attr("y", 2.9 * fig_height)
  .attr("x", 0.2 * fig_width)
  .attr("font-family", "Arvo")
  .text("K is true")
  .style("font-size", "14px")
  .style("fill", "#EDA137");
  
var table_text_hh = svg
  .append("text")
  .attr("text-anchor", "start")
  .attr("y", 2.6 * fig_height)
  .attr("x", 0.6 * fig_width)
  .attr("font-family", "Arvo")
  .text("0")
  .style("font-size", "14px")
  .style("fill", "#65AD69");
  
var table_text_hk = svg
  .append("text")
  .attr("text-anchor", "start")
  .attr("y", 2.9 * fig_height)
  .attr("x", 0.6 * fig_width)
  .attr("font-family", "Arvo")
  .text("0")
  .style("font-size", "14px")
  .style("fill", "#E86456");
  
var table_text_kh = svg
  .append("text")
  .attr("text-anchor", "start")
  .attr("y", 2.6 * fig_height)
  .attr("x", 0.95 * fig_width)
  .attr("font-family", "Arvo")
  .text("0")
  .style("font-size", "14px")
  .style("fill", "#E86456");
  
var table_text_kk = svg
  .append("text")
  .attr("text-anchor", "start")
  .attr("y", 2.9 * fig_height)
  .attr("x", 0.95 * fig_width)
  .attr("font-family", "Arvo")
  .text("0")
  .style("font-size", "14px")
  .style("fill", "#EDA137");
  
}

simple_hypothesis();

</script>

![](.)
*Fig. 2. Visualization of simple hypothesis testing with $\mu_0 = -1$ and $\mu_1=1$. Significance level $\alpha$ on the right-hand side is draggable.*

Simple hypotheses like that are not relevant in practice, <ins>but</ins>:

* They explain intuitively how to construct a test. One needs a so called **confidence interval** $c(X) \subset \Theta$ in which the unknown parameter lies with probability $1-\alpha$. In example above we used that for 

$$c(X) = [\overline{X}_n - u_{1-\alpha} \frac{\sigma}{\sqrt{n}}, \infty)$$

we have
 
$$P_{\mu_0}(\mu_0 \in c(X)) = P_{\mu_0}(\overline{X}_n \leq \mu_0 + \frac{\sigma}{\sqrt{n}} u_{1-\alpha}) = 1-\alpha.$$

Any such $c(X)$ can be used to construct a test, for example,

$$c'(X) =\Big[\overline{X}_n -u_{1-\frac{\alpha}{2}} \frac{\sigma}{\sqrt{n}}, \overline{X}_n + u_{1-\frac{\alpha}{2}} \frac{\sigma}{\sqrt{n}} \Big].$$

In addition, simple hypotheses tell you on which side the alternative lies.

* Formal results like the NP lemma are useful to derive more relevant results. 


## Monotone likelihood ratio

Let $\Theta = \mathbb{R}$, $\mathcal{P} = \lbrace P_\vartheta \mid \vartheta \in \Theta \rbrace$ and $T\colon \mathcal{X} \rightarrow \mathbb{R}$ be some statistic. Family $\mathcal{P}$ is called **class with monotone (isotonic) likelihood ratio** if for every $\vartheta < \vartheta_1$ there exists monotonically increasing function $H_{\vartheta_0, \vartheta_1} \colon \mathbb{R} \rightarrow [0, \infty)$, such that

$$\frac{p_{\vartheta_1}(x)}{p_{\vartheta_0}(x)} =H_{\vartheta_0, \vartheta_1}(T(x)) \quad P_{\vartheta_0} + P_{\vartheta_1}\text{-a.s.}$$

In our example above we had

$$\frac{p_{\mu_1}(x)}{p_{\mu_0}(x)} = \exp \Big( \frac{1}{\sigma^2} \sum_{i=1}^{n} x_i(\mu_1 - \mu_0) \Big) \cdot f(\sigma^2, \mu_1, \mu_0), $$

which is monotonically increasing in $\overline{x}_n$. This can be generalized to one-parametric [exponential families](https://astralord.github.io/posts/visual-guide-to-statistics-part-i-basics-of-point-estimation/#exponential-family).

Let $\mathcal{P} = \lbrace P_\vartheta \mid \vartheta \in \Theta \rbrace$ be class with monotone likelihood ratio in $T$, $\vartheta \in \Theta$, $\alpha \in (0, 1)$ and we test the one-sided hypothesis

$$H\colon\vartheta \leq \vartheta_0 \quad \text{vs} \quad K\colon\vartheta > \vartheta_0.$$

Let also

$$\tilde{\varphi}(x) = 1_{\lbrace T(x) > c\rbrace} + \gamma 1_{\lbrace T(x) = c\rbrace},$$

where $c = \inf \lbrace t \mid P_{\vartheta_0}(T(X) > t) \leq \alpha \rbrace$ and

$$\gamma = 
				\left \lbrace
				\begin{array}{cl}
				\frac{\alpha - P_{\vartheta_0}(T(X) > c) }{ P_{\vartheta_0}(T(X) = c) }, & \text{if } P_{\vartheta_0}(T(X) = c) \neq 0  \\
				0, & \text{otherwise}.
				\end{array}
				\right.$$

Then $1-\beta_{\tilde{\varphi}}(\vartheta_0) = \alpha$ and $\tilde{\varphi}$ is UMP test with significance level $\alpha$.

<details>
<summary>Proof</summary>
We have

$$1-\beta_{\tilde{\varphi}}(\vartheta_0)=P_{\vartheta_0}(T(X)>c) + \gamma P_{\vartheta_0}(T(X) = c) = \alpha. $$

Let $\vartheta_0 < \vartheta_1$, then due to monotonicity

$$H_{\vartheta_0, \vartheta_1}(T(x)) > H_{\vartheta_0, \vartheta_1}(c) = s \quad \Longrightarrow \quad T(x) > c $$

and

$$\tilde{\varphi}(x) =
		\left \{
		\begin{array}{cl}
		1, & H_{\vartheta_0, \vartheta_1}(x) > s, \\
		0, & H_{\vartheta_0, \vartheta_1}(x) < s.
		\end{array}
		\right.$$

Therefore $\tilde{\varphi}$ is NP-test with significance level $\alpha$ and by NP lemma

$$ \beta_{\tilde{\varphi}}(\vartheta_1) = \inf \lbrace \beta_\varphi(\vartheta_1)\ |\  \beta_\varphi(\vartheta_0) = 1-\alpha \rbrace. $$
	    
As $\tilde{\varphi}$ doesn't depend on $\vartheta_1$, this relation holds for all $\vartheta_1 > \vartheta_0$. Finally, let $\varphi'(x) = 1 - \tilde{\varphi}(x)$. Using the similar reasoning as above one can show that

$$\beta_{\varphi'}(\vartheta_2) = \inf \lbrace \beta_\varphi(\vartheta_2)\ |\ \beta_\varphi(\vartheta_0) = 1 - \alpha \rbrace \quad \forall \vartheta_2 < \vartheta_0. $$

For trivial test $\overline{\varphi} \equiv \alpha$ the following equality takes place: $\beta_{\overline{\varphi}}(\vartheta_0) = 1-\alpha$. Hence we conclude that

$$1-\beta_{\tilde{\varphi}}(\vartheta_2) = \beta_{\varphi'}(\vartheta_2) \geq \beta_{1-\overline{\varphi}}(\vartheta_2) = 1-\beta_{\overline{\varphi}}(\vartheta_2) = \alpha.  $$
	    
Hence, $1-\beta_{\tilde{\varphi}}(\vartheta_2) \geq \alpha$, $\tilde{\varphi} \in \Phi_\alpha$ and $\tilde{\varphi}$ is UMP test.
 
Also for any $\vartheta < \vartheta_0$ we have 

$$\beta_{\tilde{\varphi}}(\vartheta) = \sup \lbrace \beta_\varphi(\vartheta)\ |\ 1 - \beta_\varphi(\vartheta_0) = \alpha \rbrace,$$

because of $\beta_{\varphi'} = 1 - \beta_{\tilde{\varphi}}$.

</details>

Back to our previous example with $X_1, \dots, X_n$ with known $\sigma^2$, we know that 

$$ p_\mu(x) = (2 \pi \sigma^2)^{-\frac{n}{2}} \exp \Big( -\frac{1}{2\sigma^2}\sum_{i=1}^{n}(x_i - \mu)^2 \Big) $$

has a monotone likelihood ratio in $T(X) = \overline{X}_n$. An UMP test with level $\alpha$ is given by

$$\tilde{\varphi}(x) = 1_{\lbrace\overline{x}_n > c\rbrace } + \gamma 1_{\lbrace\overline{x}_n = c\rbrace}.$$

Since $P_{\mu_0}(T(X) = c) = 0$, then $\gamma = 0$ and we choose $c$ such that 

$$P_{\mu_0}(\overline{X}_n > c) = \alpha \Longleftrightarrow c = \mu_0 + \frac{\sigma}{\sqrt{n}} u_{1-\alpha}.$$

This UMP test 

$$\tilde{\varphi}(x) = 1_{\lbrace \overline{X}_n > \mu_0 + \frac{\sigma}{\sqrt{n}}u_{1-\alpha} \rbrace }$$

is called **the one-sided Gauss test**.

There is a heuristic how to get to the one-sided Gauss test: since $\overline{X}_n$ is UMVU for $\mu$, a reasonable strategy is to decide for $K$ if $\overline{X}_n$ is "large enough", so the test shoud be of the form 

$$\varphi(x) = 1_{\lbrace \overline{X}_n > c \rbrace }.$$

Choosing $c$ happens by controlling the error of the 1st kind. For all $\mu \leq \mu_0$ we have

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

is called **the one-sided t-test**.

## Two-sided tests

There are in general no UMP tests for

$$H\colon\vartheta = \vartheta_0 \quad \text{vs} \quad K\colon\vartheta \neq \vartheta_0,$$

because these have to be optimal for all

$$H'\colon\vartheta = \vartheta_0 \quad \text{vs} \quad K'\colon\vartheta = \vartheta_1$$

with $\vartheta_0 \neq \vartheta_1$. In case of monotone likelihood-ratio, the optimal test in this case is 

$$\varphi(x) = 1_{\lbrace T(x) > c \rbrace} + \gamma(x) 1_{\lbrace T(x) = c\rbrace}$$

for $\vartheta_1 > \vartheta_0$ and

 $$\varphi'(x) = 1_{\lbrace T(x) < c'\rbrace } + \gamma'(x) 1_{\lbrace T(x) = c'\rbrace} $$ 
 
for $\vartheta_1 < \vartheta_0$. This is not possible.

There is a theorem for one-parametric exponential family with density

$$p_\vartheta(x) = c(\vartheta)h(x)\exp(Q(\vartheta) T(x))$$

with increasing $Q$: UMPU test for

$$H \colon \vartheta \in [\vartheta_1, \vartheta_2] \quad \text{vs} \quad K\colon\vartheta \notin [\vartheta_1, \vartheta_2]$$

is

$$\varphi(x) = 
	\left \lbrace
	\begin{array}{cl}
	1, & \text{if } T(x) \notin [c_1, c_2], \\
	\gamma_i, & \text{if } T(x) = c_i, \\
	0, & \text{if } T(x) \in (c_1, c_2),
	\end{array}
	\right.$$

where the constants $c_i, \gamma_i$ determined from

$$\beta_\varphi(\vartheta_1) = \beta_\varphi(\vartheta_2) = 1-\alpha.$$

Similar results hold for $k$-parametric exponential families.

Take an example: let $X$ be exponentially distributed random variable: $X \sim \operatorname{Exp}(\vartheta)$ with density

$$f_\vartheta(x) = \vartheta e^{-\vartheta x} 1_{[0, \infty)}(x)$$

and we test

$$H\colon \vartheta \in [1, 2] \quad \text{vs} \quad K\colon\vartheta \notin [1, 2].$$

We have $T(x) = x$ and

$$\varphi(x) = 1_{\lbrace X \notin [c_1, c_2] \rbrace}.$$

It is known that for $X$ distribution function is $F(x) = 1 - e^{-\vartheta x}$, therefore

$$\begin{aligned}
P_{1}(X \in [c_1, c_2])&=e^{-c_1}-e^{-c_2} = 1-\alpha, \\
P_{2}(X \in [c_1, c_2])&=e^{-2 c_1}-e^{-2 c_2} = 1-\alpha.
\end{aligned}$$

Solving this for $c_1$ and $c_2$ we get

$$c_1 = \ln\frac{2}{2-\alpha}, \quad c_2 = \ln\frac{2}{\alpha}.$$

## Asymptotic properties of tests

Let $X_1, \dots , X_m$ i.i.d. $\sim \mathcal{N}(\mu_1, \sigma^2)$ and $Y_1, \dots , Y_n$ i.i.d. $\sim \mathcal{N}(\mu_2, \tau^2)$ are two independent samples. We want to test the hypothesis:

$$H\colon \mu_1 \leq \mu_2 \quad \text{vs} \quad K\colon \mu_1 > \mu_2. $$

We reject $H$ if $\overline{Y}_n$ is much smaller than $\overline{X}_m$.

* Let $\sigma^2=\tau^2$, but variance is unknown. We know from [Part I](https://astralord.github.io/posts/visual-guide-to-statistics-part-i-basics-of-point-estimation/) that

$$\overline{X}_m - \overline{Y}_n=\mathcal{N}\bigg(\mu_1-\mu_2, \sigma^2\bigg( \frac{1}{m}+\frac{1}{n} \bigg)\bigg)$$ 

and **pooled variance**:

$$\hat{\sigma}_{m,n}^2=\frac{1}{m+n-2}\Big( \sum_{i=1}^{m}(X_i-\overline{X}_m)^2+\sum_{i=1}^{n}(Y_i-\overline{Y}_n)^2 \Big) \sim \frac{\sigma^2}{m+n-2} \chi_{m+n-2}^2. $$

For $\mu_1=\mu_2$ we have

$$ T_{m,n}=\sqrt{\frac{mn}{m+n}}\frac{\overline{X}_m-\overline{Y}_n}{\hat{\sigma}_{m,n}} \sim t_{m+n-2}, $$

therefore test

$$ \varphi_{m,n}(x)=1_{\lbrace T_{m,n} > t_{m+n-2, 1-\alpha}\rbrace }$$

is UMPU with significance level $\alpha$. This test is called **two-sample t-test**.

* Let $\sigma^2 \neq \tau^2$, then

$$  \overline{X}_m - \overline{Y}_n=\mathcal{N}\bigg(\mu_1-\mu_2, \frac{\sigma^2}{m}+\frac{\tau^2}{n} \bigg).$$

Unbiased estimators for variances are

$$ \hat{s}_{m}^2(X)=\frac{1}{m-1}\sum_{i=1}^{m}(X_i-\overline{X}_m)^2, \quad \hat{s}_{n}^2(Y)=\frac{1}{n-1}\sum_{i=1}^{n}(Y_i-\overline{Y}_n)^2. $$

Let also 

$$\hat{s}_{m, n}^2 = \frac{1}{m}\hat{s}_{m}^2(X) + \frac{1}{n}\hat{s}_{n}^2(Y).$$

The distribution of random variable 

$$T_{m,n}^*=\frac{\overline{X}_m-\overline{Y}_n}{\hat{s}_{m, n}}$$

was unknown until recently (so called [Behrens-Fisher problem](https://en.wikipedia.org/wiki/Behrens%E2%80%93Fisher_problem)) and can't be expressed in terms of elementary functions, but from Central Limit Theorem we know that

$$\frac{\overline{X}_m-\overline{Y}_n - (\mu_1-\mu_2)}{\hat{s}_{m,n}} \xrightarrow[]{\mathcal{L}} \mathcal{N}(0,1),$$

if $m \rightarrow \infty$, $n \rightarrow \infty$ and $\frac{m}{n}\rightarrow \lambda \in (0, \infty)$. Let

$$\varphi_{m,n}^*(x)=1_{\lbrace T_{m,n}^* > u_{1-\alpha}\rbrace },$$

then

$$\begin{aligned}
		 \beta_{\varphi_{m,n}^*}(\mu_1, \mu_2) & =P_{\mu_1, \mu_2}(T_{m,n}^* \leq u_{1-\alpha})\\&=P_{\mu_1, \mu_2}\Big(\frac{\overline{X}_m-\overline{Y}_n - (\mu_1-\mu_2)}{\hat{s}_{m,n}} \leq \frac{- (\mu_1-\mu_2)}{\hat{s}_{m,n}}+ u_{1-\alpha}\Big) \\
		  & \xrightarrow[m \rightarrow \infty,\ n \rightarrow \infty,\ \frac{m}{n}\rightarrow \lambda]{}
		 \left \lbrace
		 \begin{array}{cl}
		 0, & \mu_1 > \mu_2, \\
		  1-\alpha, & \mu_1=\mu_2, \\
		  1, & \mu_1<\mu_2.
		 \end{array}
		 \right.
		 \end{aligned}$$

* We say that sequence $(\varphi_n)$ has **asymptotic significance level** $\alpha$, if 

$$\lim_{n \rightarrow \infty} \inf_{\vartheta \in \Theta_H} \beta_{\varphi_n}(\vartheta) \geq 1-\alpha.$$ 

* We say that sequence $(\varphi_n)$ is **consistent**, if 

$$\lim_{n \rightarrow \infty} \beta_{\varphi_n}(\vartheta) = 0 \quad \forall \vartheta \in \Theta_K.$$

In our example $\varphi_{m,n}^*(x)$ is consistent and has asymptotic significance level $\alpha$.

## Likelihood ratio

The common principle of building tests for 

$$H\colon\vartheta \in \Theta_H \quad \text{vs} \quad K\colon\vartheta \in \Theta_K $$

is **likelihood ratio method**. Let $f_n(x^{(n)},\vartheta)$ be density of $P_\vartheta^n$. Then 

$$\lambda(x^{(n)})=\frac{\sup_{\vartheta \in \Theta_H}f_n(x^{(n)},\vartheta)}{\sup_{\vartheta \in \Theta}f_n(x^{(n)},\vartheta)}$$

is **likelihood ratio** and

$$\varphi_n(x^{(n)})=1_{\lbrace \lambda(x^{(n)})<c \rbrace }$$

is **likelihood ratio test**. It is common to choose $c$, such that

$$\sup_{\vartheta \in \Theta_H} P_\vartheta(\lambda(X^{(n)})<c) \leq \alpha.$$

Distribution $\lambda(X^{(n)})$ nevertheless can be estimated only asymptotically.

Before we continue further, we will formulate some conditions. Let $\Theta \subset \mathbb{R}^d$ and there exist $\Delta \subset \mathbb{R}^c$ and twice continuously differentiable function $h:\Delta \rightarrow \Theta$, such that $\Theta_H = h(\Delta)$ and Jacobian of $h$ is matrix of full rank.

For example, let $X_1, \dots, X_n$ i.i.d. $\sim \mathcal{N}(\mu_1, \sigma^2)$ and $Y_1, \dots, Y_n$ i.i.d. $\sim \mathcal{N}(\mu_2, \sigma^2)$ be two independent samples. Suppose we want to test the equivalency of means:

$$H\colon \mu_1 = \mu_2 \quad \text{vs} \quad K\colon \mu_1 \neq \mu_2.$$

Then $\Theta \in \mathbb{R}^2 \times \mathbb{R}^+$, $\Delta = \mathbb{R} \times \mathbb{R}^+$ and $h(\mu, \sigma^2) = (\mu, \mu, \sigma^2)$. Jacobian matrix is

$$J = \begin{pmatrix}
	1 & 0 \\
	1 & 0 \\
	0 & 1
	\end{pmatrix}, $$

matrix of full rank.

Let 

$$\hat{\eta}_n=\arg\max_{\eta \in \Delta}f_n(X^{(n)},h(\eta)) \quad \text{and} \quad \hat{\theta}_n=\arg\max_{\vartheta \in \Theta}f_n(X^{(n)},\vartheta)$$

be maximum-likelihood estimators for families 

$$\mathcal{P}_h = \lbrace P_{h(\eta)}\ |\ \eta \in \Delta\rbrace \quad \text{and} \quad \mathcal{P}_\vartheta = \lbrace P_\vartheta\ |\ \vartheta \in \Theta \rbrace$$

respectively. Also let conditions from [theorem of asymptotic efficiency for maximum-likelihood estimators](https://astralord.github.io/posts/visual-guide-to-statistics-part-iii-asymptotic-properties-of-estimators/#asymptotic-efficiency-of-maximum-likelihood-estimators) for both families be satisfied. Then

$$ T_n=-2\log \lambda(X^{(n)})=2(\log f_n(X^{(n)}, \hat{\theta}_n)-\log f_n(X^{(n)}, h(\hat{\eta}_n))) \xrightarrow[]{\mathcal{L}} \chi_{d-c}^2,$$

if $\vartheta \in \Theta_H$.

<details>
<summary>Proof</summary>
As before we use notation

$$\ell(x, \vartheta) = \log f(x, \vartheta).$$

We start with

$$\begin{aligned}
	    T_n^{(1)} & = 2(\log f_n(X^{(n)}, \hat{\theta}_n)-\log f_n(X^{(n)}, \vartheta)) \\
	    & = 2\sum_{i=1}^{n}\Big(\ell(X_i, \hat{\theta}_n) - \ell(X_i, \vartheta)\Big) \\
	    & = 2(\hat{\theta}_n - \vartheta)^T \sum_{i=1}^{n} \dot{\ell}(X_i, \vartheta) +(\hat{\theta}_n - \vartheta)^T \sum_{i=1}^{n} \ddot{\ell}(X_i, \widetilde{\vartheta}_n)(\hat{\theta}_n - \vartheta)   \\
	    & = 2 (\hat{\theta}_n - \vartheta)^T \Big( \sum_{i=1}^{n} \dot{\ell}(X_i, \vartheta) + \sum_{i=1}^{n} \ddot{\ell}(X_i, \widetilde{\vartheta}_n)(\hat{\theta}_n - \vartheta) \Big) - (\hat{\theta}_n - \vartheta)^T\sum_{i=1}^{n}\ddot{\ell}(X_i, \widetilde{\vartheta}_n)(\hat{\theta}_n - \vartheta)
	\end{aligned}$$
	
for some $\widetilde{\theta}_n \in [\hat{\theta}_n, \vartheta]$. Using the notations from [Part III](https://astralord.github.io/posts/visual-guide-to-statistics-part-iii-asymptotic-properties-of-estimators/#asymptotic-efficiency-of-maximum-likelihood-estimators) we rewrite the first term of equation above:

$$\begin{aligned}
	 2n(\hat{\theta}_n - \vartheta)^T& \underbrace{(\dot{L}_n(\vartheta) - \ddot{L}_n(\tilde{\vartheta})(\hat{\theta}_n - \vartheta))}. \\
	 & \qquad \qquad\ \color{\Salmon}{ = 0 \text{ (by Mean Theorem)}}
	 \end{aligned}$$
	 
Also

$$T_n^{(1)} = -\sqrt{n}(\hat{\theta}_n - \vartheta)^T \ddot{L}_n(\widetilde{\vartheta}_n) \sqrt{n}(\hat{\theta}_n - \vartheta),
$$

where

$$
\begin{aligned}
	 \sqrt{n}(\hat{\theta}_n - \vartheta)^T & \xrightarrow[]{\mathcal{L}} \mathcal{N}(0, I^{-1}(f(\cdot, \vartheta))), \\
	 \ddot{L}_n(\widetilde{\vartheta}_n)& \xrightarrow[]{\mathbb{P}} -I(f(\cdot, \vartheta)), \\
	 \sqrt{n}(\hat{\theta}_n - \vartheta) &\xrightarrow[]{\mathcal{L}} \mathcal{N}(0, I^{-1}(f(\cdot, \vartheta))).
	 \end{aligned}$$
	
We know that for $X \sim \mathcal{N}_d(0, \Sigma)$ with $\Sigma > 0$ we have

$$X^T \Sigma X ~ \sim \mathcal{X}_d^2.$$

Therefore,

$$T_n^{(1)} \xrightarrow[]{\mathcal{L}} A \sim \mathcal{X}_d^2.$$

In the same way,
$$ T_n^{(2)} = 2 (\log f_n(X^{(n)}, h(\hat{\eta}_n) ) - \log f_n(X^{(n)},h(\eta))) \xrightarrow[]{\mathcal{L}} B \sim \mathcal{X}_c^2. $$
	 
If $H$ is true, then $\vartheta = h(\eta)$ and

$$T_n = T_n^{(1)} - T_n^{(2)} \xrightarrow[]{\mathcal{L}} A-B \sim \mathcal{X}_{d-c}^2,$$

which follows from independence of $A-B$ and $B$.
	 
</details>

This statement is called **Wilk's theorem** and it shows that

$$\varphi_n (X^{(n)}) =
		\left \{
		\begin{array}{cl}
		1, & -2\log\lambda(X^{(n)}) > \mathcal{X}_{d-c, 1-\alpha}^2, \\
		0, & \text{otherwise}
		\end{array}
		\right.$$		
		
is a test with asymptotic level $\alpha$. Also, sequence $(\varphi_n)$ is consistent, because 

$$\begin{aligned}
		-\frac{2}{n} \log (\lambda(X^{(n)})) & = \frac{2}{n} \sum_{i=1}^{n} \Big( \ell(X_i, \hat{\theta}_n) - \ell(X_i, h(\hat{\eta}_n)) \Big) \\
		& \xrightarrow{\mathcal{L}} 2 \mathbb{E}_\vartheta[\ell(X,\vartheta) - \ell(X, h(\eta))] \\
		& = 2 KL(\vartheta | h(\eta)) > 0,
		\end{aligned}$$
		
if $\vartheta \neq h(\eta)$. Hence for $\vartheta \in \Theta_K$

$$-2\log(\lambda(X^{(n)}))\xrightarrow{\mathcal{L}} \infty.$$

## Likelihood-ratio tests

Take an example: let $X_{ij} \sim \mathcal{N}(\mu_i, \sigma_i^2)$, $i = 1, \dots, r$ and $j = 1, \dots, n_i$, where $n_i \rightarrow \infty$ with the same speed. We test equivalence of variances:

$$ H\colon \sigma_1^2 = \dots = \sigma_r^2 \quad \text{vs} \quad K \colon \sigma_i^2 \neq \sigma_j^2 \text{ for some } i \neq j. $$
	
Here $\Theta = \mathbb{R}^r \times (\mathbb{R}^+)^r$, $\Delta = \mathbb{R}^r \times \mathbb{R}^+$ and

$$h((x_1, \dots, x_r, y)^T) = (x_1, \dots, x_r, y, \dots, y)^T.$$

Maximum-likelihood estimator is 

$$\hat{\theta}_n = (\overline{X}_{1 \cdot}, \dots, \overline{X}_{r \cdot}, \hat{s}_1^2, \dots, \hat{s}_r^2)$$

with 

$$\overline{X}_{i \cdot} = \frac{1}{n_i} \sum_{j=1}^{n_i}X_{ij} 
\quad \text{and} \quad
\hat{s}_i^2 = \frac{1}{n_i}\sum_{j=1}^{n_i}(X_{ij} -\overline{X}_{i \cdot})^2. $$ 

Then 

$$f_n(X^{(n)}, \hat{\vartheta}_n) = \prod_{i=1}^{r} (2 \pi e \hat{s}_i^2)^{-\frac{n_i}{2}}.$$

Under null hypothesis maximum-likelihood estimator maximizes

$$f_n(X^{(n)}, \hat{\eta}_n) = \prod_{i=1}^{r} (2 \pi \sigma^2)^{-\frac{n_i}{2}} \exp \Big( -\frac{1}{2\sigma^2} \sum_{j=1}^{n_i} (X_{ij} - \overline{X}_{i \cdot})^2 \Big ). $$

Setting $n = \sum_{i=1}^{r}n_i$, we get

$$ \hat{\sigma}^2 = \frac{1}{n}\sum_{i=1}^{r} \sum_{j=1}^{n_i} (X_{ij}-X_{i \cdot})^2 = \sum_{i=1}^r \frac{n_i}{n}\hat{s}_i^2. $$

Then

$$f_n(X^{(n)}, \hat{\eta}_n) = \prod_{i=1}^{r}(2\pi e\hat{\sigma}^2)^{-\frac{n_i}{2}} = (2\pi e \hat{\sigma}^2)^{-\frac{n}{2}}$$

and test statistic becomes

$$T_n = -2\log \lambda(X^{(n)}) = n \log \hat{\sigma}^2 - \sum_{i=1}^{r} n_i \log \hat{s}_i^2.$$

The test 

$$
\varphi_n(X^{(n)}) =
	\left \{
	\begin{array}{cl}
	1, & T_n > \mathcal{X}_{r-1, 1-\alpha}^2, \\
	0, & \text{otherwise}. 
	\end{array}
	\right.
$$
	
is called **the Bartlett test**.


<button id="sample-button-brt">Sample</button> <button id="add-button">Add</button> <button id="delete-button">Delete</button> <button id="reset-button-2">Reset</button>


<div id="asymptotic_test"></div> 

<script>
  
d3.select("#asymptotic_test")
  .style("position", "relative");

function asymptotic_test() {

var mus = [],
    sigmas = [],
    alpha = 0.05,
    n = 30,
    curve_id = -1,
    std_avg = 0;
    
var smpl_dots = [], tn_dots = [], stds = [];
var si_texts = [], sigma_text = null, t_text = null;

var colors = ["#65AD69", "#EDA137", "#E86456", "#B19CD9", "#A4D8D8"];
var booked_colors = [];

var margin = {top: 30, right: 0, bottom: 20, left: 30},
    width = 750 - margin.left - margin.right,
    height = 430 - margin.top - margin.bottom,
    fig_height = 250 - margin.top - margin.bottom,
    fig_width = 350;
    
var svg = d3.select("div#asymptotic_test")
  .append("svg")
    .attr("width", width + margin.left + margin.right)
    .attr("height", height + margin.top + margin.bottom)
  .append("g")
    .attr("transform",
          "translate(" + margin.left + "," + margin.top + ")");

var x = d3.scaleLinear()
          .domain([-10, 10])
          .range([0, fig_width]);
            
var xAxis = svg.append("g")
   .attr("transform", "translate(0," + fig_height + ")")
   .call(d3.axisBottom(x));
  
xAxis.selectAll(".tick text")
   .attr("font-family", "Arvo");


var xRight = d3.scaleLinear()
          .domain([0, 20])
          .range([1.2 * fig_width, 1.9 * fig_width]);
          
var xAxisRight = svg.append("g")
   .attr("transform", "translate(0," + fig_height + ")")
   .call(d3.axisBottom(xRight).ticks(5));
  
xAxisRight.selectAll(".tick text")
   .attr("font-family", "Arvo");
   

var y = d3.scaleLinear()
          .range([fig_height, 0])
          .domain([0, 1]);
            
var yAxis = svg.append("g")
    .call(d3.axisLeft(y).ticks(4));
  
yAxis.selectAll(".tick text")
    .attr("font-family", "Arvo");

var yRight = d3.scaleLinear()
          .range([fig_height, 0])
          .domain([0, 1]);
            
var yAxisRight = svg.append("g")
   .attr("transform", "translate(" + 1.2 * fig_width + ",0)")
    .call(d3.axisLeft(yRight).ticks(4));
  
yAxisRight.selectAll(".tick text")
    .attr("font-family", "Arvo");
    


var gauss_curves = [];

function addCurve(mu, sigma) {
    reset();
    const k = mus.length;
    mus.push(mu);
    sigmas.push(sigma);
    if (curve_id >= 0) {
        gauss_curves[curve_id]
            .transition()
            .attr("opacity", ".2");
    }
    curve_id = k;               
    var data = gauss_data(mus[k], sigmas[k], -10, 10);
    
    var color = "currentColor";
    for (var i = 0; i < colors.length; i += 1) {
        if (booked_colors.indexOf(colors[i]) < 0) {
             color = colors[i];
             booked_colors.push(color);
             break;
        }
    }
    
    gauss_curves.push(svg
                       .append('g')
                       .append("path")
                       .datum(data)
                       .attr("fill", color) 
                       .attr("border", 0)
                       .attr("opacity", (k == curve_id) ? ".8" : ".2")
                       .attr("stroke", "currentColor")
                       .attr("stroke-width", 1)
                       .attr("stroke-linejoin", "round")
                       .attr("d",  d3.line()
                         .curve(d3.curveBasis)
                         .x(function(d) { return x(d.x); })
                         .y(function(d) { return y(d.y); })
                      )
                      .on('mouseover', function() {
                          d3.select(this)
                            .transition()
                            .attr("opacity", ".8")
                            .style("cursor", "pointer");
	                   })
	                   .on('mouseout', function() {
	                       d3.select(this)
	                         .transition()
	                         .attr("opacity", (curve_id == k) ? ".8" : ".2");
	                       })
	                   .on('click', function() {
	                       gauss_curves[curve_id]
	                         .transition()
	                         .attr("opacity", ".2");
	                       curve_id = k;
	                       d3.select(this)
	                         .transition()
	                         .attr("opacity", ".8");
	                       muHandler.attr("cx", mu_x(mus[k]));
	                       sigmaHandler.attr("cx", sigma_x(sigmas[k]));
	                   })
                    );
}


function initData() {
    addCurve(-1, 1);
    addCurve(1, 1);
}

initData();

function updateCurve(mu, sigma) {
    mus[curve_id] = mu;
    sigmas[curve_id] = sigma;
    var data = gauss_data(mus[curve_id], sigmas[curve_id], -10, 10);
    gauss_curves[curve_id]
      .datum(data)
      .transition()
      .duration(0)
      .attr("d",  d3.line()
        .curve(d3.curveBasis)
        .x(function(d) { return x(d.x); })
        .y(function(d) { return y(d.y); })
      );
}

function updateMu(mu) {
    updateCurve(mu, sigmas[curve_id]);
}

function updateSigma(sigma_sq) {
    reset();
    updateCurve(mus[curve_id], Math.sqrt(sigma_sq));
}

function trivialRound(x) { return x; }

var mu_x = d3.scaleLinear()
    .domain([-5, 5])
    .range([0, width * 0.4])
    .clamp(true);
    
var sigma_x = d3.scaleLinear()
    .domain([0.15, 2])
    .range([0, width * 0.4])
    .clamp(true);
    
var muHandler = createSlider(svg, updateMu, mu_x, margin.left, 1.7 * fig_height, "", "#A9A750", mus[curve_id], trivialRound);

var sigmaHandler = createSlider(svg, updateSigma, sigma_x, margin.left, 1.9 * fig_height, "", "#A9A750", sigmas[curve_id], trivialRound);

d3.select("#asymptotic_test")
  .append("div")
  .text("\\(\\mu_i \\)")
  .style("font-size", "17px")
  .attr("font-family", "Arvo")
  .style("position", "absolute")
  .style("left", margin.left + "px")
  .style("top", 1.7 * fig_height + 15 + "px");
  
d3.select("#asymptotic_test")
  .append("div")
  .text("\\(\\sigma_i^2 \\)")
  .style("font-size", "17px")
  .attr("font-family", "Arvo")
  .style("position", "absolute")
  .style("left", margin.left + "px")
  .style("top", 1.9 * fig_height + 15 + "px");
  
d3.csv("../../../../assets/chi_sf.csv").then(data => {
  const quantiles = [3.84, 5.99, 7.81, 9.49];
  
  var chi_curve0 = svg
    .append('g')
    .append("path")
      .datum(data)
      .attr("fill", "#348ABD")
      .attr("border", 0)
      .attr("opacity", ".8") 
      .attr("stroke", "currentColor")
      .attr("stroke-width", 1)
      .attr("stroke-linejoin", "round")
      .attr("d",  d3.line()
        .curve(d3.curveBasis)
          .x(function(d) { return xRight(d["chi_x0_1"]); })
          .y(function(d) { return yRight(d["chi_y0_1"]); })
      );
      
  var chi_curve1 = svg
    .append('g')
    .append("path")
      .datum(data)
      .attr("fill", "#348ABD")
      .attr("border", 0)
      .attr("opacity", ".2")
      .attr("stroke", "currentColor")
      .attr("stroke-width", 1)
      .attr("stroke-linejoin", "round")
      .attr("d",  d3.line()
        .curve(d3.curveBasis)
          .x(function(d) { return xRight(d["chi_x1_1"]); })
          .y(function(d) { return yRight(d["chi_y1_1"]); })
      );
      
  var u_dot = svg.append('g')
     .selectAll("dot")
     .data([{'x': xRight(quantiles[0]), 'y': yRight(0.05)}])
     .enter()
     .append("circle")
       .attr("cx", function (d) { return d.x; } )
       .attr("cy", function (d) { return d.y; } )
       .attr("r", 4)
       .style("fill", "#fff")
       .attr("stroke", "#348ABD")
       .attr("stroke-width", 2);
     
  function updateChi(n) {
    chi_curve0
      .datum(data)
      .transition()
      .duration(1200)
      .attr("d",  d3.line()
        .curve(d3.curveBasis)
          .x(function(d) { return xRight(d["chi_x0_" + n]); })
          .y(function(d) { return yRight(d["chi_y0_" + n]); })
      );
      
    chi_curve1
      .datum(data)
      .transition()
      .duration(1200)
      .attr("d",  d3.line()
        .curve(d3.curveBasis)
          .x(function(d) { return xRight(d["chi_x1_" + n]); })
          .y(function(d) { return yRight(d["chi_y1_" + n]); })
      );
    
    u_dot
     .transition()
     .duration(1200)
       .attr("cx", function (d) { return xRight(quantiles[n-1]); } )
       .attr("cy", function (d) { return yRight(0.05); } )
  }
  

  d3.select("#add-button").on("click", function() {
    if (gauss_curves.length < colors.length) {
        var mu = 10 * Math.random() - 5;
        var sigma = 1;
        addCurve(mu, sigma);
        muHandler.attr("cx", mu_x(mus[curve_id]));
        sigmaHandler.attr("cx", sigma_x(sigmas[curve_id]));
        updateChi(gauss_curves.length - 1);
    }
  });
  
  d3.select("#delete-button").on("click", function() {
    if (gauss_curves.length > 2) {
        reset();
        booked_color = gauss_curves[curve_id].attr("fill");
        booked_color_id = booked_colors.indexOf(booked_color);
        booked_colors.splice(booked_color_id, 1);
        gauss_curves[curve_id].remove();
       
        mus.splice(curve_id, 1);
        sigmas.splice(curve_id, 1);
        gauss_curves.splice(curve_id, 1);
        
        curve_id = Math.max(0, curve_id - 1);
        gauss_curves[curve_id]
	         .transition()
	         .attr("opacity", ".8");
	         
        muHandler.attr("cx", mu_x(mus[curve_id]));
        sigmaHandler.attr("cx", sigma_x(sigmas[curve_id]));
        updateChi(gauss_curves.length - 1);
        
        for (var i = 0; i < gauss_curves.length; i += 1) {
          const k = i;
          gauss_curves[i]
                      .on('mouseover', function() {
                          d3.select(this)
                            .transition()
                            .attr("opacity", ".8");
	                   })
	                   .on('mouseout', function() {
	                       d3.select(this)
	                         .transition()
	                         .attr("opacity", (curve_id == k) ? ".8" : ".2");
	                       })
	                   .on('click', function() {
	                       gauss_curves[curve_id]
	                         .transition()
	                         .attr("opacity", ".2");
	                       curve_id = k;
	                       d3.select(this)
	                         .transition()
	                         .attr("opacity", ".8");
	                       muHandler.attr("cx", mu_x(mus[k]));
	                       sigmaHandler.attr("cx", sigma_x(sigmas[k]));
	                   })
        }
    }
  });

});

var avg_dur = 1200;
function sample() {
      clearSamples();
      var random_samples = [];
      var avgs = [];
      std_avg = 0;
      for (var k = 0; k < gauss_curves.length; k += 1) {
          random_samples.push([]);
          smpl_dots.push([]);
          var color = gauss_curves[k].attr("fill");
          const ystd = -0.1 * (k + 2);
          var avg = 0;
          for (var i = 0; i < n; i += 1) {
              random_samples[k].push(mus[k] + sigmas[k] * randn_bm());
              smpl_dots[k].push(svg.append('g')
                                .selectAll("dot")
                                .data([{x: random_samples[k][i], y: 1}])
                                .enter()
                                .append("circle")
                                .attr("cx", function (d) { return x(d.x); } )
                                .attr("cy", function (d) { return y(d.y); } )
                                .attr("r", 3)
                                .style("fill", color)
                                .attr("stroke", "black")
                                .attr("stroke-width", 1));
          
              smpl_dots[k][i]
                .transition()
                .duration(avg_dur)
                .attr("cx", function (d) { return x(random_samples[k][i]); } )
                .attr("cy", function (d) { return y(ystd); } );
              
              avg += random_samples[k][i];
          }
          avg /= n;
          avgs.push(avg);
	      
          var std = 0;
          for (var i = 0; i < n; i += 1) {
              std += (random_samples[k][i] - avgs[k]) ** 2;
          }
          std /= n;
          stds.push(std);
          std_avg += std;
	      
	   }
      
      std_avg /= gauss_curves.length;
                           
      var T_n = gauss_curves.length * Math.log(std_avg);
      for (var i = 0; i < gauss_curves.length; i += 1) {
        T_n -= Math.log(stds[i]);
      }
      T_n *= n;
           
      var tn_dot = svg.append('g')
        .selectAll("dot")
        .data([{x: T_n, y: 0}])
        .enter()
        .append("circle")
        .attr("cx", function (d) { return xRight(d.x); } )
        .attr("cy", function (d) { return yRight(d.y); } )
        .attr("r", 3)
        .style("fill", "#348ABD")
        .attr("stroke", "black")
        .attr("stroke-width", 1)
        .attr("opacity", 0);
     
     tn_dot
         .transition()
         .duration(avg_dur)
         .attr("opacity", "1");
     
     tn_dots.push(tn_dot);
     
     updateSiTexts();
     updateStatTexts();
}

d3.select("#sample-button-brt").on("click", function() { sample(); });

function clearSamples() {
  for (var k = 0; k < smpl_dots.length; k += 1) {
    for (var i = 0; i < smpl_dots[k].length; i += 1) {
      smpl_dots[k][i].remove();
    }
  }
  smpl_dots = [];
  stds = [];
  clearSiTexts();
  clearStatTexts();
}

function reset() {
  clearSamples();
  for (var i = 0; i < tn_dots.length; i += 1) {
    tn_dots[i].remove();
  }
  tn_dots = [];
}

d3.select("#reset-button-2").on("click", function() { reset(); });

d3.select("#asymptotic_test")
  .append("div")
  .text("\\(\\alpha \\)")
  .style("font-size", "15px")
  .attr("font-family", "Arvo")
  .style("position", "absolute")
  .style("left", 1.2 * fig_width + 5 + margin.left + "px")
  .style("top", 15 + "px");
  
d3.select("#asymptotic_test")
  .append("div")
  .text("\\(\\chi^2_{r-1, 1-\\alpha} \\)")
  .style("font-size", "15px")
  .attr("font-family", "Arvo")
  .style("position", "absolute")
  .style("left", 1.9 * fig_width + 5 + margin.left + "px")
  .style("top", fig_height + 15 + "px");


var labels_x = 250;
var labels_y = 0;

svg.append("path")
   .attr("stroke", "#348ABD")
   .attr("stroke-width", 4)
   .attr("opacity", ".8")
   .datum([{x: labels_x + fig_width, y: labels_y}, {x: labels_x + fig_width + 25, y: labels_y}])
   .attr("d",  d3.line()
       .x(function(d) { return d.x; })
       .y(function(d) { return d.y; }));
       
svg.append("path")
   .attr("stroke", "currentColor")
   .attr("stroke-width", 1)
   .datum([{x: labels_x + fig_width, y: labels_y - 2}, {x: labels_x + fig_width + 25, y: labels_y - 2}])
   .attr("d",  d3.line()
       .x(function(d) { return d.x; })
       .y(function(d) { return d.y; }));
       
d3.select("#asymptotic_test")
  .append("div")
  .text("\\(1-F_{\\chi^2_{r-1}}(x) \\)")
  .style('color', '#348ABD')
  .style("font-size", "15px")
  .attr("font-family", "Arvo")
  .style("position", "absolute")
  .style("left", labels_x + fig_width + margin.left + 30 + "px")
  .style("top", labels_y + 15 + "px");
  
svg.append('g')
     .selectAll("dot")
     .data([{'x': labels_x + fig_width + 13, 'y': labels_y + 20}])
     .enter()
     .append("circle")
       .attr("cx", function (d) { return d.x; } )
       .attr("cy", function (d) { return d.y; } )
       .attr("r", 3)
        .style("fill", "#348ABD")
        .attr("stroke", "black")
        .attr("stroke-width", 1);

d3.select("#asymptotic_test")
  .append("div")
  .text("\\(T_n \\)")
  .style('color', '#348ABD')
  .style("font-size", "15px")
  .attr("font-family", "Arvo")
  .style("position", "absolute")
  .style("left", labels_x + fig_width + margin.left + 30 + "px")
  .style("top", labels_y + 35 + "px");

svg
  .append("text")
  .attr("text-anchor", "start")
  .attr("y", 5)
  .attr("x", 0.7 * fig_width )
  .attr("font-family", "Arvo")
  .text("X distributions")
  .style("font-size", "14px")
  .style("fill", "currentColor");
  
d3.select("#asymptotic_test")
  .append("div")
  .text("\\(\\hat{\\sigma}^2 \\)")
  .style("font-size", "12px")
  .attr("font-family", "Arvo")
  .style("position", "absolute")
  .style("left", 1.2 * fig_width + 100 + margin.left + "px")
  .style("top", fig_height + 58 + "px");
  
d3.select("#asymptotic_test")
  .append("div")
  .text("\\(T_n \\)")
  .style("font-size", "12px")
  .attr("font-family", "Arvo")
  .style("position", "absolute")
  .style("left", 1.2 * fig_width + 100 + margin.left + "px")
  .style("top", fig_height + 79 + "px");
  
d3.select("#asymptotic_test")
  .append("div")
  .text("\\(\\hat{s}_i^2: \\)")
  .style("font-size", "12px")
  .attr("font-family", "Arvo")
  .style("position", "absolute")
  .style("left", 1.2 * fig_width + 25 + margin.left + "px")
  .style("top", fig_height + 58 + "px");

svg
  .append("text")
  .attr("text-anchor", "start")
  .attr("y", 1.8 * fig_height)
  .attr("x", 1.2 * fig_width + 5)
  .attr("font-family", "Arvo")
  .text("Significance level: 0.05")
  .style("font-size", "14px")
  .style("fill", "#348ABD");

function createSiTexts() {
  for (var i = 0; i < 5; i += 1) {
    si_texts.push(svg.append("text")
                     .attr("text-anchor", "start")
                     .attr("y", y(-0.1 * (i + 2.2)))
                     .attr("x", 1.2 * fig_width + 50)
                     .attr("font-family", "Arvo")
                     .style("font-size", "12px")
                     .text(""));
  }
}

function updateSiTexts() {
  for (var i = 0; i < gauss_curves.length; i += 1) {
    var color = gauss_curves[i].attr("fill");
    var rounded_si = Math.round(1000 * stds[i]) / 1000;
    si_texts[i]
      .transition()
      .duration(avg_dur)
      .text(rounded_si.toString())
      .style("fill", color);
  }
}

function clearSiTexts() {
 for (var i = 0; i < si_texts.length; i += 1) {
    si_texts[i]
      .transition()
      .duration(avg_dur)
      .text("");
  }
}

function createStatTexts() {
  sigma_text = svg.append("text")
                     .attr("text-anchor", "start")
                     .attr("y", y(-0.22))
                     .attr("x", 1.2 * fig_width + 115)
                     .attr("font-family", "Arvo")
                     .style("font-size", "12px")
                     .text("")
                     .style("fill", "#348ABD");
                     
  t_text = svg.append("text")
                     .attr("text-anchor", "start")
                     .attr("y", y(-0.32))
                     .attr("x", 1.2 * fig_width + 115)
                     .attr("font-family", "Arvo")
                     .style("font-size", "12px")
                     .text("")
                     .style("fill", "#348ABD");
}

function updateStatTexts() {
    var sigma_num = Math.round(1000 * std_avg) / 1000;
    sigma_text
      .transition()
      .duration(avg_dur)
      .text("= " + sigma_num);
    
    var t_num = xRight.invert(tn_dots[tn_dots.length - 1].attr("cx"));
    t_num = Math.round(1000 * t_num) / 1000;
    t_text
      .transition()
      .duration(avg_dur)
      .text("= " + t_num);
}

function clearStatTexts() {
  if (sigma_text != null) {
    sigma_text
      .transition()
      .text("");
    t_text
      .transition()
      .text("");
  }
}

createSiTexts();
createStatTexts();

}

asymptotic_test();

</script>

![](.)
*Fig. 3. Visualization of Bartlett test. Up to five normally distributed samples can be added, each with $n_i=30$. Significance level $\alpha$ is fixed at $0.05$. Choose different variations of $\sigma_i^2$ to observe how it affects test statistic $T_n$.* 


Take another example: suppose we have two discrete variables $A$ and $B$ (e.g. such as gender, age, education or income), where $A$ can take $r$ values and $B$ can take $s$ values. Further suppose that $n$ individuals are randomly sampled. A **contingency table** can be created to display the joint sample distribution of $A$ and $B$.

|          | $1$          | $\cdots$ | $s$           | Sum |
| -------- | ------------ | -------- | ------------- | ------- |
| $1$      | $X_{11}$     | $\cdots$ | $X_{1s}$      | $X_{1\cdot} = \sum_{j=1}^n X_{1j}$ |
| $\vdots$ | $\vdots$     | $\vdots$ | $\vdots$      | $\vdots$ |
| $r$      | $X_{r1}$     | $\cdots$ | $X_{rs}$      | $X_{r\cdot}$ |
| **Sum**  | $X_{\cdot1}$ | $\cdots$ | $X_{\cdot s}$ | $n$ |

We model vector $X$ with multinomial distribution:

$$(X_1, \dots X_n)^T \sim \mathcal{M}(n, p_{11}, \dots, p_{rs}),$$

where $\sum_{ij} p_{ij} = 1$. Joint density is

$$ f_n(x^{(n)}, p) = P_p(X_{ij}=x_{ij}) = \frac{n!}{\prod_{i,j=1}^{r,s} x_{ij}!} \prod_{i,j=1}^{r,s} (p_{ij})^{x_{ij}}, $$

where $x_{ij} = \lbrace 0, \dots, n\rbrace$ and $\sum_{i,j=1}^{r,s} x_{ij} = n$. Maximum-likelihood estimator is

$$\hat{p}_{ij} = \frac{X_{ij}}{n}$$

(in analogy to binomial distribution) and

$$f_n(X^{(n)}, \hat{p}) = \frac{n!}{\prod_{i,j=1}^{r,s} X_{ij}!} \prod_{i,j=1}^{r,s} \Big(\frac{X_{ij}}{n}\Big)^{X_{ij}}$$

Suppose we want to test independence between $A$ and $B$:

$$ H\colon p_{ij} = p_i q_j \ \forall i,j \quad \text{vs} \quad K\colon p_{ij} \neq p_i q_j \text{ for some } i \neq j,$$

where $p_i = p_{i \cdot} = \sum_{j=1}^{s}p_{ij}$ and $q_j = p_{\cdot, j} = \sum_{i=1}^{r}p_{ij}$. Here $d = rs-1$, $c = r + s - 2$ and $d-c = (r-1)(s-1)$. If null hypothesis is true, then 

$$f_n(X^{(n)}, p, q) = \frac{n!}{\prod_{i,j=1}^{r,s} X_{ij}!} \prod_{i,j=1}^{r,s} (p_i q _j)^{X_{ij}} = \frac{n!}{\prod_{i,j=1}^{r,s} X_{ij}!} \prod_{i}^{r} p_i^{X_{i \cdot}} \prod_{j=1}^{s} q_j ^ {X_{\cdot j}}.  $$

Maximum-likelihood estimators are

$$\hat{p}_i = \frac{X_{i \cdot}}{n} \quad \text{and} \quad \hat{q}_j = \frac{X_{\cdot j}}{n}, $$

and likelihood function is

$$f_n(X^{(n)}, \hat{p}, \hat{q}) = \frac{n!}{\prod_{i,j=1}^{r,s} X_{ij}!} \prod_{i,j=1}^{r,s} \Big( \frac{X_{i \cdot} X_{\cdot j}}{n^2} \Big)^{X_{ij}}. $$

We get

$$T_n = -2 \log \lambda(X^{(n)}) = 2 \sum_{i=1}^r  \sum_{j=1}^s X_{ij} \log \Big( \frac{nX_{ij}}{ X_{i \cdot} X_{\cdot j} } \Big)$$

and 

$$\varphi_n(X^{(n)}) =
   	\left \{
   	\begin{array}{cl}
   	1, & T_n > \mathcal{X}_{(r-1)(s-1), 1-\alpha}^2, \\
   	0, & \text{otherwise},
   	\end{array}
   	\right.$$
   	
which is called **chi-square independence test**. Using Taylor expansion with Law of Large Number we can get asymptotic equivalent

$$\tilde{T}_n = \sum_{i=1}^{r} \sum_{j=1}^s \frac{\Big(X_{ij} -\frac{X_{i \cdot} X_{\cdot j}}{n}\Big)^2}{\frac{X_{i \cdot} X_{\cdot j}}{n}}.$$

Usually,

$$V_n = \sqrt{\frac{\tilde{T}_n}{n (\min(r, s) - 1)}}  $$

is used as dependency measure between $A$ and $B$, because under both null hypothesis and alternative convergence takes place

$$V_n^2  \xrightarrow{\mathbb{P}} \frac{1}{\min(r, s) - 1}\sum_{i=1}^{r} \sum_{j=1}^s \frac{(p_{ij} - p_{i \cdot}p_{\cdot j} )^2}{p_{i \cdot}p_{\cdot j}}$$

<div id="htmp"></div>
<script>

d3.select("#htmp")
  .style("position", "relative");
  
function plt_heatmap() {

var margin = {top: 30, right: 30, bottom: 5, left: 50},
  width = 750 - margin.left - margin.right,
  height = 300 - margin.top - margin.bottom,
  fig_width = 300, fig_height = 240;

var svg = d3.select("#htmp")
.append("svg")
  .attr("width", width + margin.left + margin.right)
  .attr("height", height + margin.top + margin.bottom)
.append("g")
  .attr("transform",
        "translate(" + margin.left + "," + margin.top + ")");

var r = 4, s = 5;
var rows = [], columns = [];
for (var i = 0; i < r; i += 1) {
    rows.push(i + 1);
}
for (var i = 0; i < s; i += 1) {
    columns.push(i + 1);
}

var x = d3.scaleBand()
  .range([ 0, fig_width ])
  .domain(columns)
  .padding(0.01);
  
svg.append("g")
  .attr("transform", "translate(0," + fig_height + ")")
  .call(d3.axisBottom(x))
  .selectAll(".tick text")
  .attr("font-family", "Arvo");

var y = d3.scaleBand()
  .range([ fig_height, 0 ])
  .domain(rows)
  .padding(0.01);
  
svg.append("g")
  .call(d3.axisLeft(y))
  .selectAll(".tick text")
  .attr("font-family", "Arvo");

var xRight = d3.scaleLinear()
          .domain([0, 50])
          .range([1.2 * fig_width, 2.2 * fig_width]);
            
svg.append("g")
   .attr("transform", "translate(0," + 0.5 * fig_height + ")")
   .call(d3.axisBottom(xRight))
   .selectAll(".tick text")
   .attr("font-family", "Arvo");


var yRight = d3.scaleLinear()
          .range([0.5 * fig_height, 0.2 * fig_height])
          .domain([0, 1]);
            
svg.append("g")
    .attr("transform", "translate(" + 1.2 * fig_width + ",0)")
    .call(d3.axisLeft(yRight).ticks(1))
    .selectAll(".tick text")
    .attr("font-family", "Arvo");

var color = d3.scaleLinear()
  .range(["white", "#65AD69"])
  .domain([1, 25]);

var data = [];
for (var i = 0; i < s; i += 1) {
  for (var j = 0; j < r; j += 1) {
    var value = 1 + Math.round(19 * Math.random());
    data.push({'i': i + 1, 'j': j + 1,
               'value': value});
  }
}

var t_n = 0, v_n = 0;
var tn_text = svg.append("text")
               .attr("text-anchor", "start")
               .attr("y", 0.8 * fig_height + 3)
               .attr("x", 1.4 * fig_width + 20)
               .attr("font-family", "Arvo")
               .attr("font-weight", 700)
               .style("font-size", "13px")
               .style("fill", "#348ABD");
             
               
var vn_text = svg.append("text")
               .attr("text-anchor", "start")
               .attr("y", 0.8 * fig_height + 3)
               .attr("x", 1.8 * fig_width + 20)
               .attr("font-family", "Arvo")
               .attr("font-weight", 700)
               .style("font-size", "13px")
               .style("fill", "#EDA137");

function updateTn() {
   t_n = 0;
   v_n = 0;
   
   var n = 0, xi = [0, 0, 0, 0], xj = [0, 0, 0, 0, 0];
   for (var i = 0; i < s; i += 1) {
     for (var j = 0; j < r; j += 1) {
        var xij = data[i * r + j]['value'];
		  n += xij;
		  xj[i] += xij;
		  xi[j] += xij;
	  }
	}
	
	for (var i = 0; i < s; i += 1) {
	  for (var j = 0; j < r; j += 1) {
	    var xij = data[i * r + j]['value'];
	    if (xij > 0) {	
	      t_n += xij * Math.log(n * xij / (xi[j] * xj[i]));
	    }
	    if ((xi[j] > 0) && (xj[i] > 0)) {
	      v_n += (xij - xi[j] * xj[i] / n) ** 2 / (xi[j] * xj[i] / n);
	    }
	  }
	}
	
	v_n = Math.sqrt(v_n / ((Math.min(r, s) - 1) * n) );
	
	tn_text
    .transition()
    .duration(500)
    .text(" = " + Math.round(1000 * t_n) / 1000);
    
	vn_text
    .transition()
    .duration(500)
    .text(" = " + Math.round(1000 * v_n) / 1000);
    
   tn_dot
    .data([{'x': xRight(t_n), 'y': yRight(t_n > c ? 1 : 0)}])
    .transition()
    .duration(500)
    .attr("cx", function (d) { return d.x; } )
    .attr("cy", function (d) { return d.y; } )
}
                   
  var tooltip = d3.select("#htmp")
    .append("div")
    .style("opacity", 0)
    .attr("class", "tooltip")
    .style("background-color", "white")
    .style("border", "solid")
    .style("border-width", "2px")
    .style("border-radius", "10px")
    .style("padding", "5px");
    
  var mouseover = function(d) {
    tooltip
      .style("opacity", 1);
      
    d3.select(this)
      .style("stroke", "black")
      .style("opacity", 1);
  };
  
  var mousemove = function(event, d) {
    tooltip
      .html(d.value)
      .style("left", (d3.pointer(event)[0] + 70) + "px")
      .style("top", (d3.pointer(event)[1]) + "px");
  };
  
  var mouseleave = function(d) {
    tooltip
      .style("opacity", 0);
      
    d3.select(this)
      .style("stroke", "none")
      .style("opacity", 0.8);
  };
  
  var onclick = function(event, d) {
    if (event.ctrlKey || event.metaKey) {
      d['value'] = Math.max(d['value'] - 1, 0);
    }
    else {
      d['value'] += 1;
    }
    updateTn();
    
    d3.select(this)
      .style("fill", function(d) { return color(d['value'])} );
      
    mousemove(event, d);
  };
 
var c = 21.026;
var phi_data_0 = [{'x': 0, 'y': 0}, {'x': c, 'y': 0}];
var phi_data_1 = [{'x': c, 'y': 1}, {'x': 50, 'y': 1}];
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
          .x(function(d) { return xRight(d['x']); })
          .y(function(d) { return yRight(d['y']); })
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
          .x(function(d) { return xRight(d['x']); })
          .y(function(d) { return yRight(d['y']); })
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
          .x(function(d) { return xRight(d['x']); })
          .y(function(d) { return yRight(d['y']); })
   );

var phi_dot = svg.append('g')
   .selectAll("dot")
   .data([{'x': xRight(c), 'y': yRight(1)}])
   .enter()
   .append("circle")
     .attr("cx", function (d) { return d.x; } )
     .attr("cy", function (d) { return d.y; } )
     .attr("r", 4)
     .style("fill", "#fff")
     .attr("stroke", "#348ABD")
     .attr("stroke-width", 2);
     
var tn_dot = svg.append('g')
    .selectAll("dot")
    .data([{'x': xRight(t_n), 'y': yRight(t_n > c ? 1 : 0)}])
    .enter()
    .append("circle")
    .attr("cx", function (d) { return d.x; } )
    .attr("cy", function (d) { return d.y; } )
    .attr("r", 3)
    .style("fill", "#348ABD")
    .attr("stroke", "black")
    .attr("stroke-width", 1)
    .attr("opacity", 1);    
 
updateTn();
                  
d3.select("#htmp")
  .append("div")
  .text("\\(X_{i \\cdot} \\)")
  .style("font-size", "15px")
  .attr("font-family", "Arvo")
  .style("position", "absolute")
  .style("left", margin.left - 35 + "px")
  .style("top", 0.5 * fig_height + margin.top - 10 + "px");
  
d3.select("#htmp")
  .append("div")
  .text("\\(X_{\\cdot j} \\)")
  .style("font-size", "15px")
  .attr("font-family", "Arvo")
  .style("position", "absolute")
  .style("left", 0.5 * fig_width + margin.left - 5 + "px")
  .style("top", fig_height + margin.top + 25 + "px");
  
d3.select("#htmp")
  .append("div")
  .text("\\(T_n \\)")
  .style("font-size", "15px")
  .attr("font-family", "Arvo")
  .style("position", "absolute")
  .style("left", 2.25 * fig_width + margin.left + "px")
  .style("top", 0.6 * fig_height + "px");

 d3.select("#htmp")
  .append("div")
  .text("\\(T_n \\)")
  .style('color', '#348ABD')
  .style("font-size", "15px")
  .attr("font-family", "Arvo")
  .style("position", "absolute")
  .style("left", 1.4 * fig_width + margin.left + "px")
  .style("top", 0.8 * fig_height + 15 + "px");
  
d3.select("#htmp")
  .append("div")
  .text("\\(\\chi_{12, 0.95}^2 \\)")
  .style('color', '#348ABD')
  .style("font-size", "15px")
  .attr("font-family", "Arvo")
  .style("position", "absolute")
  .style("left", xRight(c) + 40 + "px")
  .style("top", 0.5 * fig_height + 45 + "px");
  
d3.select("#htmp")
  .append("div")
  .text("\\(V_n \\)")
  .style('color', '#EDA137')
  .style("font-size", "15px")
  .attr("font-family", "Arvo")
  .style("position", "absolute")
  .style("left", 1.8 * fig_width + margin.left + "px")
  .style("top", 0.8 * fig_height + 15 + "px");
  
  var heatmap = svg.selectAll()
      .data(data)
      .enter()
      .append("rect")
      .attr("x", function(d) { return x(d['i']) })
      .attr("y", function(d) { return y(d['j']) })
      .attr("rx", 8)
      .attr("ry", 8)
      .attr("width", x.bandwidth() )
      .attr("height", y.bandwidth() )
      .style("fill", function(d) { return color(d['value'])} )
      .style("stroke-width", 1)
      .style("stroke", "none")
      .style("opacity", 0.8)
    .on("mouseover", mouseover)
    .on("mousemove", mousemove)
    .on("mouseleave", mouseleave)
    .on("click", onclick);
    
  
}

plt_heatmap();

</script>

![](.)
*Fig. 4. Visualization for chi-square independence test with $r=4$ and $s=5$. Significance level $\alpha$ is fixed at $0.05$. Click on the cell of contingency table to increase $X_{ij}$ value, and CTRL + click to decrease (or ⌘ + click for Mac OS).*