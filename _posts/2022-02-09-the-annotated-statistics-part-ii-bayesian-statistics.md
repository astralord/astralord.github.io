---
layout: post
title: 'The Annotated Statistics. Part II: Bayesian Statistics'
date: 2022-02-09 03:22 +0800
categories: [Statistics]
tags: [statistics, parameter-estimation, bayesian-statistics, bayes-estimator, minimax-estimator]
math: true
enable_d3: true
---

> Part II introduces different approach to parameters estimation called Bayesian interpretation.

We noted in the previous part that it is extremely unlikely to get a uniformly best estimator. An alternative way to compare risk functions is to integrate or calculate the maximum.

Let's think of parameter $\vartheta$ as a realization of random variable $\theta$ with distribution $\pi$. We call $\pi$ - **a prior distribution** for $\vartheta$. For an estimator $g \in \mathcal{K}$ and its risk $R(\cdot, g)$

$$ R(\pi, g) = \int_{\Theta} R(\theta, g) \pi(d \vartheta) $$

is called the **Bayes risk of $g$ with respect to $\pi$**. An estimator $\tilde{g} \in \mathcal{K}$ is called a **Bayes estimator** if it minimizes the Bayes risk over all estimators, that is

$$ R(\pi, \tilde{g}) = \inf_{g \in \mathcal{K}} R(\pi, g). $$

The right hand side of the equation above is call the **Bayes risk**.

<h3>D3.js Bar Chart Using YAML and Jekyll</h3>
<p>This is a D3.js bar chart that is driven from dynamically generated JSON, from YAML stored in the _data folder within this Github Pages repository running Jekyll.</p>
<div id="chart"></div>
<p>The YAML can be found in <a href="https://github.com/kinlane/d3-js-using-yaml-jekyll/tree/gh-pages/_data" target="_blank">_data/bar-chart.yaml</a>, but is transformed into the JSON needed for this chart, using <a href="https://github.com/kinlane/d3-js-using-yaml-jekyll/blob/gh-pages/data/bar-chart.json" target="_blank">/data/bar-chart.json</a>, demonstrating how YAML can be used to drive visualizations on Github.</p>
<style>

/* Bar Chart */
.bar {
  fill: steelblue;
}

.bar:hover {
  fill: brown;
}

.axis {
  font: 10px sans-serif;
}

.axis path,
.axis line {
  fill: none;
  stroke: #000;
  shape-rendering: crispEdges;
}

.x.axis path {
  display: none;
}

</style>

<script>

var margin = {top: 20, right: 20, bottom: 30, left: 40},
    width = 750 - margin.left - margin.right,
    height = 500 - margin.top - margin.bottom;

var x = d3.scale.ordinal()
    .rangeRoundBands([0, width], .1);

var y = d3.scale.linear()
    .range([height, 0]);

var xAxis = d3.svg.axis()
    .scale(x)
    .orient("bottom");

var yAxis = d3.svg.axis()
    .scale(y)
    .orient("left")
    .ticks(10);

var svg = d3.select("#chart").append("svg")
    .attr("width", width + margin.left + margin.right)
    .attr("height", height + margin.top + margin.bottom)
  .append("g")
    .attr("transform", "translate(" + margin.left + "," + margin.top + ")");

d3.json("/data/bar-chart.json", function(error, data) {
  if (error) throw error;

  x.domain(data.map(function(d) { return d.letter; }));
  y.domain([0, d3.max(data, function(d) { return d.frequency; })]);

  svg.append("g")
      .attr("class", "x axis")
      .attr("transform", "translate(0," + height + ")")
      .call(xAxis);

  svg.append("g")
      .attr("class", "y axis")
      .call(yAxis)
    .append("text")
      .attr("transform", "rotate(-90)")
      .attr("y", 6)
      .attr("dy", ".71em")
      .style("text-anchor", "end")
      .text("Number of Years");

  svg.selectAll(".bar")
      .data(data)
    .enter().append("rect")
      .attr("class", "bar")
      .attr("x", function(d) { return x(d.letter); })
      .attr("width", x.rangeBand())
      .attr("y", function(d) { return y(d.frequency); })
      .attr("height", function(d) { return height - y(d.frequency); });
});

function type(d) {
  d.frequency = +d.frequency;
  return d;
}
</script>
