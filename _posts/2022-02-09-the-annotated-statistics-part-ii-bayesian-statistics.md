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

<!-- Load d3.js -->
<script src="https://d3js.org/d3.v4.js"></script>

<!-- Add a slider -->
<input type="range" name="mySlider" id=mySlider min="10" max="100" value="50">

<!-- Create a div where the graph will take place -->
<div id="my_dataviz"></div>

The right hand side of the equation above is call the **Bayes risk**.

<script src="//d3js.org/d3.v3.min.js"></script>
<script>

// set the dimensions and margins of the graph
var margin = {top: 30, right: 30, bottom: 30, left: 50},
    width = 460 - margin.left - margin.right,
    height = 400 - margin.top - margin.bottom;

// append the svg object to the body of the page
var svg = d3.select("#my_dataviz")
  .append("svg")
    .attr("width", width + margin.left + margin.right)
    .attr("height", height + margin.top + margin.bottom)
  .append("g")
    .attr("transform",
          "translate(" + margin.left + "," + margin.top + ")");

// get the data
d3.csv("https://raw.githubusercontent.com/holtzy/data_to_viz/master/Example_dataset/1_OneNum.csv", function(data) {

  // add the x Axis
  var x = d3.scaleLinear()
            .domain([0, 1000])
            .range([0, width]);
  svg.append("g")
      .attr("transform", "translate(0," + height + ")")
      .call(d3.axisBottom(x));

  // add the y Axis
  var y = d3.scaleLinear()
            .range([height, 0])
            .domain([0, 0.01]);
  svg.append("g")
      .call(d3.axisLeft(y));

  // Compute kernel density estimation
  var kde = kernelDensityEstimator(kernelEpanechnikov(7), x.ticks(40))
  var density =  kde( data.map(function(d){  return d.price; }) )

  // Plot the area
  var curve = svg
    .append('g')
    .append("path")
      .attr("class", "mypath")
      .datum(density)
      .attr("fill", "#69b3a2")
      .attr("opacity", ".8")
      .attr("stroke", "#000")
      .attr("stroke-width", 1)
      .attr("stroke-linejoin", "round")
      .attr("d",  d3.line()
        .curve(d3.curveBasis)
          .x(function(d) { return x(d[0]); })
          .y(function(d) { return y(d[1]); })
      );

  // A function that update the chart when slider is moved?
  function updateChart(binNumber) {
    // recompute density estimation
    kde = kernelDensityEstimator(kernelEpanechnikov(7), x.ticks(binNumber))
    density =  kde( data.map(function(d){  return d.price; }) )
    console.log(binNumber)
    console.log(density)

    // update the chart
    curve
      .datum(density)
      .transition()
      .duration(1000)
      .attr("d",  d3.line()
        .curve(d3.curveBasis)
          .x(function(d) { return x(d[0]); })
          .y(function(d) { return y(d[1]); })
      );
  }

  // Listen to the slider?
  d3.select("#mySlider").on("change", function(d){
    selectedValue = this.value
    updateChart(selectedValue)
  })

});


// Function to compute density
function kernelDensityEstimator(kernel, X) {
  return function(V) {
    return X.map(function(x) {
      return [x, d3.mean(V, function(v) { return kernel(x - v); })];
    });
  };
}
function kernelEpanechnikov(k) {
  return function(v) {
    return Math.abs(v /= k) <= 1 ? 0.75 * (1 - v * v) / k : 0;
  };
}

</script>