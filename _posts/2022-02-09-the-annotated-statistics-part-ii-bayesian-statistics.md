---
layout: post
title: 'The Annotated Statistics. Part II: Bayesian Statistics'
date: 2022-02-09 03:22 +0800
categories: [Statistics]
tags: [statistics, parameter-estimation, bayesian-statistics, bayes-estimator, minimax-estimator]
math: true
enable_d3: true
---
<script src="//code.jquery.com/jquery.js"></script>
<style>

.node {
  stroke: #fff;
  stroke-width: 1.5px;
}

.link {
  stroke: #999;
  stroke-opacity: .6;
}

</style>


> Part II introduces different approach to parameters estimation called Bayesian interpretation.

We noted in the previous part that it is extremely unlikely to get a uniformly best estimator. An alternative way to compare risk functions is to integrate or calculate the maximum.

Let's think of parameter $\vartheta$ as a realization of random variable $\theta$ with distribution $\pi$. We call $\pi$ - **a prior distribution** for $\vartheta$. For an estimator $g \in \mathcal{K}$ and its risk $R(\cdot, g)$

$$ R(\pi, g) = \int_{\Theta} R(\theta, g) \pi(d \vartheta) $$

is called the **Bayes risk of $g$ with respect to $\pi$**. An estimator $\tilde{g} \in \mathcal{K}$ is called a **Bayes estimator** if it minimizes the Bayes risk over all estimators, that is

$$ R(\pi, \tilde{g}) = \inf_{g \in \mathcal{K}} R(\pi, g). $$

<!-- Create a div where the graph will take place -->
<div id="chi_t_plt"></div> 

<!-- Load d3.js -->
<script src="//d3js.org/d3.v3.min.js"></script>

<!-- Add a slider -->
<input type="range" name="ddof_slider" id=ddof_slider min="1" max="12" value="5">


The right hand side of the equation above is call the **Bayes risk**.
???

<script>

// set the dimensions and margins of the graph
var margin = {top: 10, right: 350, bottom: 30, left: 30},
    width = 600 - margin.left - margin.right,
    height = 200 - margin.top - margin.bottom;

// append the svg object to the body of the page
var chi_svg = d3.select("#chi_t_plt")
  .append("svg")
    .attr("width", width + margin.left + margin.right)
    .attr("height", height + margin.top + margin.bottom)
  .append("g")
    .attr("transform",
          "translate(" + margin.left + "," + margin.top + ")");

// set the dimensions and margins of the graph

var margin = {top: 0, right: 10, bottom: 35, left: 300}
    
// append the svg object to the body of the page
var t_svg = chi_svg
  .append("svg")
    .attr("width", width + margin.left + margin.right)
    .attr("height", height + margin.top + margin.bottom)
  .append("g")
    .attr("transform",
          "translate(" + margin.left + "," + margin.top + ")");

// get the data
d3.csv("../../../../assets/chi-t.csv", function(error, data) {
  if (error) throw error;

  // add the x Axis
  var chi_x = d3.scaleLinear()
            .domain([-0, 40])
            .range([0, width]);
            
  chi_svg.append("g")
      .attr("transform", "translate(0," + height + ")")
      .call(d3.axisBottom(chi_x));

  // add the x Axis
  var t_x = d3.scaleLinear()
            .domain([-20, 20])
            .range([0, width]);
            
  t_svg.append("g")
      .attr("transform", "translate(0," + height + ")")
      .call(d3.axisBottom(t_x));

  // add the y Axis
  var y = d3.scaleLinear()
            .range([height, 0])
            .domain([0, 0.5]);
            
  chi_svg.append("g")
      .call(d3.axisLeft(y));
  
  
  var t_y = d3.scaleLinear()
            .range([height, 5])
            .domain([0, 0.5]);
                
  t_svg.append("g")
      .call(d3.axisLeft(t_y));

  // Plot the area
  var chi_curve = chi_svg
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
          .x(function(d) { return chi_x(d.chi_x); })
          .y(function(d) { return y(d["chi_5"]); })
      );
      
  // Plot the area
  var t_curve = t_svg
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
          .x(function(d) { return t_x(d.t_x); })
          .y(function(d) { return t_y(d["t_5"]); })
      );

  // A function that update the chart when slider is moved?
  function updateChart(n) {
    // update the chart
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

  // Listen to the slider
  d3.select("#ddof_slider").on("change", function(d){
    selectedValue = this.value
    updateChart(selectedValue)
  })
});

</script>


<div id='d3div'></div>

<script>
  
var width = $("#d3div").width(),
    height = 400;

var color = d3.scale.category20();

var force = d3.layout.force()
    .charge(-62)
    .linkDistance(80)
    .size([width, height]);

var svg = d3.select("#d3div").append("svg")
    .attr("width", width)
    .attr("height", height);

d3.json("../../../../assets/jazz_scales_network_minCTs6.json", function(graph) {

  force
      .nodes(graph.nodes)
      .links(graph.links)
      .start();

  var link = svg.selectAll(".link")
      .data(graph.links)
    .enter().append("line")
      .attr("class", "link")
      .style("stroke-width", function(d) { return Math.sqrt(d.value); });

  var node = svg.selectAll(".node")
      .data(graph.nodes)
    .enter().append("circle")
      .attr("class", "node")
      .attr("r", 5)
      .style("fill", function(d) { return color(d.group); })
      .call(force.drag);

  node.append("title")
      .text(function(d) { return d.name; });

  force.on("tick", function() {
    link.attr("x1", function(d) { return d.source.x; })
        .attr("y1", function(d) { return d.source.y; })
        .attr("x2", function(d) { return d.target.x; })
        .attr("y2", function(d) { return d.target.y; });

    node.attr("cx", function(d) { return d.x; })
        .attr("cy", function(d) { return d.y; });
  });
  
});


</script>
