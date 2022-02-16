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

<!-- Load d3.js -->
<script src="https://d3js.org/d3.v4.min.js"></script>

<!-- Add a slider -->
<input type="range" name="ddof_slider" id=ddof_slider min="1" max="12" value="5">

<!-- Create a div where the graph will take place -->
<div id="chi_t_plt"></div> 

The right hand side of the equation above is call the **Bayes risk**.
12345

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
d3.csv("../assets/chi-t.csv", function(error, data) {
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

<button type='button'>Randomize Data</button>

<script src="https://d3js.org/d3.v3.min.js"></script>
<script>
  
var w = 600;
var h = 250;
var padding = 30;

 // Create dynamic random dataset function
var datagen = function() {
    var dataset = [];
    var numValues = 25;
    var maxRange = Math.random() * 1000;
    for (var i = 0; i < numValues; i++) {
        //create x and y coords
        var xnum = Math.floor(Math.random() * maxRange);
        var ynum = Math.floor(Math.random() * maxRange);
        //add number to array
        dataset.push([xnum, ynum]);
    }
    return dataset;
};
 //creat data
dataset = datagen();

 // Create scale functions
var xScale = d3.scale.linear()
    .domain([0, d3.max(dataset, function(d) {
        return d[0];
    })])
    .range([padding, w - padding * 2]);

var yScale = d3.scale.linear()
    .domain([0, d3.max(dataset, function(d) {
        return d[1];
    })])
    .range([h - padding, padding]);

 // Define Axis
var xAxis = d3.svg.axis()
    .scale(xScale)
    .orient("bottom")
    .ticks(5);
var yAxis = d3.svg.axis()
    .scale(yScale)
    .orient('left')
    .ticks(5);

 // Create svg element 
var svg = d3.select('body')
    .append('svg')
    .attr('width', w)
    .attr('height', h);

 // Create circles
svg.selectAll('circle')
    .data(dataset)
    .enter()
    .append('circle')
    .attr('cx', function(d) {
        return xScale(d[0]);
    })
    .attr('cy', function(d) {
        return yScale(d[1]);
    })
    .attr('r', 4)
    .attr('fill', 'teal');

 // Create axis
svg.append('g') // new group element 
.attr('class', 'x axis')
 //move to bottom
.attr('transform', 'translate(' + 0 + ',' + (h - padding) + ')')
    .call(xAxis);

svg.append('g')
    .attr('class', 'y axis')
 //move left a bit to compensate for padding
.attr('transform', 'translate(' + padding + ',' + 0 + ')')
    .call(yAxis);

 // On click, update with new random data
d3.select('button')
    .on('click', function(d) {
        //renew data
        dataset = datagen();

        //Update scale domains
        xScale.domain([0, d3.max(dataset, function(d) {
            return d[0];
        })]);
        yScale.domain([0, d3.max(dataset, function(d) {
            return d[1];
        })]);

        // Update all circles
        svg.selectAll('circle')
            .data(dataset)
            .transition() // Transition 1
        .duration(1000)
            .ease('circle')
            .each('start', function() {
                d3.select(this)
                    .attr('fill', 'gray')
                    .attr('r', 2);
            })
            .attr('cx', function(d) {
                return xScale(d[0]);
            })
            .attr('cy', function(d) {
                return yScale(d[1]);
            })
            .transition() // Transition 2, equiv to below
        .duration(250)
            .attr('fill', 'teal')
            .attr('r', 4);

        // .each('end', function() {
        //     d3.select(this)
        //         .transition()
        //         .duration(250)
        //         .attr('fill', 'teal')
        //         .attr('r', 4);
        // });

        // Update axis
        svg.select('.x.axis')
            .transition()
            .duration(1000)
            .call(xAxis);

        svg.select('.y.axis')
            .transition()
            .duration(1000)
            .call(yAxis);

    });

</script>

123

<div id='d3div'></div>

123
