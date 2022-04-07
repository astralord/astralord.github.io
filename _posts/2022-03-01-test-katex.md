---
layout: post
title: 'Test KaTeX'
date: 2022-03-01 03:22 +0800
categories: [Test]
tags: [test]
math: true
enable_d3: true
published: true
---

<script src="https://d3js.org/d3.v4.min.js"></script>

<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.13.13/dist/katex.min.css" integrity="sha384-RZU/ijkSsFbcmivfdRBQDtwuwVqK7GMOw6IMvKyeWL2K5UAlyp6WonmB8m7Jd0Hn" crossorigin="anonymous">
  <script defer src="https://cdn.jsdelivr.net/npm/katex@0.13.13/dist/katex.min.js" integrity="sha384-pK1WpvzWVBQiP0/GjnvRxV4mOb0oxFuyRxJlk6vVw146n3egcN5C925NCP7a7BY8" crossorigin="anonymous">
  </script>
  <script defer src="https://cdn.jsdelivr.net/npm/katex@0.13.13/dist/contrib/auto-render.min.js" integrity="sha384-vZTG03m+2yp6N6BNi5iM4rW4oIwk5DfcNdFfxkk9ZWpDriOkXX8voJBFrAO7MpVl" crossorigin="anonymous"
        onload="renderMathInElement(document.body);">
</script>

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

}

</style>

<div id="circle_fig">
    Node 1 <span id="node-text">\(\lambda x^2 \)</span>
</div>

<script>
function circle_fun() {

const div = d3.select("#circle_fig");
const span = div.select("#node-text");

const width = 50;
const height = 50;
const r = 25;

var container = span.append("span")
  .style("display", "inline-block"); 
  
const svg = container.append("svg")
  .attr("width", width)
  .attr("height", height);


const g = svg.append("g")
  .attr("id", "node");


const circle = g.append("circle")
  .attr("cx", width/2)
  .attr("cy", height/2)
  .attr("r", r)
  .attr("fill", "#faddcd");

span.style("position", "relative");

container.style("position", "absolute")
  .style("z-index", -1);

  const span_dim = span.node().getBoundingClientRect();
  const span_left = span_dim.left;
  const span_top = span_dim.top;

  var container_dim = container.node().getBoundingClientRect();
  var container_left = container_dim.left;
  var container_top = container_dim.top;

  const g_dim = g.node().getBoundingClientRect();
  const g_left = g_dim.left;
  const g_top = g_dim.top;
  const g_width = g_dim.width;
  const g_height = g_dim.height;

  const delta_g_left = g_left - container_left;
  const delta_g_top = g_top - container_top;


window.addEventListener("load", function(){

  const tex = span.select("span.katex");
  const tex_dim = tex.node().getBoundingClientRect();
  const tex_left = tex_dim.left;
  const tex_top = tex_dim.top;
  const tex_width = tex_dim.width;
  const tex_height = tex_dim.height;

  const delta_tex_left = tex_left - span_left;
  const delta_tex_top = tex_top - span_top;

  container.style("left", delta_tex_left + tex_width/2 - delta_g_left - g_width/2 + "px");
  container.style("top", delta_tex_top + tex_height/2 - delta_g_top - g_height/2 + "px");

});

}

circle_fun();

</script>


<div id="just_fig">
    Node 2 <span id="node-text">\(\mathbb{E} \chi \)</span>
</div>

<link href="https://fonts.googleapis.com/css?family=Arvo" rel="stylesheet">

<div id="chi_t_plt"></div> 

<script>
function chi_t_plts() {

var margin = {top: 20, right: 0, bottom: 30, left: 30},
    width = 700 - margin.left - margin.right,
    height = 200 - margin.top - margin.bottom,
    fig_width = 300;
    
var chi_svg = d3.select("#chi_t_plt")
  .append("svg")
    .attr("width", width + margin.left + margin.right)
    .attr("height", height + margin.top + margin.bottom)
  .append("g")
    .attr("transform",
          "translate(" + margin.left + "," + margin.top + ")");

var span_chi = d3.select("#chi_t_plt").append("span")
  .style('color', '#EDA137')
  .style("font-size", "17px")
  .style("font-weight", "700")
  .style("y", "60")
  .attr("x", 225)
  .attr("font-family", "Arvo")
  .attr("font-weight", 700)
  .attr("font-size", 20)
  .text("\\(\\chi_n^2\\)")
  .style("position", "absolute")
  .style("left", "290px")
  .style("bottom", "450px");
  
var span_t = d3.select("#chi_t_plt").append("span")
  .style('color', '#348ABD')
  .style("font-size", "17px")
  .style("font-weight", "700")
  .style("y", "60")
  .attr("x", 225)
  .attr("font-family", "Arvo")
  .attr("font-weight", 700)
  .attr("font-size", 20)
  .text("\\(t_n \\)")
  .style("position", "absolute")
  .style("left", "650px")
  .style("bottom", "450px");
    
chi_svg.append("text")
  .attr("text-anchor", "start")
  .attr("y", 60)
  .attr("x", 225)
  .attr("font-family", "Arvo")
  .attr("font-weight", 700)
  .attr("font-size", 20)
  .text("χ₅")
  .style("fill", "#EDA137").append('tspan')
    .text('2')
    .style('font-size', '.6rem')
    .attr('dx', '-.6em')
    .attr('dy', '-.9em')
    .attr("font-weight", 700);

var margin = {top: 0, right: 0, bottom: 35, left: 350};
    
var t_svg = chi_svg
  .append("svg")
    .attr("width", width + margin.left + margin.right)
    .attr("height", height + margin.top + margin.bottom)
  .append("g")
    .attr("transform",
          "translate(" + margin.left + "," + margin.top + ")");
     
var subscript_symbols = ['₁', '₂', '₃', '₄', '₅', '₆', '₇', '₈', '₉', '₁₀', '₁₁', '₁₂'];

t_svg.append("text")
  .attr("text-anchor", "start")
  .attr("y", 60)
  .attr("x", 225)
  .attr("font-family", "Arvo")
  .attr("font-weight", 700)
  .attr("font-size", 20)
  .text("t" + subscript_symbols[4])
  .style("fill", "#348ABD");
    
d3.csv("../assets/chi-t.csv", function(error, data) {
  if (error) throw error;

  var chi_x = d3.scaleLinear()
            .domain([-0, 40])
            .range([0, fig_width]);
            
  var xAxis = chi_svg.append("g")
      .attr("transform", "translate(0," + height + ")")
      .call(d3.axisBottom(chi_x));
  
  xAxis.selectAll(".tick text")
     .attr("font-family", "Arvo");

  var t_x = d3.scaleLinear()
            .domain([-20, 20])
            .range([0, fig_width]);
            
  xAxis = t_svg.append("g")
      .attr("transform", "translate(0," + height + ")")
      .call(d3.axisBottom(t_x));
       
  xAxis.selectAll(".tick text")
     .attr("font-family", "Arvo");

  var y = d3.scaleLinear()
            .range([height, 0])
            .domain([0, 0.5]);
            
  var yAxis = chi_svg.append("g")
      .call(d3.axisLeft(y).ticks(5));
  
  yAxis.selectAll(".tick text")
     .attr("font-family", "Arvo");
     
  var t_y = d3.scaleLinear()
            .range([height, 5])
            .domain([0, 0.5]);
                
  yAxis = t_svg.append("g")
      .call(d3.axisLeft(t_y).ticks(5));
      
  yAxis.selectAll(".tick text")
     .attr("font-family", "Arvo");

  var chi_curve = chi_svg
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
          .x(function(d) { return chi_x(d.chi_x); })
          .y(function(d) { return y(d["chi_5"]); })
      );
      
  var t_curve = t_svg
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
          .x(function(d) { return t_x(d.t_x); })
          .y(function(d) { return t_y(d["t_5"]); })
      );

  function updateChart(n) {
    n = parseInt(n);
    
    chi_curve
      .datum(data)
      .transition()
      .duration(0)
      .attr("d",  d3.line()
        .curve(d3.curveBasis)
          .x(function(d) { return chi_x(d.chi_x); })
          .y(function(d) { return y(d["chi_" + n]); })
      );
      
    t_curve
      .datum(data)
      .transition()
      .duration(0)
      .attr("d",  d3.line()
        .curve(d3.curveBasis)
          .x(function(d) { return t_x(d.t_x); })
          .y(function(d) { return t_y(d["t_" + n]); })
      );
      
    chi_svg.select("text").text("χ" + subscript_symbols[n - 1]).append('tspan')
    .text('2')
    .style('font-size', '.6rem')
    .attr('dx', '-.6em')
    .attr('dy', '-.9em')
    .attr("font-weight", 700);
    t_svg.select("text").text("t" + subscript_symbols[n - 1]);
    
  }
  
var slider_svg = d3.select("#chi_t_plt")
  .append("svg")
  .attr("width", width + 20)
  .attr("height", 70)
  .append("g")
  .attr("transform", "translate(" + 25 + "," + 20 + ")");

var n_x = d3.scaleLinear()
    .domain([1, 12])
    .range([0, width / 2])
    .clamp(true);
    
function roundN(x) { return Math.round(x - 0.5); }

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

createSlider(slider_svg, updateChart, n_x, 160, 0.1 * height, "n", "#696969", 5, roundN);

});
}

chi_t_plts();

</script>
