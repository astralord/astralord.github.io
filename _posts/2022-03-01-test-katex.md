---
layout: post
title: 'Test KaTeX'
date: 2022-03-21 03:22 +0800
categories: [Test]
tags: [test]
math: true
enable_d3: true
published: true
---



<script src="https://d3js.org/d3.v7.min.js"></script>

<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.13.13/dist/katex.min.css" integrity="sha384-RZU/ijkSsFbcmivfdRBQDtwuwVqK7GMOw6IMvKyeWL2K5UAlyp6WonmB8m7Jd0Hn" crossorigin="anonymous">
  <script defer src="https://cdn.jsdelivr.net/npm/katex@0.13.13/dist/katex.min.js" integrity="sha384-pK1WpvzWVBQiP0/GjnvRxV4mOb0oxFuyRxJlk6vVw146n3egcN5C925NCP7a7BY8" crossorigin="anonymous">
  </script>
  <script defer src="https://cdn.jsdelivr.net/npm/katex@0.13.13/dist/contrib/auto-render.min.js" integrity="sha384-vZTG03m+2yp6N6BNi5iM4rW4oIwk5DfcNdFfxkk9ZWpDriOkXX8voJBFrAO7MpVl" crossorigin="anonymous"
        onload="renderMathInElement(document.body);">
</script>


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