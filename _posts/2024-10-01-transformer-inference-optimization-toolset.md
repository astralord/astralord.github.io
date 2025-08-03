---
layout: post
title: 'Transformers Inference Optimization Toolset'
date: 2024-10-01 10:00 +0800
categories: [Large Language Models, Engineering]
tags: [jax, gpu, kv-cache, linear attention, cuda kernels, online softmax, flash attention, lightning attention, ring attention, vllm, sglang]
math: true
enable_d3: true
published: true
---

> Large Language Models are pushing the boundaries of artificial intelligence, but their immense size poses significant computational challenges. As these models grow, so does the need for smart optimization techniques to keep them running efficiently on modern hardware.
> 
> In this post, we'll explore key optimization strategies that are making LLMs faster and more memory-efficient. We'll start with a brief look at GPU memory hierarchy, which forms the foundation for many of these techniques. Then, we'll explore algorithms that allow LLMs to process information more quickly and handle longer contexts. Understanding these techniques offers valuable insights helping to unlock the full potential of Large Language Models.

The idea of this post is not just to discuss transformer-specific optimizations, since there are plenty of resources, where one can examine every inch of transformer to make it faster (my favourite one is the ["Let's reproduce GPT-2" by Andrej Karpathy](https://www.youtube.com/watch?v=l8pRSuU81PU)). The main goal is to lower the entry barrier for those curious researchers who are currently unable to piece together the huge number of articles and papers into one picture.

A lot of optimization techniques will be left out, like for example quantization methods, which are relatively diverse and deserve a separate post. Also we'll mostly discuss transformer inference and won't mention some training tricks, such as mixed-precision training, gradient checkpointing or sequence packing. But even so a lot of optimizations from this post could be applied to training as well.

## GPU architecture overview

To tackle language model speedup problem, first we need to understand the concept of the hardware we work on. While Google's TPUs and Apple silicon chips are rising up, NVIDIA's **GPUs** still dominate the market, so they'll be the subject of our in-depth look. 

Graphic processor unit performs all of the computations by multiple **streaming multiprocessors (SM)** (these are similar to the cores in the CPU). SM is basic GPU building block: it has its own instruction schedulers and various instruction execution pipelines. Modern GPUs are also equipped with special off-chip memory called **high bandwidth memory (HBM)**, where data is initially stored and ultimately written back. Unlike the system **dynamic random access memory (DRAM)**, which is controlled by CPU and typically optimized for low latency access, HBM is physically bonded to the GPUs in stacked layers with thousands of pins, providing massively parallel data throughput by design. 

Streaming multiprocessors access data and code from HBM via the **L2 cache**. It acts as an intermediate level between off-chip and on-chip memory and caches data that be shared among multiple SMs. It is also situated in the path of data moving between devices. Finally, each SM has its own **L1 cache** and **shared memory (SRAM)**, a low-latency on-chip memory caches: they are order of magnitude faster than HBM but many orders of magnitude smaller in capacity. L1 cache is managed by the GPU hardware, while SRAM can be explicitly managed by the programmer through NVIDIA tools. 

The GPUs can communicate to each other with a high bandwidth interconnect called **NVLink**, and they can talk to the outside world with a **PCIe bus** (a high-speed bus standard, common on motherboards to transfer data) or a special ethernet alternative called **Infiniband**. Usually, 8 GPUs are packed into a single node. Feel free to check out my post on [parallelization strategies](https://astralord.github.io/posts/exploring-parallel-strategies-with-jax/) to learn more on multi-device training.

<script src="https://d3js.org/d3.v5.min.js"></script>
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

</style>

<div id="gpu_arch" class="svg-container" align="center"></div> 

<script>

const colors = ['#65AD69', '#EDA137', '#E86456', '#348ABD'];
          
function rect(svg, x, y, w, h, color, opacity=1.0) {
	svg.append('rect')
	  .attr('x', x)
	  .attr('y', y)
	  .attr('width', w)
	  .attr('height', h)
	  .attr('stroke', 'black')
	  .attr('stroke-width', 1)
	  .attr("rx", Math.min(3, Math.min(w, h) / 3))
	  .attr('fill', color)
	  .attr('opacity', opacity);
}

function text_(svg, text, x, y, size=14, color="currentColor") {
	svg.append('text')
	  .attr('x', x)
	  .attr('y', y)
	  .text(text)
	  .style("font-size", size + "px")
	  .style("fill", color)
	  .attr("font-family", "Arvo");
}

function line_(svg, x0, y0, x1, y1) {
	svg.append("path")
	   .datum([{x: x0, y: y0},
	           {x: x1, y: y1}])
	   .attr("fill", "none")
	   .attr("stroke-width", 1)
	   .attr("stroke", "currentColor")
	   .attr("d",  d3.line()
	       .x(function(d) { return d.x; })
	       .y(function(d) { return d.y; }));
}

function left_triangle(svg, x, y) {
	svg.append("path")
	   .datum([{x: x + 3, y: y - 1},
	           {x: x, y: y},
	           {x: x + 3, y: y + 1}])
	   .attr("fill", "none")
	   .attr("stroke-width", 1)
	   .attr("stroke", "currentColor")
	   .attr("d",  d3.line()
	       .x(function(d) { return d.x; })
	       .y(function(d) { return d.y; }));
}

function right_triangle(svg, x, y) {
	svg.append("path")
	   .datum([{x: x - 3, y: y - 1},
	           {x: x, y: y},
	           {x: x - 3, y: y + 1}])
	   .attr("fill", "none")
	   .attr("stroke-width", 1)
	   .attr("stroke", "currentColor")
	   .attr("d",  d3.line()
	       .x(function(d) { return d.x; })
	       .y(function(d) { return d.y; }));}

function long_right_arrow(svg, x, y, length=50) {
	line_(svg, x, y, x + length, y);
	left_triangle(svg, x, y);
	right_triangle(svg, x + length, y);
}

function long_up_arrow(svg, x, y, length=50) {
	line_(svg, x, y, x, y + length);
	       
	svg.append("path")
	   .datum([{x: x - 1, y: y + length - 3},
	           {x: x, y: y + length},
	           {x: x + 1, y: y + length - 3}])
	   .attr("fill", "none")
	   .attr("stroke-width", 1)
	   .attr("stroke", "currentColor")
	   .attr("d",  d3.line()
	       .x(function(d) { return d.x; })
	       .y(function(d) { return d.y; }));
	       
	svg.append("path")
	   .datum([{x: x - 1, y: y + 3},
	           {x: x, y: y},
	           {x: x + 1, y: y + 3}])
	   .attr("fill", "none")
	   .attr("stroke-width", 1)
	   .attr("stroke", "currentColor")
	   .attr("d",  d3.line()
	       .x(function(d) { return d.x; })
	       .y(function(d) { return d.y; }));
}

function system_dram_rect(svg, x, y) {
	rect(svg, x, y, 120, 160, colors[3]);
	text_(svg, "System", x + 36, y + 75, 14, "black");
	text_(svg, "DRAM", x + 38, y + 95, 14, "black");
}

function dram_rect(svg, x, y) {
	rect(svg, x, y, 60, 50, colors[2]);
	text_(svg, "Device", x + 8, y + 20, 14, "black");
	text_(svg, "DRAM", x + 8, y + 40, 14, "black");
}

function l2_rect(svg, x, y) {
	rect(svg, x, y, 35, 35, colors[1]);
	text_(svg, "L2", x + 10, y + 22, 14, "black");
}

function l1_rect(svg, x, y) {
	rect(svg, x, y, 120, 30, 'none');
	rect(svg, x + 10, y + 5, 70, 20, colors[0]);
	text_(svg, "L1/SRAM", x + 13, y + 20, 14, "black");
	text_(svg, "SM", x + 88, y + 20);
}

function pci_arrow(svg, x, y) {
	long_right_arrow(svg, x, y);
	text_(svg, 'PCIe', x + 14, y - 3, size=11);
}

function nvlink_arrow(svg, x, y) {
	long_up_arrow(svg, x, y, length=26);
	text_(svg, 'NVLink', x - 45, y + 18, size=11);
}

function gpu_rect(svg, x, y) {
	svg.append('rect')
	  .attr('x', x)
	  .attr('y', y)
	  .attr('width', 120)
	  .attr('height', 60)
	  .attr('stroke', 'currentColor')
	  .attr('stroke-width', 2)
	  .attr("rx", 3)
	  .style('stroke-dasharray', ('2,3'))
	  .attr('fill', 'none');
	  
	dram_rect(svg, x + 8, y + 5);
	rect(svg, x + 72, y + 24, 12, 12, colors[1]);
	line_(svg, x + 68, y + 30, x + 72, y + 30);
	line_(svg, x + 84, y + 30, x + 88, y + 30);
	line_(svg, x + 88, y + 10, x + 88, y + 48);
	
	for (var i = 0; i != 5; i += 1) {
		rect(svg, x + 90, y + 10 + i * 8, 6, 6, colors[0]);
		rect(svg, x + 98, y + 10 + i * 8, 6, 6, colors[0]);
		rect(svg, x + 106, y + 10 + i * 8, 6, 6, colors[0]);
	}
}

function cache_rect(svg, x, y) {
	l2_rect(svg, x + 10, y + 68);
	
	line_(svg, x + 75, y + 15, x + 75, y + 155);
	line_(svg, x + 50, y + 85, x + 75, y + 85);
	left_triangle(svg, x + 50, y + 85);
	
	l1_rect(svg, x + 100, y);
	line_(svg, x + 75, y + 15, x + 96, y + 15);
	right_triangle(svg, x + 95, y + 15);
	
	l1_rect(svg, x + 100, y + 35);
	line_(svg, x + 75, y + 50, x + 96, y + 50);
	right_triangle(svg, x + 95, y + 50);
	
	l1_rect(svg, x + 100, y + 70);
	line_(svg, x + 75, y + 85, x + 96, y + 85);
	right_triangle(svg, x + 95, y + 85);
	
	l1_rect(svg, x + 100, y + 140);
	line_(svg, x + 75, y + 155, x + 96, y + 155);
	right_triangle(svg, x + 95, y + 155);
	
	svg.append("path")
	   .datum([{x: x, y: y + 180}, 
	           {x: x + 230, y: y + 180}, 
	           {x: x + 230, y: y - 10}, 
	           {x: x, y: y - 10},
	           {x: x, y: y + 10},
	           {x: x - 35, y: y + 30},
	           {x: x, y: y + 50},
	           {x: x, y: y + 181}])
	   .attr("fill", "none")
	   .attr("stroke-width", 3)
	   .attr("opacity", 0.4)
	   .attr("stroke", "gray")
	   .attr("d",  d3.line()
	       .x(function(d) { return d.x; })
	       .y(function(d) { return d.y; }));
	       
	svg.append('path')
	   .datum([{x: x + 140, y: y + 105}, 
	           {x: x + 140, y: y + 135}])
	  .attr('stroke', 'currentColor')
	  .attr('stroke-width', 2)
	  .style('stroke-dasharray', ('2,5'))
	  .attr('fill', 'none')
	  .attr("d",  d3.line()
	       .x(function(d) { return d.x; })
	       .y(function(d) { return d.y; }));
}

function gpu_arch() {
	var svg = d3.select("#gpu_arch")
				  .append("svg")
				  .attr("width", 600)
				  .attr("height", 215);
	const x_st = 10, y_st = 30;
	
	system_dram_rect(svg, x_st, y_st);
	pci_arrow(svg, x_st + 130, y_st + 30);
	pci_arrow(svg, x_st + 130, y_st + 124);
	
	gpu_rect(svg, x_st + 190, y_st);
	text_(svg, "GPU 0", x_st + 230, y_st - 10);
	
	gpu_rect(svg, x_st + 190, y_st + 94);
	text_(svg, "GPU 1", x_st + 230, y_st + 175);
	
	nvlink_arrow(svg, x_st + 228, y_st + 64);
	cache_rect(svg, x_st + 350, y_st);
	       
}

gpu_arch();
</script>

![](.)
*Schematic example of memory architecture with 2 GPU devices.*

Now, when it comes to GPU capability, we must look at three things:

* Compute performance measured by the number of **trillion float operations per second (TFLOPS)**.
* **GPU memory** required to store model parameters, hidden activations and cache values, measured in GBs. For instance, GPT-3 has 175 billion parameters, so we need 350 GBs of memory just to keep them on device in fp16.
* **Memory bandwidth** measured in GB/s - the speed of bytes movement from 
GPU to processing units.

GPU capabilities grow exponentially fast. According to NVIDIA documentation, T4 graphics card released in 2018 had **65 TFLOPs**, <span style="color:#65AD69">40 SMs</span> with <span style="color:#65AD69">64KB</span> L1 cache each, <span style="color:#EDA137">4MB</span> L2 cache with <span style="color:#EDA137">1.3TB/s</span> bandwidth and <span style="color:#E86456">16GB</span> HBM with <span style="color:#E86456">300 GB/s</span> bandwidth. After just 2 years A100 was released with **312 TFLOPs** and <span style="color:#65AD69">108 SMs</span> with <span style="color:#65AD69">192KB</span> of L1, <span style="color:#EDA137">40MB</span> of L2 cache and <span style="color:#E86456">80 GB</span> of HBM with <span style="color:#E86456">1.55 TB/s</span> bandwidth. Compare those numbers to the latest B100 card, which can perform **1.8 PFLOPs** and which HBM has a capacity of <span style="color:#E86456">192 GB</span> and a throughput of <span style="color:#E86456">8 TB/s</span>.

<div id="gpu_timeline" class="svg-container" align="center"></div> 

<script>

function add_gpu_dot(svg, x, y, cx, cy, x_shift, y_shift) {
   	svg.append('g')
   		.selectAll("dot")
   		.data([{x: cx, y: cy}])
   		.enter()
   		.append("circle")
   			.attr("cx", function (d) { return x(d.x); } )
   			.attr("cy", function (d) { return y(d.y); } )
   			.attr("r", 3)
   			.style("fill", colors[3])
   			.attr("stroke", "black")
   			.attr("stroke-width", 1)
     		.attr("transform", "translate(" + x_shift + ", " +  y_shift + ")");
}

function gpu_timeline() {
	var svg = d3.select("#gpu_timeline")
				  .append("svg")
				  .attr("width", 600)
				  .attr("height", 190);
	const x_shift = 100, y_shift = 20;
		
	var x = d3.scaleLog()
		.domain([50, 2000])
		.range([0, 300]).base(2).nice();
		
   var xAxis = svg.append("g")
      .attr("transform", "translate(" + x_shift + ", " +  (y_shift + 150) + ")")
      .call(d3.axisBottom(x).ticks(4));
      
   xAxis.selectAll(".tick text")
     .attr("font-family", "Arvo");
   
   var y = d3.scaleLog()
            .domain([300, 8000])
            .range([150, 0]).base(2).nice();
            
   var yAxis = svg.append("g")
      .attr("transform", "translate(" + x_shift + ", " + y_shift + ")")
      .call(d3.axisLeft(y).ticks(3));
      
   	yAxis.selectAll(".tick text")
   		.attr("font-family", "Arvo");
   	
   	text_(svg, "Performance", 405, 170, size=11);
	text_(svg, "TFLOPs", 405, 186, size=11);
	text_(svg, "Bandwidth", 35, 10, size=11);
	text_(svg, "GB/s", 65, 26, size=11);

   	add_gpu_dot(svg, x, y, 65, 300, x_shift, y_shift);
	text_(svg, "T4", 135, 155, size=11);
	
   	add_gpu_dot(svg, x, y, 125, 900, x_shift, y_shift);
	text_(svg, "V100", 165, 110, size=11);
	
   	add_gpu_dot(svg, x, y, 312, 1555, x_shift, y_shift);
	text_(svg, "A100", 235, 85, size=11);

   	add_gpu_dot(svg, x, y, 1000, 3000, x_shift, y_shift);
	text_(svg, "H100", 315, 55, size=11);
	
   	add_gpu_dot(svg, x, y, 1800, 8000, x_shift, y_shift);
	text_(svg, "B100", 400, 20, size=11);
}

gpu_timeline();

</script>

![](.)
*Memory bandwidth vs compute performance rapid growth. Note that both axes are log-scale.[^GTL]*

Time required for memory accesses can vary depending on the devices, their modifications and infrastructure setups. But the main point to remember is that if we compare throughput numbers, we will see that some of them differ by orders of magnitude:

- To read data from L1 cache / SRAM: `x` ns.
- To read data from L2 cache: `2-3x` ns.
- To read data from HBM memory: `10x` ns.
- To share data between GPUs with NVLink (both ways): `50-60x` ns.
- To load data from CPU to GPU DRAM through PCIe bus: `~300x` ns.

These numbers show us that while the number of operations per second matters, the operand placement can be even more important when we optimizing for inference speed. *Keep in mind that the slower memory always dominates performance bottlenecks.* 

### Arithmetic intensity vs `ops:byte`

Depending on the balance of computation and memory accesses, operations can be classified as follows: 

1. **Compute-bound**: the time spent on arithmetic operations exceeds time spent for other operations such as memory accesses. Typical examples are linear layer with large inner dimension or convolutional layer with large number of channels.
2. **Memory-bound**: the time taken by memory accesses exceeds computation time. Most operations are memory bound, e.g. elementwise operations (activation functions, dropouts) or reductions (sum, softmax, normalization). 
3. **Overhead-bound**: everything else, such as communication-bound, interpreter-bound, etc. We won't discuss it in this post, however I strongly advise to check out [Making Deep Learning Go Brrrr From First Principles](https://horace.io/brrr_intro.html) blogpost to better understand GPU mechanisms and why in most of the cases our bottleneck may not be related to them at all.

The balance between first two is commonly measured by the **arithmetic intensity**, which is the number of arithmetic operations per byte of memory access required to run the computation. For example, if we apply ReLU activation to an input tensor $x$ (assuming it's in half-precision), we need to

* read 2 bytes
* make 1 comparison
* write 2 bytes

for each element of the tensor. Regardless of the size of $x$, arithmetic intensity for ReLU is equal to $\frac{\\# \operatorname{flops}}{\\# \operatorname{bytes}}=\frac{1}{4}$. Again, this means that for each operation we need to make 4 memory accesses.

Arithmetic intensity is commonly compared to a hardware-specific `ops:byte` ratio to determine whether we are in a compute- or memory-bound scenario. To explain how it works, let's take for example a linear layer forward pass on A100 GPU. Given an input batch $x \in \mathbb{R}^{B \times d}$ and weight matrix $\mathbf{W} \in \mathbb{R}^{d \times d}$ (here $B$ is a batch size and $d$ is an embedding dimension) linear layer basically represents a matrix multiplication $x\mathbf{W}$. We can calculate that linear layer computation requires $2Bd^2$ flops.[^MMM] Hence the compute-time for A100 will be

$$T_{\operatorname{compute}} = \frac{\# \operatorname{flops}}{\operatorname{compute performance}} = \frac{2Bd^2}{312 \cdot 10^{12}} s.$$

At the same time we need to read $2d^2$ bytes from memory to load weight matrix $\mathbf{W}$ (again under the condition that we work with fp16/bf16). Also, just for simplicity let's say that $B \ll d$ and we can neglect the loading time of $x$ compared to weight matrix $\mathbf{W}$. Model parameters are usually stored at HBM, therefore

$$T_{\operatorname{memory}} = \frac{\# \operatorname{bytes}}{\operatorname{memory bandwidth}} = \frac{2d^2}{1.55 \cdot 10^{12}} s.$$

Recall that arithmetic intensity is equal to $\frac{\\# \operatorname{flops}}{\\# \operatorname{bytes}}$, while `ops:byte` is given by $\frac{\operatorname{compute performance}}{\operatorname{memory bandwidth}}$. To find the bottleneck for our model we look at the ratio of these two terms, which is

$$\frac{T_{\operatorname{compute}}}{T_{\operatorname{memory}}} \approx \frac{B}{200}.$$

This means that for batch sizes below $200$ our system performance remains memory-bound. Enlarging input batch to a value greater than $200$ increases the computation time, while keeping the memory transfer time constant, which brings us to the compute-bound scenario. 

The `ops:byte` ratio analysis is useful, but keep in mind, that it assumes that a GPU workload is sufficiently large to saturate compute and memory pipelines. If the workload is not large enough, or does not have sufficient parallelism, the processor will be under-utilized and performance will be limited by latency. 

## High-level algorithmic optimizations

Now, we are ready to delve into the specifics of transformer optimization. We've defined transformer architecture earlier in [previous blog-posts](https://astralord.github.io/posts/building-aligned-intelligence-systems-part-i-creating-gpt-assistant/). Let's recall shortly that **scaled dot product attention** operation takes a set of queries $\mathbf{Q}$, keys $\mathbf{K}$ and values $\mathbf{V}$ as input and outputs

$$\operatorname{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \operatorname{softmax} \Big( \frac{\mathbf{QK}^T}{\sqrt{d}}  \Big) \cdot \mathbf{V}, $$

where $d$ is a hidden dimensionality for queries and keys. When we work with GPT-based models, we use **masked attention** where $\operatorname{softmax}$ input is modified with $\text{mask}$ tensor, setting masked attention values to $-\infty$ if we don't want to attend to corresponding tokens. Input tensors are $\mathbf{Q} \in \mathbb{R}^{L \times d}$ and $\mathbf{K}, \mathbf{V} \in \mathbb{R}^{M \times d}$, where $L$ and $M$ are sequence lengths[^VD].

Also let's recap **multi-head attention layer (MHA)** definition:
 
$$\operatorname{MultiHead}(\mathbf{Q}, \mathbf{K}, \mathbf{V})=[\operatorname{head}_1; \dots; \operatorname{head}_k] \cdot \mathbf{W}^O,$$

where

$$\operatorname{head}_i = \operatorname{Attention}(\mathbf{QW}_i^Q, \mathbf{KW}_i^K, \mathbf{VW}_i^V), \quad i = 1, \dots, h.$$

with learnable parameters $\mathbf{W}^Q_{1 \dots h}, \mathbf{W}^K_{1 \dots h}, \mathbf{W}^V_{1 \dots h}$ and $\mathbf{W}^O$. If MHA receives $\mathbf{Q} = \mathbf{K}$ (and normally $ = \mathbf{V}$), we call it **multi-head self-attention**, otherwise it is called **multi-head cross-attention**. We'll focus on self-attention mechanism as it's widely used in generative LLMs.

Let's introduce new tensor names to simplify the notation: we'll call dot product $\mathbf{S} := \mathbf{QK}^T \in \mathbb{R}^{L \times L}$, normalized attention weights $\mathbf{P} := \operatorname{softmax}(\mathbf{S} \circ \text{mask}) \in \mathbb{R}^{L \times L}$ ($\text{mask}$ is broadcastable to $\mathbf{S}$) and output $\mathbf{O} := \mathbf{PV} \in \mathbb{R}^{L \times d}$.

### KV Cache

When we work with models like GPT, text generation occurs in two stages:

1. **Prefill** - the model ingests large chunk of our prompt tokens in parallel, computing all hidden states and outputs in one pass.
2. When prefill is finished **auto-regressive decoding** is launched. Decoding is in general more time-consuming than prefill due to its sequential nature: response tokens are always generated one after another.

<div id="text_generation" class="svg-container" align="center"></div> 

<script>

const matrix_colors = ['#C7E9E3', '#FDBFB9', '#FFF6B7', '#FEDAB1', 
                       '#D9EFB5', '#FFFFDA', '#E6F5E2', '#B3B3B3'];

function trivialRound(x) { return x; }
function roundN(x) { return Math.round(x); }

function createSlider(svg_, parameter_update, x, loc_x, loc_y, letter, color, init_val, round_fun, x_ticks, shift_text=23, text_size=14) {
    var slider = svg_.append("g")
      .attr("class", "slider")
      .attr("transform", "translate(" + loc_x + "," + loc_y + ")");
    
    var drag = d3.drag()
	        .on("start.interrupt", function() { slider.interrupt(); })
	        .on("start drag", function(event, d) { 
	          handle.attr("cx", x(round_fun(x.invert(d3.event.x))));  
	          parameter_update(round_fun(x.invert(d3.event.x)));	         });
	         
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
  .data(x.ticks(x_ticks))
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
	  .attr("x", loc_x - shift_text)
	  .attr("font-family", "Arvo")
	  .attr("font-size", text_size)
	  .text(letter)
	  .style("fill", color);
	  	  
	return handle;
}

function vector_rect(svg, x, y, w, h, shift, rct_sz, color, opacity=1.0) {
	for (var i = 0; i < w; i += 1) {
		for (var j = 0; j < h; j += 1) {
			rect(svg,
				x + i * shift, 
				y + j * shift, 
				rct_sz, 
				rct_sz, 
				color,
				opacity);
		}
	}
}

function causal_mask(svg, x, y, w, h, shift, rct_sz, color, sliding_window_size=10, mask_shift=0, opacity=1) {
	for (var i = 0; i < w; i += 1) {
		for (var j = 0; j < h; j += 1) {
			rect(svg,
				x + i * shift, 
				y + j * shift, 
				rct_sz, 
				rct_sz, 
				((i > j + mask_shift || i <= j - sliding_window_size) ? 'white' : color),
				opacity);
		}
	}
}

function draw_sampling_text(svg, x_start, y_start, rct_sz, shift) {
	text_(svg, "Q", x_start + 8, y_start + 2 * shift);
	
	text_(svg, "K", x_start + 9 * rct_sz, 20);
	text_(svg, "T", x_start + 9 * rct_sz + 10, 12, size=8);
	text_(svg, "S", x_start + 9 * rct_sz, 230);
	text_(svg, "mask", x_start + 23 * rct_sz, 60);
		
   	svg.append('g')
   		.selectAll("dot")
   		.data([{x: x_start + shift * 14.4, y: y_start + shift * 7.85}])
   		.enter()
   		.append("circle")
   			.attr("cx", function (d) { return d.x; } )
   			.attr("cy", function (d) { return d.y; } )
   			.attr("r", 3)
   			.style("fill", "none")
   			.attr("stroke", "currentColor")
   			.attr("stroke-width", 1);
	
	text_(svg, "V", x_start + 30 * shift + 8, y_start + 2 * shift);
	text_(svg, "O", x_start + 35 * shift + 8, y_start + 2 * shift);
	text_(svg, "(", x_start - 15, y_start + 8.5 * shift, size=30);
	text_(svg, ")", x_start + 31 * rct_sz, y_start + 8.5 * shift, size=30);
	text_(svg, "⋅", x_start + shift * 28, y_start + shift * 8.55, size=30);
	text_(svg, "=", x_start + shift * 33, y_start + shift * 8.25, size=18);
	text_(svg, "softmax", x_start - 5 * shift, y_start + shift * 8.25);
}

function text_generation() {
	var svg = d3.select("#text_generation")
				  .append("svg")
				  .attr("width", 700)
				  .attr("height", 300);
	const x_start = 90, y_start = 30;
	const shift = 14, rct_sz = 12;
	var init_seq_len = 10, dim = 2;
	
	function draw_attn_cells(seq_len) {
	   svg.selectAll('rect').remove();
		vector_rect(svg, x_start, y_start + 3 * shift, dim, seq_len, shift, rct_sz, matrix_colors[0]);
		
		vector_rect(svg, x_start + 3 * shift, y_start, seq_len, dim, shift, rct_sz, matrix_colors[1]);
		
		vector_rect(svg, x_start + 3 * shift, y_start + 3 * shift, seq_len, seq_len, shift, rct_sz, matrix_colors[2]);
		
		causal_mask(svg, x_start + 16 * shift, y_start + 3 * shift, seq_len, seq_len, shift, rct_sz, matrix_colors[7]);
		
		vector_rect(svg, x_start + 30 * shift, y_start + 3 * shift, dim, seq_len, shift, rct_sz, matrix_colors[3]);
		
		vector_rect(svg, x_start + 35 * shift, y_start + 3 * shift, dim, seq_len, shift, rct_sz, matrix_colors[4]);
		
		init_seq_len = seq_len;
	}
	
	draw_sampling_text(svg, x_start, y_start, rct_sz, shift);
	draw_attn_cells(init_seq_len);

	var l_x = d3.scaleLinear()
	    .domain([1, 10])
	    .range([0, 320])
	    .clamp(true);
	    
	createSlider(svg, draw_attn_cells, l_x, x_start + 80, 270, "L", "currentColor", init_seq_len, roundN, 10);
}

text_generation();

</script>

![](.)
*Representation of causal self-attention during text generation for different sequence lengths $L$. Scaling coefficient $\sqrt{d}$ is omitted. Attention values can be computed only once for the whole bunch of prompt tokens, but then sequential one-by-one computations are required to generate response tokens.*

```python
@jit
def dot_product_attention(query, key, value, mask=None):
    d = query.shape[-1]
    # attn_logits shape is [batch..., num_heads, q_seq_len, kv_seq_len]
    attn_logits = jnp.einsum('...lhd,...mhd->...hlm', query, key)
    attn_logits = attn_logits / jnp.sqrt(d) # normalize logits
    if mask is not None:
        big_neg = jnp.finfo(attn_logits.dtype).min
        attn_logits = jnp.where(mask, big_neg, attn_logits)
    # logits -> weights
    attention = nn.softmax(attn_logits, axis=-1)
    # return weighted sum over values for each query position
    output = jnp.einsum('...hlm,...mhv->...lhv', attention, value)
    return output, attention
```

Let's check whether MHA inference is compute- or memory-bound.

* Computational complexity:

	- To compute query $\mathbf{Q}$ we multiply input matrix $x \in \mathbb{R}^{L \times d}$ with matrices $\mathbf{W}^Q_{1 \dots h} \in \mathbb{R}^{d \times \frac{d}{h}}$ across $h$ heads which takes $\mathcal{O}(Ld^2)$ operations. The same amount of compute is needed for $\mathbf{K}$ and $\mathbf{V}$.
	- Attention computation requires $\mathcal{O}(L^2d)$ for both $\mathbf{S} = \mathbf{QK}^T$ and $\mathbf{O}=\mathbf{PV}$.

	In total, we need to perform $\mathcal{O}(Ld^2 + L^2d)$ operations.

* Memory accesses:

	- Input $x$ and intermediate tensors $\mathbf{Q}$, $\mathbf{K}$, $\mathbf{V}$, $\mathbf{O}$ occupy $\mathcal{O}(Ld)$ bytes.
	- Attention logits $\mathbf{S}$ and weights $\mathbf{P}$ take $\mathcal{O}(L^2h)$ bytes for all heads in total.
	- Projection weights $\mathbf{W}^{Q,K,V}_{1 \dots h}$ require $\mathcal{O}(d^2)$ memory space.

	The entire memory size to be accessed is equal to the sum of the sizes of all the tensors involved, which is $\mathcal{O}(Ld + L^2h + d^2)$ bytes. Therefore, we have arithmetic intensity proportional to 
	
	$$\frac{L^2d + Ld^2}{L^2h + Ld + d^2} \xrightarrow[L \gg d]{} \frac{d}{h}$$

<div id="arithmetic_intensity" class="svg-container" align="center"></div> 

<script>

function arithmetic_intensity() {

	var svg = d3.select("#arithmetic_intensity")
				  .append("svg")
				  .attr("width", 500)
				  .attr("height", 200);
				  
	const x_start = 80, y_start = 10;
	const x_end = 450, y_end = 180;
	const dim = 12288, h = 96;
	
	text_(svg, "arithmetic", 0, y_start, size=11);
	text_(svg, "intensity", 5, y_start + 10, size=11);
	text_(svg, "L", x_end + 20, y_end + 5, size=11);
	text_(svg, "d/h -", 55, 152, size=11);
	
	var ai_x = d3.scaleLinear()
        .domain([0, 30000])
        .range([x_start, x_end]);
        
  	var xAxis = svg.append("g")
      .attr("transform", "translate(0," + y_end + ")")
      .call(d3.axisBottom(ai_x).ticks(4));
      
   xAxis.selectAll(".tick text")
	   .attr("font-family", "Arvo");
	
	var ai_y = d3.scaleLinear()
	          .range([y_end, y_start])
	          .domain([0, 700]);
	          
	var yAxis = svg.append("g")
	   .attr("transform", "translate(" + x_start + ", 0)")
	   .call(d3.axisLeft(ai_y).ticks(0));
	   
	yAxis.selectAll(".tick text")
	    .attr("font-family", "Arvo");
	    
   var ai_data = [];
   for (var l = 0; l < 30000; l += 10) {
   		var lsq = l * l, dsq = dim * dim;
   		ai_data.push({x: l, y: (lsq * dim + dsq * l) / (lsq * h + l * dim + dsq) });
	};
	
	var ai_curve = svg
	    .append('g')
	    .append("path")
	      .datum(ai_data)
	      .attr("fill", "none")
	      .attr("border", 0)
	      .attr("stroke", "currentColor")
	      .attr("stroke-width", 2)
	      .attr("stroke-linejoin", "round")
	      .attr("d",  d3.line()
	        .curve(d3.curveBasis)
	          .x(function(d) { return ai_x(d.x); })
	          .y(function(d) { return ai_y(d.y); })
	      );
	      
	var ai_limit = svg
	    .append('g')
	    .append("path")
	      .datum([{x: 0, y: dim / h}, {x: 30000, y: dim / h}])
	      .attr("fill", "none")
	      .attr("border", 0)
	      .attr("stroke", "currentColor")
	      .attr("stroke-width", 1)
	      .attr("stroke-linejoin", "round")
	      .style('stroke-dasharray', ('2,3'))
	      .attr("d",  d3.line()
	        .curve(d3.curveBasis)
	          .x(function(d) { return ai_x(d.x); })
	          .y(function(d) { return ai_y(d.y); })
	      );
	      
	
	var l_x = d3.scaleLinear()
	    .domain([1, 10])
	    .range([0, 320])
	    .clamp(true);

	createSlider(svg, draw_attn_cells, l_x, x_start, 280, "L", "currentColor", init_seq_len, roundN, 10);
	
}

arithmetic_intensity();

</script>

![](.)
*Arithmetic intensity vs sequence length $L$ for multi-head self-attention with embedding dimension $d=12,288$ and number of heads $h=96$ (GPT-3 scale).*

We've noticed already that modern GPU hardware has the computational capacity orders of magnitude higher than the memory bandwidth. As the graph shows, for sufficiently large sequence length the arithmetic intensity is always larger than embedding dimension per attention head $\frac{d}{h}$, which is usually one or few hundreds.[^AIL] Hence, the arithmetic intensity is equal if not greater than `ops:byte` ratio.

Generally, this would imply high algorithm efficiency, but the situation is different for the second phase, text generation. First thing to notice here is that in generation scenario there is no need to compute the attention outputs for each token in the input sequence $x$, only for the last one to decode the next token $x_{L+1}$. Thus there is no need to send the whole query $\mathbf{Q}$ vector into attention mechanism.

The second important thing is that we can reuse previously computed activations, namely we can cache $\mathbf{K}$ and $\mathbf{V}$ values during generation process, hence the naming. We store the **KV cache** to improve efficiency and reduce redundant computational requirements, especially for long sequences.

<div id="text_generation_w_kv_cache" class="svg-container" align="center"></div> 

<script>

function draw_sampling_text_2(svg, x_start, y_start, rct_sz, shift) {
	text_(svg, "Q", x_start + 8, y_start + 2 * shift);
	
	text_(svg, "K", x_start + 9 * rct_sz, y_start - 10);
	text_(svg, "T", x_start + 9 * rct_sz + 10, y_start - 18, size=8);
	text_(svg, "S", x_start + 9 * rct_sz, y_start + 200);
	
	text_(svg, "V", x_start + 17 * shift + 8, y_start + 2 * shift);
	text_(svg, "O", x_start + 21 * shift + 8, y_start + 2 * shift);
	text_(svg, "(", x_start - 15, y_start + 8.5 * shift, size=30);
	text_(svg, ")", x_start + 14 * shift, y_start + 8.5 * shift, size=30);
	text_(svg, "⋅", x_start + 15.3 * shift, y_start + shift * 8.55, size=30);
	text_(svg, "=", x_start + 19.7 * shift, y_start + shift * 8.25, size=18);
	text_(svg, "softmax", x_start - 5 * shift, y_start + shift * 8.25);
		
	text_(svg, "KV cache", x_start + 27 * shift, y_start - 10, size=11);
}

function text_generation_w_kv_cache() {
	var svg = d3.select("#text_generation_w_kv_cache")
				  .append("svg")
				  .attr("width", 550)
				  .attr("height", 300);
				  
	const x_start = 110, y_start = 30;
	const shift = 14, rct_sz = 12;
	var init_seq_len = 10, dim = 2;
	
	function draw_attn_cells_w_kv_cache(seq_len) {
	   svg.selectAll('rect').remove();
		vector_rect(svg, x_start, y_start + 3 * shift, dim, seq_len - 1, shift, rct_sz, matrix_colors[0], 0.1);
		vector_rect(svg, x_start, y_start + (seq_len + 2) * shift, dim, 1, shift, rct_sz, matrix_colors[0]);
		
		vector_rect(svg, x_start + 3 * shift, y_start, seq_len, dim, shift, rct_sz, matrix_colors[1]);
		
		for (var i = 1; i < seq_len; i += 1) {
			vector_rect(svg, x_start + 3 * shift, y_start + (i + 2) * shift, i, 1, shift, rct_sz, matrix_colors[2], 0.1);
		}
		vector_rect(svg, x_start + 3 * shift, y_start + (seq_len + 2) * shift, seq_len, 1, shift, rct_sz, matrix_colors[2]);
		
		vector_rect(svg, x_start + 17 * shift, y_start + 3 * shift, dim, seq_len, shift, rct_sz, matrix_colors[3]);
		
		vector_rect(svg, x_start + 21 * shift, y_start + 3 * shift, dim, seq_len - 1, shift, rct_sz, matrix_colors[4], 0.1);
		vector_rect(svg, x_start + 21 * shift, y_start + (seq_len + 2) * shift, dim, 1, shift, rct_sz, matrix_colors[4]);
		
		if (seq_len > 1) {
			svg.append('rect')
			  .attr('x', x_start + 3 * shift - 3)
			  .attr('y', y_start - 3)
			  .attr('width', (seq_len - 1) * shift + 2)
			  .attr('height', 2 * shift + 4)
			  .attr('stroke', 'currentColor')
			  .attr('stroke-width', 2)
			  .attr("rx", 3)
			  .style('stroke-dasharray', ('2,3'))
			  .attr('fill', 'none');
			
			svg.append('rect')
			  .attr('x', x_start + 17 * shift - 3)
			  .attr('y', y_start + 3 * shift - 3)
			  .attr('width', 2 * shift + 4)
			  .attr('height', (seq_len - 1) * shift + 2)
			  .attr('stroke', 'currentColor')
			  .attr('stroke-width', 2)
			  .attr("rx", 3)
			  .style('stroke-dasharray', ('2,3'))
			  .attr('fill', 'none');
		}
		
		svg.append('rect')
		  .attr('x', x_start + 28 * shift - 3)
		  .attr('y', y_start - 3)
		  .attr('width', 2 * shift + 4)
		  .attr('height', 2 * shift + 4)
		  .attr('stroke', 'currentColor')
		  .attr('stroke-width', 2)
		  .attr("rx", 3)
		  .style('stroke-dasharray', ('2,3'))
		  .attr('fill', 'none');
		  
		vector_rect(svg, x_start + 28 * shift, y_start, dim, 1, shift, rct_sz, matrix_colors[1]);
		vector_rect(svg, x_start + 28 * shift, y_start + shift, dim, 1, shift, rct_sz, matrix_colors[3]);
	
		init_seq_len = seq_len;
	}

	draw_sampling_text_2(svg, x_start, y_start, rct_sz, shift);
	draw_attn_cells_w_kv_cache(init_seq_len);

	var l_x = d3.scaleLinear()
	    .domain([1, 10])
	    .range([0, 320])
	    .clamp(true);

	createSlider(svg, draw_attn_cells_w_kv_cache, l_x, x_start, 270, "L", "currentColor", init_seq_len, roundN, 10);
}

text_generation_w_kv_cache();

</script>

![](.)
*Representation of causal self-attention with KV cache during decoding phase. Each timestep $\mathbf{K}$ and $\mathbf{V}$ computed for the last token are added to the cache and re-used in the future steps.*

Let's get flops and memory accesses count for text generation at each generation step (we can simply multiply these counts by $L$ steps                          to get values for the whole sequence).

* Compute:

	- To compute query $\mathbf{Q}$ we multiply input vector $x \in \mathbb{R}^{d}$ with matrices $\mathbf{W}^Q_{1 \dots h} \in \mathbb{R}^{d \times \frac{d}{h}}$ across $h$ heads which takes $\mathcal{O}(d^2)$. The same is for $\mathbf{K}$ and $\mathbf{V}$ when we store the KV cache.
	- Attention computation requires at most $\mathcal{O}(Ld)$ for both $\mathbf{S} = \mathbf{QK}^T$ and $\mathbf{O}=\mathbf{PV}$.
	
	In total we need to perform $\mathcal{O}(d^2 + Ld)$ operations for each step. The number of operations stays the same for $L$ steps as in prefill stage.

* Memory:
	
	- Input $x$ and intermediate tensors $\mathbf{Q}$, $\mathbf{O}$ occupy $\mathcal{O}(d)$ bytes, but $\mathbf{K}$ and $\mathbf{V}$ in cache require $\mathcal{O}(Ld)$ space.
	- Attention logits $\mathbf{S}$ and weights $\mathbf{P}$ take at most $\mathcal{O}(Lh)$ bytes across all heads.
	- Projection weights $\mathbf{W}^{Q,K,V}_{1 \dots h}$ take again $\mathcal{O}(d^2)$ bytes.
	
	Hence in total we need $\mathcal{O}(Ld + Lh + d^2)$ bytes. And finally

$$\text{arithmetic intensity} \propto \frac{Ld + d^2}{Ld + Lh + d^2} < 1,$$

which is definitely smaller than `ops:byte` ratio and we end up memory-bound. Although we reduced the amount of operations by a factor of $L$ by removing $L-1$ queries, the number of memory accesses did not decrease as much. The reason for that is that at each step we retrieve all the values ​​from the KV cache, the size of which increases in proportion to the length of the sequence. 

And that brings us to another drawback - the need to store KV cache requires a lot of HBM capacity, e.g. when we launch a decoding on a transformer with $n$ layers, we need to store $\mathcal{O}(Lnd)$ bytes of KV cache. Thus we either need to make sure that we have enough of memory to accommodate it or to load it from CPU DRAM, which is one or two orders of magnitude slower compared to reading from HBM.

A real world example: take A100 GPU with $80$ GB of HBM. Say, we work with GPT-3 model (I believe it can be considered *large* yet these days) with $n=96$ and $d=12,288$ and try to fit in the context of length $L=4096$. Then the space we need additionally is

$$\underset{\mathbf{K/V}}{2} \cdot \underset{\text{float16}}{2} \cdot \underset{\text{sequence length}}{4096} \cdot \underset{\text{number of layers}}{96} \cdot \underset{\text{embedding dimension}}{12,288} = \underset{\text{bytes}}{19,327,352,832}.$$

So $18$ GB or $22.5\%$ of A100 memory space is required for KV cache of just one sequence sample. Given that [the majoriy of the GPU memory would be taken by model parameters](https://kipp.ly/transformer-param-count/), we can conclude that even without enlarging our batch size we may quickly run out of memory.

### Multi-/Grouped-query attention

In the standard attention mechanism, the $\mathbf{KV}$ pairs are computed for each vector $\mathbf{Q}$ independently. This means that for each token in the input sequence, a separate key-value pair is computed and cached. However, in many cases, different query vectors may share similar attention patterns, and thus, the corresponding keys and values could be reused across multiple queries. **Multi-query attention (MQA)**([Shazeer, 2019](https://arxiv.org/pdf/1911.02150)) shares the cached key-value pairs across multiple queries, thus substantially reducing the memory requirements associated with the KV cache during text generation. 

MQA not only lowers memory consumption, but it also leads to higher inference throughput. Our computational complexity doesn't change, because from algorithmic point of view the number of matrix multiplications stays the same and we only reuse $\mathbf{K}$ and $\mathbf{V}$ for different heads. But in terms of memory, KV cache requires only $\mathcal{O}\big( L n \frac{d}{h} \big)$ space now and arithmetic intensity is proportional to

$$\frac{Ld + d^2}{L\frac{d}{h} + Lh + d^2} \xrightarrow[L \gg d]{} \frac{dh}{d+h^2} \approx h,$$

meaning it is steadily growing with increasing $L$ until it reaches the plateau of few orders of magnitude. We might be still memory-bound in most cases, but with multi-query attention technique we can

- reduce the size of KV cache by a factor of $h$ and
- potentially make decoding algorithm $h$ times faster. 

<div id="arithmetic_intensity_mqa" class="svg-container" align="center"></div> 

<script>

function arithmetic_intensity_mqa() {

	var svg = d3.select("#arithmetic_intensity_mqa")
				  .append("svg")
				  .attr("width", 500)
				  .attr("height", 200);
				  
	const x_start = 80, y_start = 10;
	const x_end = 450, y_end = 180;
	const dim = 12288, h = 96;
	
	text_(svg, "arithmetic", 0, y_start, size=11);
	text_(svg, "intensity", 5, y_start + 10, size=11);
	text_(svg, "L", x_end + 20, y_end + 5, size=11);
	text_(svg, "MHA", x_end + 20, y_end - 30, size=11);
	text_(svg, "MQA", x_end + 20, y_end - 110, size=11);
	
	var ai_x = d3.scaleLinear()
        .domain([0, 30000])
        .range([x_start, x_end]);
        
  	var xAxis = svg.append("g")
      .attr("transform", "translate(0," + y_end + ")")
      .call(d3.axisBottom(ai_x).ticks(4));
      
   xAxis.selectAll(".tick text")
	   .attr("font-family", "Arvo");
	
	var ai_y = d3.scaleLinear()
	          .range([y_end, y_start])
	          .domain([0, 5]);
	          
	var yAxis = svg.append("g")
	   .attr("transform", "translate(" + x_start + ", 0)")
	   .call(d3.axisLeft(ai_y).ticks(3));
	   
	yAxis.selectAll(".tick text")
	    .attr("font-family", "Arvo");
	    
   var mha_data = [];
   for (var l = 0; l < 30000; l += 10) {
   		var lsq = l * l, dsq = dim * dim;
   		mha_data.push({x: l, y: (l * dim + dsq) / (l * dim + l * h + dsq) });
	};
	    
   var mqa_data = [];
   for (var l = 0; l < 30000; l += 10) {
   		var lsq = l * l, dsq = dim * dim;
   		mqa_data.push({x: l, y: (l * dim + dsq) / (l * dim / h + l * h + dsq) });
	};
	
	svg
	    .append('g')
	    .append("path")
	      .datum(mha_data)
	      .attr("fill", "none")
	      .attr("border", 0)
	      .attr("stroke", "currentColor")
	      .attr("stroke-width", 2)
	      .attr("stroke-linejoin", "round")
	      .style('stroke-dasharray', ('5,2'))
	      .attr("d",  d3.line()
	        .curve(d3.curveBasis)
	          .x(function(d) { return ai_x(d.x); })
	          .y(function(d) { return ai_y(d.y); })
	      );
	      
	svg
	    .append('g')
	    .append("path")
	      .datum(mqa_data)
	      .attr("fill", "none")
	      .attr("border", 0)
	      .attr("stroke", "currentColor")
	      .attr("stroke-width", 2)
	      .attr("stroke-linejoin", "round")
	      .attr("d",  d3.line()
	        .curve(d3.curveBasis)
	          .x(function(d) { return ai_x(d.x); })
	          .y(function(d) { return ai_y(d.y); })
	      );
	      
	var l_x = d3.scaleLinear()
	    .domain([1, 10])
	    .range([0, 320])
	    .clamp(true);

	createSlider(svg, draw_attn_cells, l_x, x_start, 280, "L", "currentColor", init_seq_len, roundN, 10);
	
}

arithmetic_intensity_mqa();

</script>

![](.)
*Arithmetric intensity vs sequence length $L$ for multi-query and multi-head self-attention mechanisms during auto-regressive generation with embedding dimension $d=12,288$ and number of heads $h=96$.*

In our example above with GPT-3 model, using MQA would make KV cache $h=96$ times smaller, thus the required space would take around $200$ MB which is just $0.25\%$ of one A100 GPU memory.

Of course, such acceleration and memory reduction come with a price - we cut model parameters and therefore its potential capacity. The possible way to avoid quality degradation is to use technique, which interpolates between MHA and MQA - **grouped query attention (GQA)** ([Ainslie et al. (2023)](https://arxiv.org/pdf/2305.13245)). With GQA we split $h$ query heads into $g$ groups, each with its own keys and values. Note that for $g=1$ GQA is equal to multi-query and for $g=h$ GQA is the same as multi-head attention. The choice of $g$ is a trade-off between memory savings and potential accuracy loss. A larger group size will result in more memory savings but may also lead to a larger approximation error in the attention computations. In practice, the optimal group size may need to be determined empirically based on the specific model architecture and the trade-off between memory efficiency and model performance.

<div id="groupquery_attention" class="svg-container" align="center"></div> 

<script>

function gqa_line(svg, x1, y1, x2, y2) {
	svg.append('g')
	    .append("path")
	      .datum([{x: x1, y: y1}, {x: x2, y: y2}])
	      .attr("fill", "none")
	      .attr("border", 0)
	      .attr("stroke", "currentColor")
	      .attr("stroke-width", 2)
	      .attr("stroke-linejoin", "round")
	      .style('stroke-dasharray', ('2,3'))
	      .attr("d",  d3.line()
	        .curve(d3.curveBasis)
	          .x(function(d) { return d.x; })
	          .y(function(d) { return d.y; })
	      );
}

function groupquery_attention() {
	var svg = d3.select("#groupquery_attention")
				  .append("svg")
				  .attr("width", 800)
				  .attr("height", 255);
				  
	const x_start = 55, y_start = -5;
	const shift = 10, rct_sz = 8;
	const dim = 2, num_heads = 6, num_kv_heads = 3, seq_len = 6;
	var num_heads_per_part = Math.floor(num_heads / num_kv_heads);
	
	text_(svg, "Value", 1, y_start + 60, size=12);
	text_(svg, "heads", 1, y_start + 80, size=12);
	text_(svg, "Key", 6, y_start + 140, size=12);
	text_(svg, "heads", 1, y_start + 160, size=12);
	text_(svg, "Query", 0, y_start + 220, size=12);
	text_(svg, "heads", 1, y_start + 240, size=12);
	
	text_(svg, "Multi-head", x_start + 54, 10, size=12);
	text_(svg, "Grouped-query", x_start + 271, 10, size=12);
	text_(svg, "Multi-query", x_start + 506, 10, size=12);
	
	for (var i = 0; i != num_heads; i += 1) {
		vector_rect(svg, x_start + 3 * i * shift, y_start + 200, dim, seq_len, shift, rct_sz, matrix_colors[0]);
		
		gqa_line(svg, x_start + 3 * i * shift + rct_sz + 1, y_start + 180, x_start + 3 * i * shift + rct_sz + 1, y_start + 198);
		
		vector_rect(svg, x_start + 3 * i * shift, y_start + 120, dim, seq_len, shift, rct_sz, matrix_colors[1]);
		
		gqa_line(svg, x_start + 3 * i * shift + rct_sz + 1, y_start + 100, x_start + 3 * i * shift + rct_sz + 1, y_start + 118);
		
		vector_rect(svg, x_start + 3 * i * shift, y_start + 40, dim, seq_len, shift, rct_sz, matrix_colors[3]);
	}
	
	for (var i = 0; i != num_heads; i += 1) {
		vector_rect(svg, x_start + 230 + 3 * i * shift, y_start + 200, dim, seq_len, shift, rct_sz, matrix_colors[0]);
	}
	
	for (var i = 0; i != num_kv_heads; i += 1) {
		vector_rect(svg, x_start + 245 + 6 * i * shift, y_start + 120, dim, seq_len, shift, rct_sz, matrix_colors[1]);
		vector_rect(svg, x_start + 245 + 6 * i * shift, y_start + 40, dim, seq_len, shift, rct_sz, matrix_colors[3]);
		gqa_line(svg, x_start + 246 + 6 * i * shift + rct_sz, y_start + 100, x_start + 246 + 6 * i * shift + rct_sz, y_start + 118);
		gqa_line(svg, x_start + 246 + 6 * i * shift + rct_sz, y_start + 180, x_start + 230 + 6 * i * shift + rct_sz, y_start + 198);
		gqa_line(svg, x_start + 246 + 6 * i * shift + rct_sz, y_start + 180, x_start + 251 + (6 * i + 1) * shift + rct_sz, y_start + 198);
	}
	
	for (var i = 0; i != num_heads; i += 1) {
		vector_rect(svg, x_start + 460 + 3 * i * shift, y_start + 200, dim, seq_len, shift, rct_sz, matrix_colors[0]);
		gqa_line(svg, x_start + 544, y_start + 180, x_start + 471 + 3 * i * shift, y_start + 198);
	}
	
	vector_rect(svg, x_start + 535, y_start + 120, dim, seq_len, shift, rct_sz, matrix_colors[1]);
	vector_rect(svg, x_start + 535, y_start + 40, dim, seq_len, shift, rct_sz, matrix_colors[3]);
	gqa_line(svg, x_start + 544, y_start + 100, x_start + 544, y_start + 118);
		
}

groupquery_attention();

</script>

![](.)
*MHA (with number of heads $h=6$) vs GQA (with number of groups $g=3$) vs MQA*

```python
@jit
def gqa_dot_product_attention(query, key, value, mask=None):
    num_heads, num_kv_heads = query.shape[-2], key.shape[-2]
    # broadcast K/V heads to match number of Q heads
    num_heads_per_kv = num_heads // num_kv_heads
    key = jnp.repeat(key, num_heads_per_kv, axis=-2)
    value = jnp.repeat(value, num_heads_per_kv, axis=-2)
    return dot_product_attention(query, key, value, mask)
```

Below is a comparison table with batched decoding/inference algorithms complexities for input $x \in \mathbb{R}^{B \times L \times d}$ and large context size $L$. Note, that the computation complexity is the same for all algorithms in the table! However, the real effectiveness can vary greatly depending on the setting.


|    | Vanilla Attention | Attention with KV Cache | GQA with KV Cache | 
| -------- | ------- | ------- | ------- | ------- |
| Compute performance  | $BLd(d + L) $ | $BLd(d + L)$ | $BLd(d + L)$ 
| Memory bandwidth | $BL (d + hL) + d^2$ | $BL^2(d + h) + d^2$ |$BL^2\big(\frac{d}{g} + h \big) + d^2$ 
| KV cache size | $0$ | $B L n d $ | $BL n \frac{d}{g} $ 
| Arithmetic intensity for sufficiently large $L$ | $\frac{d}{h}$  | $\frac{d}{d + h}$ | $\frac{dg}{d+hg}$ 

### Multi-head latent attention

Another way to reduce KV cache bottleneck is **multi-head latent attention (MLA)** introduced with [DeepSeek V2](https://arxiv.org/pdf/2405.04434) model. In standard multi-head self-attention mechanism all queries, keys and values are calculated as linear projections of input embedding $x \in \mathbb{R}^{d}$ with learnable parameters $\mathbf{W}^Q_{1 \dots h}$, $\mathbf{W}^K_{1 \dots h}$, $\mathbf{W}^V_{1 \dots h}$ respectively. In addition to that the low-rank joint compression latent vector $\mathbf{C}$ is computed for both keys and values in MLA:

$$\mathbf{C^{KV}} = x \mathbf{W}_{d}^{KV}$$

with down-projection matrix $\mathbf{W}_{d}^{KV} \in \mathbb{R}^{d \times d_c}$ ($d_c \ll d$). Vector $\mathbf{C^{KV}} \in \mathbb{R}^{d_c}$ called **compressed latent vector** and is stored in KV cache instead of full-sized values of $\mathbf{K}$ and $\mathbf{V}$, reducing its size by a factor of $\frac{d}{d_c}$. When it comes to retrieval, keys and values can be restored as

$$\mathbf{K} = \mathbf{C^{KV}} \mathbf{W}_{u}^{K}, \quad \mathbf{V} = \mathbf{C^{KV}} \mathbf{W}_{u}^{V}$$

with up-projection matrices $\mathbf{W}_{u}^{K}$, $\mathbf{W}_{u}^{V} \in \mathbb{R}^{d_c \times d}$. Furthermore, authors of MLA also performed low-rank compression for the queries:

$$\mathbf{Q} = \mathbf{C^{Q}} \mathbf{W}_{u}^{Q}, \quad \mathbf{C^{Q}} = x \mathbf{W}_{d}^{Q}$$

with up- and down-projection matrices $\mathbf{W}_{u}^{Q} \in \mathbb{R}^{d'_c \times d}$ and $\mathbf{W}_{d}^{Q} \in \mathbb{R}^{d \times d'_c}$ and the query compression dimension $d'_c$. While it doesn't affect KV cache, it reduces the activation memory during training.

<div id="multilatent_attention" class="svg-container" align="center"></div> 

<script>

function gqa_line(svg, x1, y1, x2, y2) {
	svg.append('g')
	    .append("path")
	      .datum([{x: x1, y: y1}, {x: x2, y: y2}])
	      .attr("fill", "none")
	      .attr("border", 0)
	      .attr("stroke", "currentColor")
	      .attr("stroke-width", 2)
	      .attr("stroke-linejoin", "round")
	      .style('stroke-dasharray', ('2,3'))
	      .attr("d",  d3.line()
	        .curve(d3.curveBasis)
	          .x(function(d) { return d.x; })
	          .y(function(d) { return d.y; })
	      );
}

function multilatent_attention() {
	var svg = d3.select("#multilatent_attention")
				  .append("svg")
				  .attr("width", 800)
				  .attr("height", 255);
				  
	const x_start = 205, y_start = -5;
	const shift = 10, rct_sz = 8;
	const dim = 2, num_heads = 6, num_kv_heads = 3, seq_len = 6;
	var num_heads_per_part = Math.floor(num_heads / num_kv_heads);
	
	text_(svg, "Value", 151, y_start + 60, size=12);
	text_(svg, "heads", 151, y_start + 80, size=12);
	text_(svg, "Key", 156, y_start + 140, size=12);
	text_(svg, "heads", 151, y_start + 160, size=12);
	text_(svg, "Query", 150, y_start + 220, size=12);
	text_(svg, "heads", 151, y_start + 240, size=12);
	
	text_(svg, "Multi-head latent", x_start + 33, y_start + 15, size=12);
	
	text_(svg, "projection", x_start + 200, y_start + 110, size=10);
	text_(svg, "⟺", x_start + 217, y_start + 117, size=15);
	
	text_(svg, "Compressed", x_start + 300, y_start + 93, size=12);
	text_(svg, "latent vector", x_start + 300, y_start + 113, size=12);
	text_(svg, "(KV cache)", x_start + 305, y_start + 133, size=12);
	
	for (var i = 0; i != num_heads; i += 1) {
		vector_rect(svg, x_start + 3 * i * shift, y_start + 200, dim, seq_len, shift, rct_sz, matrix_colors[0]);
		
		gqa_line(svg, x_start + 3 * i * shift + rct_sz + 1, y_start + 180, x_start + 3 * i * shift + rct_sz + 1, y_start + 198);
		
		vector_rect(svg, x_start + 3 * i * shift, y_start + 120, dim, seq_len, shift, rct_sz, matrix_colors[1]);
		
		gqa_line(svg, x_start + 3 * i * shift + rct_sz + 1, y_start + 100, x_start + 3 * i * shift + rct_sz + 1, y_start + 118);
		
		vector_rect(svg, x_start + 3 * i * shift, y_start + 40, dim, seq_len, shift, rct_sz, matrix_colors[3]);
	}
		
	var c_shift = 3 * (num_heads + 3) * shift;
	vector_rect(svg, x_start + c_shift, y_start + 80, 1, seq_len, shift, rct_sz, matrix_colors[3]);
	vector_rect(svg, x_start + c_shift + rct_sz + 2, y_start + 80, 1, seq_len, shift, rct_sz, matrix_colors[1]);
	
	svg.append('rect')
	  .attr('x', x_start + c_shift - 3)
	  .attr('y', y_start + 77)
	  .attr('width', 2 * shift + 4)
	  .attr('height', seq_len * shift + 4)
	  .attr('stroke', 'currentColor')
	  .attr('stroke-width', 2)
	  .attr("rx", 3)
	  .style('stroke-dasharray', ('2,3'))
	  .attr('fill', 'none');
	  
	svg.append("path")
	   .datum([{x: x_start + c_shift - 5, y: y_start + 75},
	           {x: x_start + 3 * num_heads * shift, y: y_start + 40}])
	   .attr("fill", "none")
	   .attr("stroke-width", 1)
	   .attr("stroke", "currentColor")
	   .attr("d",  d3.line()
	       .x(function(d) { return d.x; })
	       .y(function(d) { return d.y; }));
	       
	svg.append("path")
	   .datum([{x: x_start + c_shift - 5, y: y_start + 145},
	           {x: x_start + 3 * num_heads * shift, y: y_start + 180}])
	   .attr("fill", "none")
	   .attr("stroke-width", 1)
	   .attr("stroke", "currentColor")
	   .attr("d",  d3.line()
	       .x(function(d) { return d.x; })
	       .y(function(d) { return d.y; }));
}

multilatent_attention();

</script>

![](.)
*Intuitive illustration of how the KV joint compression in MLA reduces the KV cache.*

##### Decoupled Rotary Position Embedding

To improve the efficiency of MLA, a *weight-absorption trick* can be applied: since attention logits $\mathbf{S}$ for context tokens are

$$
\mathbf{S} = \mathbf{Q}\mathbf{K}^T = x {\color{Salmon}{\mathbf{W}^Q{\mathbf{W}_{u}^{K}}^T}} \mathbf{C^{KV}}^T,
$$

matrices $\mathbf{W}^Q {\mathbf{W}_{u}^{K}}^T$ can be merged together and there is no need to materialize full tensor of $\mathbf{K}$ from KV cache during inference. The same is true for $\mathbf{V}$, since weight-absorption can be applied to output:

$$
\mathbf{O} \mathbf{W}^O = \mathbf{PV} \mathbf{W}^O = \mathbf{PC} {\color{Salmon}{\mathbf{W}_{u}^{V}\mathbf{W}^O}},
$$

The problem with low-rank KV compression is that it is incompatible with commonly used [rotary positional embeddings transformation](https://arxiv.org/pdf/2104.09864), which is position-sensitive for both keys and queries. To be specific, when we apply RoPE transformation $\mathbf{R}_{\Theta}$

$$\mathbf{Q}_{\text{rope}} = \mathbf{Q}\mathbf{R}_{\Theta}, \quad \mathbf{K}_{\text{rope}} = \mathbf{K}\mathbf{R}_{\Theta},$$

for query token $\mathbf{q}_m$ at position $m$ and key token $\mathbf{k}_n$ at position $n$ the attention logit is

$$
\begin{aligned}
\mathbf{q}_m\mathbf{k}_n^T &= \big(x_m \mathbf{W}^{Q} \mathbf{R}_{\Theta, m}\big) \big(x_n\mathbf{W}^{K}\mathbf{R}_{\Theta, n}\big)^T \\ &=x_m \mathbf{W}^{Q} {\color{Salmon}{\mathbf{R}_{\Theta, m} \mathbf{R}_{\Theta, n}}} {\mathbf{W}^{K}}^T x_n^T \\ &= x_m \mathbf{W}^{Q} {\color{Salmon}{\mathbf{R}_{\Theta, n - m}}} \mathbf{W}^{K} x_n^T.
\end{aligned}$$

Now $\mathbf{W}_{u}^{K}$ cannot be absorbed into $\mathbf{W}^Q$ any more during inference, since positional-dependent matrix $\mathbf{R}_{\Theta, n - m}$ lies in-between. As a solution, the decoupled RoPE strategy is proposed: query and a shared key for each head are split along embedding dimension in two parts

$$
\begin{aligned}
\mathbf{Q} = [\mathbf{Q}_{\text{rope}}, \mathbf{Q}_{\text{nope}}], &\quad \mathbf{Q}_{\text{rope}} \in \mathbf{R}^{L \times d_R}, \mathbf{Q}_{\text{nope}} \in \mathbf{R}^{L \times d_h} \\ \mathbf{K} = [\mathbf{K}_{\text{rope}}, \mathbf{K}_{\text{nope}}], & \quad \mathbf{K}_{\text{rope}} \in \mathbf{R}^{L \times d_R}, \mathbf{K}_{\text{nope}} \in \mathbf{R}^{L \times d_h}
\end{aligned}$$

with $d_h + d_R$ - total dimension for each head. During inference, the decoupled key should also be cached, therefore, MLA requires a total KV cache containing $d_c + d_R$ elements for each token in each layer (would be $B L n (d_c + d_R) $ in the table above).

### Chunked prefill

A large KV cache is not the only source of memory problems as the sequence length increases. During prefill phase we compute all the outputs and $\mathbf{K}\mathbf{V}$ pairs in one pass. This requires us to compute the attention matrix $\mathbf{S} \in \mathbb{R}^{L \times L}$, which depends quadratically on the context length. What if the prompt size is so large that we can't fit in all attention weights? We can compute them by passing single tokens one-by-one as we do it in decoding stage, though this procedure is much slower since it's memory-bound. But since we know future tokens in advance, we can feed them to model in **chunks**:

- Take first $L_c$ tokens ($L_c < L$) from the prompt, run them through the prefill stage and store their KV cache values. Attention weights will be $\mathbf{S} \in \mathbb{R}^{L_c \times L_c}$.
- Then apply the same procedure for the next $L_c$ tokens, but now use cached $\mathbf{KV}$ pairs to attend to the tokens in a previous chunk. Attention weights then will be $\mathbf{S} \in \mathbb{R}^{L_c \times 2L_c}$.
- Repeat until the whole prompt is prefilled. The maximum size of $\mathbf{S}$ at the end will be ${L_c \times L_c}$.

<div id="chunked_prefill" class="svg-container" align="center"></div> 

<script>

function chunked_prefill() {
	var svg = d3.select("#chunked_prefill")
				  .append("svg")
				  .attr("width", 700)
				  .attr("height", 290);
				  
	const x_start = 70, y_start = 30;
	const shift = 14, rct_sz = 12;
	var init_seq_len = 10;
	const chunk_size = 4, dim = 2;
	
	function draw_attn_cells_w_chunking(chunk_id) {
		var seq_len = Math.min(chunk_id * chunk_size, init_seq_len);
	   svg.selectAll('rect').remove();
		kv_width = Math.round(seq_len / chunk_size - 1) * chunk_size;
			
		vector_rect(svg, x_start, y_start + 3 * shift, dim, kv_width, shift, rct_sz, matrix_colors[0], 0.1);
		vector_rect(svg, x_start, y_start + (kv_width + 3) * shift, dim, seq_len - kv_width, shift, rct_sz, matrix_colors[0]);
		
		vector_rect(svg, x_start + 3 * shift, y_start, seq_len, dim, shift, rct_sz, matrix_colors[1]);
		
		var max_i = Math.round(seq_len / chunk_size);
		for (var i = 0; i < max_i; i += 1) {
			var opacity = (i + 1 >= max_i) ? 1 : 0.1;
			var height = (i + 1 >= max_i) ? seq_len - kv_width : chunk_size;
			vector_rect(svg, x_start + 3 * shift, y_start + (3 + i * chunk_size) * shift, Math.min((i + 1) * chunk_size, seq_len), height, shift, rct_sz, matrix_colors[2], opacity);
			
			causal_mask(svg, x_start + 16 * shift, y_start + (3 + i * chunk_size) * shift, Math.min((i + 1) * chunk_size, seq_len), height, shift, rct_sz, matrix_colors[7], chunk_size, i * chunk_size, opacity);
			
		}
		
		vector_rect(svg, x_start + 30 * shift, y_start + 3 * shift, dim, seq_len, shift, rct_sz, matrix_colors[3]);
		
		vector_rect(svg, x_start + 35 * shift, y_start + 3 * shift, dim, kv_width, shift, rct_sz, matrix_colors[4], 0.1);
		vector_rect(svg, x_start + 35 * shift, y_start + (kv_width + 3) * shift, dim, seq_len - kv_width, shift, rct_sz, matrix_colors[4]);
		
		if (seq_len > chunk_size) {
			svg.append('rect')
			  .attr('x', x_start + 3 * shift - 3)
			  .attr('y', y_start - 3)
			  .attr('width', kv_width * shift + 2)
			  .attr('height', 2 * shift + 4)
			  .attr('stroke', 'currentColor')
			  .attr('stroke-width', 2)
			  .attr("rx", 3)
			  .style('stroke-dasharray', ('2,3'))
			  .attr('fill', 'none');
			
			svg.append('rect')
			  .attr('x', x_start + 30 * shift - 3)
			  .attr('y', y_start + 3 * shift - 3)
			  .attr('width', 2 * shift + 4)
			  .attr('height', kv_width * shift + 2)
			  .attr('stroke', 'currentColor')
			  .attr('stroke-width', 2)
			  .attr("rx", 3)
			  .style('stroke-dasharray', ('2,3'))
			  .attr('fill', 'none');
		}
		
		svg.append('rect')
		  .attr('x', x_start + 40 * shift - 3)
		  .attr('y', y_start - 3)
		  .attr('width', 2 * shift + 4)
		  .attr('height', 2 * shift + 4)
		  .attr('stroke', 'currentColor')
		  .attr('stroke-width', 2)
		  .attr("rx", 3)
		  .style('stroke-dasharray', ('2,3'))
		  .attr('fill', 'none');
		  
		vector_rect(svg, x_start + 40 * shift, y_start, dim, 1, shift, rct_sz, matrix_colors[1]);
		vector_rect(svg, x_start + 40 * shift, y_start + shift, dim, 1, shift, rct_sz, matrix_colors[3]);
		}

	draw_sampling_text(svg, x_start, y_start, rct_sz, shift);
	draw_attn_cells_w_chunking(init_seq_len);
	
	text_(svg, "KV cache", x_start + 39 * shift, y_start - 10, size=11);

	var l_x = d3.scaleLinear()
		 .domain([1, 3])
	    .range([0, 320])
	    .clamp(true);

	createSlider(svg, draw_attn_cells_w_chunking, l_x, x_start + 120, 270, "Chunk", "currentColor", init_seq_len, roundN, 2, 30, 12);
}

chunked_prefill();

</script>

![](.)
*Chunked prefill with $L_c = 4$ and prompt length $10$.*

With chunked prefill the maximum size of $\mathbf{S}$ depends linearly on $L$ multiplied by controllable constant coefficient $L_c$. 

### Sliding window attention

We can see that even with multi-query attention and chunked prefill our KV cache and attention weights are still increasing with growing context during both prefill phase and decoding phase. If we truncate the amount of tokens each other token can attend to by some constant $L_w$, our memory requirements will not depend on the input sequence length. This is exactly what a technique called **sliding window attention** does: it changes attention mask from lower-diagonal matrix to a band matrix.

<div id="sliding_window" class="svg-container" align="center"></div> 

<script>

function sliding_window() {
	var svg = d3.select("#sliding_window")
				  .append("svg")
				  .attr("width", 300)
				  .attr("height", 160);
	const x_start = 100, y_start = 20;
	const shift = 14, rct_sz = 12;
	text_(svg, "mask", x_start + 4.2 * rct_sz, 10);
	causal_mask(svg, x_start, y_start, 10, 10, shift, rct_sz, matrix_colors[7], 4);
}

sliding_window();

</script>

![](.)
*Mask for attention weights with $L=10$ and sliding window $L_w=4.$*

This makes attention layer focus only on local context. But notice that tokens can implicitly attend to previous $n \cdot L_w$ tokens, where $n$ is a number of layers in our transformer model. This is very similar to how receptive field works in convolutional networks. Choosing the optimal window size involves a trade-off between memory efficiency and maintaining context. Larger window size preserve more context but require more memory, while smaller window size is more memory-efficient but it may lose some context.

When we use sliding window attention, we might use **rolling KV cache** as well. As KV cache is now restricted by a given constant $2 \cdot L_w$ and only one $\mathbf{KV}$ pair is changed at each step, we can remove pair related the oldest token that we won't attend to anymore and replace it with a newest one. In practice we keep the write pointer at the oldest pair and move it by one after its replacement. When we reach the end of the buffer, we move it back to the first position.

<div id="text_generation_w_rolling_kv_cache" class="svg-container" align="center"></div> 

<script>

function text_generation_w_rolling_kv_cache() {
	var svg = d3.select("#text_generation_w_rolling_kv_cache")
				  .append("svg")
				  .attr("width", 550)
				  .attr("height", 300);
				  
	const x_start = 110, y_start = 30;
	const shift = 14, rct_sz = 12;
	var init_seq_len = 10;
	const window_size = 4, dim = 2;
	
	function draw_attn_cells_w_kv_cache(seq_len) {
	   svg.selectAll('rect').remove();
		vector_rect(svg, x_start, y_start + 3 * shift, dim, seq_len - 1, shift, rct_sz, matrix_colors[0], 0.1);
		vector_rect(svg, x_start, y_start + (seq_len + 2) * shift, dim, 1, shift, rct_sz, matrix_colors[0]);
		
		vector_rect(svg, x_start + Math.max(seq_len - window_size + 3, 3) * shift, y_start, Math.min(seq_len, window_size), dim, shift, rct_sz, matrix_colors[1]);
		
		for (var i = 1; i < seq_len; i += 1) {
			vector_rect(svg, x_start + Math.max(i - window_size + 3, 3) * shift, y_start + (i + 2) * shift, Math.min(i, window_size), 1, shift, rct_sz, matrix_colors[2], 0.1);
		}
		
		vector_rect(svg, x_start + Math.max(seq_len - window_size + 3, 3) * shift, y_start + (seq_len + 2) * shift, Math.min(seq_len, window_size), 1, shift, rct_sz, matrix_colors[2]);
		
		vector_rect(svg, x_start + 17 * shift, y_start + Math.max(seq_len - window_size + 3, 3) * shift, dim, Math.min(seq_len, window_size), shift, rct_sz, matrix_colors[3]);
		
		vector_rect(svg, x_start + 21 * shift, y_start + 3 * shift, dim, seq_len - 1, shift, rct_sz, matrix_colors[4], 0.1);
		vector_rect(svg, x_start + 21 * shift, y_start + (seq_len + 2) * shift, dim, 1, shift, rct_sz, matrix_colors[4]);
		
		if (seq_len > 1) {
			svg.append('rect')
			  .attr('x', x_start + Math.max(seq_len - window_size + 3, 3) * shift - 3)
			  .attr('y', y_start - 3)
			  .attr('width', (Math.min(seq_len, window_size) - 1) * shift + 2)
			  .attr('height', 2 * shift + 4)
			  .attr('stroke', 'currentColor')
			  .attr('stroke-width', 2)
			  .attr("rx", 3)
			  .style('stroke-dasharray', ('2,3'))
			  .attr('fill', 'none');
			
			svg.append('rect')
			  .attr('x', x_start + 17 * shift - 3)
			  .attr('y', y_start + Math.max(seq_len - window_size + 3, 3) * shift - 3)
			  .attr('width', 2 * shift + 4)
			  .attr('height', (Math.min(seq_len, window_size) - 1) * shift + 2)
			  .attr('stroke', 'currentColor')
			  .attr('stroke-width', 2)
			  .attr("rx", 3)
			  .style('stroke-dasharray', ('2,3'))
			  .attr('fill', 'none');
		}
		
		svg.append('rect')
		  .attr('x', x_start + 28 * shift - 3)
		  .attr('y', y_start - 3)
		  .attr('width', 2 * shift + 4)
		  .attr('height', 2 * shift + 4)
		  .attr('stroke', 'currentColor')
		  .attr('stroke-width', 2)
		  .attr("rx", 3)
		  .style('stroke-dasharray', ('2,3'))
		  .attr('fill', 'none');
		  
		vector_rect(svg, x_start + 28 * shift, y_start, dim, 1, shift, rct_sz, matrix_colors[1]);
		vector_rect(svg, x_start + 28 * shift, y_start + shift, dim, 1, shift, rct_sz, matrix_colors[3]);
		
		init_seq_len = seq_len;
	}

	draw_sampling_text_2(svg, x_start, y_start, rct_sz, shift);
	draw_attn_cells_w_kv_cache(init_seq_len);

	var l_x = d3.scaleLinear()
	    .domain([1, 10])
	    .range([0, 320])
	    .clamp(true);

	createSlider(svg, draw_attn_cells_w_kv_cache, l_x, x_start, 270, "L", "currentColor", init_seq_len, roundN, 10);
}

text_generation_w_rolling_kv_cache();

</script>

![](.)
*Decoding with sliding window attention with $L_w=4$.*

Another advantage of sliding window attention is that combining it with chunked prefill does not only keep maximum size of attention matrix constant ($\mathbf{S} \in \mathbb{R}^{L_c \times L_w}$), but also reduces the number of dot-products to compute it.

The drawback of sliding window attention is that it may lead to degradation as not all interactions between tokens are captured. An interesting phenomenon was found by [Xiao et al. (2024)](https://arxiv.org/pdf/2309.17453), which they called **attention sink**: keeping the $\mathbf{KV}$ of a small number of tokens in the beginning of the sequence will largely recover the performance of window attention. They observe that LLMs outputs strong attention scores towards initial tokens as a "sink" even if they are not semantically important.


### Linear attention

**Linear attention** mechanism ([Katharopoulos et al. (2020)](https://arxiv.org/pdf/2006.16236)) is an alternative family of methods to avoid $\mathcal{O}(L^2)$ scaling for long sequences. Linear attention approximates the standard attention mechanism while achieving linear time and space complexity. The key idea is in reformulating the attention operation using the associative property of matrix multiplication and **kernel functions**. 

Kernel function $\mathcal{K}(q, k)$ can be thought of as similarity measure between pair of inputs $q$ and $k$, exactly like in any attention mechanism. To simplify computation kernel function is oftenly chosen so that it can be represented in the form of a **feature map** $\phi$:

$$\mathcal{K}(q, k) = \phi(q)^T \phi(k).$$

If we find such feature map, it would allow us to implicitly compute similarities between queries and keys without explicitly computing the full attention matrix $\mathbf{QK^T}$.

In attention mechanism unnormalized similarity between query embedding $\mathbf{q}$ and key embedding $\mathbf{k}$ is measured as $\mathcal{K}(\mathbf{q}, \mathbf{k}) = \exp \big( \frac{\mathbf{q}^T\mathbf{k}}{\sqrt{d}} \big)$. Each element of softmax masked attention matrix $\mathbf{P} = \operatorname{softmax}(\mathbf{S} \circ \operatorname{mask})$, the normalized similarity between query row $\mathbf{Q}_i$ and key row $\mathbf{K}_j$ ($j \leq i$), can be represented as

$$\mathbf{P}_{ij}=\frac{\mathcal{K}(\mathbf{Q}_i, \mathbf{K}_j)}{\sum_{j \leq i} \mathcal{K}(\mathbf{Q}_i, \mathbf{K}_j)}.$$

Using feature maps we can rewrite each row $i$ of dot-product attention output $\mathbf{O} = \mathbf{P} \mathbf{V}$ as

$$
\begin{aligned} 
\mathbf{O}_{i}
&= \sum_{j \leq i} \mathbf{P}_{ij} \mathbf{V}_j \\
&= \frac{\sum_{j \leq i} \mathcal{K}(\mathbf{Q}_i, \mathbf{K}_j) \cdot \mathbf{V}_{j}}{\sum_{j \leq i} \mathcal{K}(\mathbf{Q}_i, \mathbf{K}_j)} \\
&= \frac{\sum_{j \leq i} \phi(\mathbf{Q}_i)^T \phi(\mathbf{K}_j) \cdot \mathbf{V}_{j}}{\sum_{j \leq i} \phi(\mathbf{Q}_i)^T \phi(\mathbf{K}_j) } \\
&= \frac{ \phi(\mathbf{Q}_i)^T \cdot \color{Salmon}{\sum_{j \leq i} \phi(\mathbf{K}_j) \mathbf{V}^T_{j}} }{ \phi(\mathbf{Q}_i)^T \cdot \color{#007BA7}{\sum_{j \leq i} \phi(\mathbf{K}_j)}} \\
&= \frac{ \phi(\mathbf{Q}_i)^T \cdot \color{Salmon}{u_i}}{ \phi(\mathbf{Q}_i)^T \cdot \color{#007BA7}{\mathbf{Z}_i} }.
\end{aligned}
$$

The above equation is simpler to follow when the numerator is written in vectorized form as follows,

$$\big( \phi(\mathbf{Q})\phi(\mathbf{K})^T \big) \mathbf{V} = \phi(\mathbf{Q})\big( \phi(\mathbf{K})^T \mathbf{V} \big).$$

Regardless of the value $L$ we no longer need to store the quadratically growing attention matrix, we only need $\mathcal{O}(d^2)$ space for $u_L = \phi(\mathbf{K})^T \mathbf{V} \in \mathbb{R}^{d \times d}$:

<div id="linear_attention" class="svg-container" align="center"></div> 

<script>

function linear_attention() {
	
	function draw_sampling_text_3(svg, x_start, y_start, rct_sz, shift) {
		text_(svg, "φ(Q)", x_start, y_start + 6 * shift);
		text_(svg, "φ(K)", x_start + 9 * rct_sz, y_start + 10 * shift);
		
		text_(svg, "T", x_start + 9 * rct_sz + 30, y_start + 10 * shift - 8, size=8);
		text_(svg, "U", x_start + 15 * shift + 7, y_start + 200);
		
		text_(svg, "V", x_start + 15 * shift + 8, y_start - 5);
		text_(svg, "O", x_start + 19 * shift + 8, y_start + 6 * shift);
		text_(svg, "(", x_start + 3 * shift, y_start + 12.5 * shift, size=30);
		text_(svg, ")", x_start + 17 * shift, y_start + 12.5 * shift, size=30);
		text_(svg, "⋅", x_start + 2 * shift, y_start + shift * 12.5, size=30);
		text_(svg, "=", x_start + 18 * shift, y_start + shift * 12.25, size=18);
	}
	
	var svg = d3.select("#linear_attention")
				  .append("svg")
				  .attr("width", 550)
				  .attr("height", 300);
				  
	const x_start = 100, y_start = 20;
	const shift = 14, rct_sz = 12;
	var init_seq_len = 10;
	const window_size = 4, dim = 2;
	
	function draw_attn_cells(seq_len) {
	   svg.selectAll('rect').remove();
		vector_rect(svg, x_start, y_start + 7 * shift, dim, seq_len, shift, rct_sz, matrix_colors[0]);
		
		vector_rect(svg, x_start + 4 * shift, y_start + 11 * shift, seq_len, dim, shift, rct_sz, matrix_colors[1]);
		
		vector_rect(svg, x_start + 15 * shift, y_start, dim, seq_len, shift, rct_sz, matrix_colors[3]);
		
		vector_rect(svg, x_start + 15 * shift, y_start + 11 * shift, dim, dim, shift, rct_sz, matrix_colors[2]);
		
		vector_rect(svg, x_start + 19 * shift, y_start + 7 * shift, dim, seq_len, shift, rct_sz, matrix_colors[4]);
		
		init_seq_len = seq_len;
	}
	
	draw_attn_cells(init_seq_len);
	draw_sampling_text_3(svg, x_start, y_start, rct_sz, shift);
		
	var l_x = d3.scaleLinear()
	    .domain([1, 10])
	    .range([0, 320])
	    .clamp(true);

	createSlider(svg, draw_attn_cells, l_x, x_start, 280, "L", "currentColor", init_seq_len, roundN, 10);
}

linear_attention();

</script>

![](.)
*Neither prefill nor decoding phase with linear attention require $\mathcal{O}(L^2)$ space anymore. Scalar denominator $\phi(\mathbf{Q}_L)^T \cdot \mathbf{Z}_L$ is omitted here.*

Another interesting property emerges with introduction of feature maps: linear attention computation can be expressed recurrently. Note that we have

$$
\begin{aligned}
u_i &= u_{i-1} + \phi ( \mathbf{K}_i ) \mathbf{V}_{i}^T, \\ \mathbf{Z}_i &= \mathbf{Z}_{i-1} + \phi (\mathbf{K}_i),
\end{aligned}
$$

assuming $u_0, \mathbf{Z}_0$ are both 0-valued. 

This allows us to keep only constant-sized hidden states $u$ and $\mathbf{Z}$ to compute the attention during auto-regressive decoding and we don't need to feed linearly increasing inputs to the model.

#### The Hedgehog & the Porcupine

The choice of the feature map $\phi$, such that

$$\phi(q) \phi(k)^T \approx \exp(qk^T)$$

and is a non-trivial task. Although linear attention reduces computational complexity, it may lead to a decrease in model performance if kernel approximation doesn't capture key properties of full attention.

Authors of the original linear attention used $\phi(x) = \operatorname{ELU}(x)+1$ as a feature map in their experiments. Another option is to use standard $\operatorname{ReLU}$ function (though it'll set the gradients to $0$ for negative inputs). But while such choices lead to simple and effective computations, [Zhang et al. (2024)](https://arxiv.org/pdf/2402.04347) showed that, unlike softmax attention, these feature maps lead to the loss of two key features associated with higher performance:

* **Low-entropy "spikyness"**: intuitively, attentions are expected to attend only to relevant tokens and ignore irrelevant ones.

* **Dot-product monotonicity**: attention weights increase as the dot products of their corresponding queries and keys increase. Otherwise, the lack of this monotonicity can produce unstable gradients during training.

They observed that 2nd-degree Taylor approximation of exponential function $\phi_{\text{taylor}}(\mathbf{x}) = \big[1, x_1, \dots, x_d\big] \cup \big[x_i \cdot x_j \vert i, j \in [d] \big]$ for $d$-dimensional vector $\mathbf{x}$ retains both the spikiness and monotonic properties and this corresponds to (near)-matching softmax attention performance. Unfortunately, $\phi_{\text{taylor}}(\mathbf{x})$ maps to $\mathbb{R}^{1 + d + d^2}$ space, resulting in $\mathcal{O}(L d^3)$ attention complexity, which becomes costly with growing embedding dimension.

To solve this problem they propose **Hedgehog**[^HP], learnable linear layer with exponential activation function, trained to capture these properties and mimic softmax attention weights:

$$\phi_\text{mlp}(\mathbf{x}) = \exp (\mathbf{xW}).$$

To learn a softmax approximation, they train $\phi_\text{mlp}(\mathbf{x})$ to minimize the cross-entropy loss between the computed linear attention weights and those that would have been computed via softmax masked attention $\mathbf{P}$:

$$\mathcal{L}_i = -\sum_{j \leq i} \mathbf{P}_{ij} \cdot \log \frac{\phi_\text{mlp}(\mathbf{Q}_i)^T\phi_\text{mlp}(\mathbf{K}_j)}{\sum_{k \leq i} \phi_\text{mlp}(\mathbf{Q}_i)^T\phi_\text{mlp}(\mathbf{K}_k)}.$$

## Low-level hardware optimizations

### CUDA kernel fusion

We've observed before that bandwidth cost (moving data from one place in memory to another) is usually the most substantial factor when we talk about performance. Let's step back a bit and take another detailed look at the GPU hardware to understand why this is the case.

GPU is designed to perform thousands of simple operations in parallel, called **threads**. Each thread has its own (the fastest on GPU) memory, called **register**. Threads which run on the same SM can be grouped into **thread blocks** to be executed at the same time (the number of maximum threads in a block is limited by the architecture, usually it's 1024). Threads within a thread block can load data from global memory (HBM) into shared memory (SRAM), which is used for communication between threads, perform computations, and write results back to global memory.

When SM is given one or more thread blocks to execute, it partitions them into **warps**. A warp is a set of 32 threads, such that all the threads in a warp execute the same instruction. Finally, multiple thread blocks are combined to form a **grid**. All the blocks in the same grid contain the same number of threads. Since the number of threads in a block is limited, grids can be used for computations that require a large number of thread blocks to operate in parallel.

The computations are defined in **kernels**, small C++ functions, which supposed to be executed multiple times in parallel by different threads (as opposed to only once like regular C++ functions). A kernel is launched as a grid and different kernels can have different grid and block configurations.

Let's take A100 as an example again. It has 108 SMs, each can run up to 2048 threads. Shared memory capacity for each SM is up to 192KB. This means that we can take large matrices (up to few MBs), split them up in smaller chunks that fit into A100 registers and SRAM and then do matrix multiplication at the speed of 18 TB/s, SRAM bandwidth. This in total makes GPU much more efficient than CPU for matrix multiplications.

But we have a lot of non-matmul operations in deep learning, such as normalization layers, activation functions or dropouts. Even though they only account for a small fraction of the total FLOPs, GPUs run them much slower. Firstly, because they have specialized units for matrix multiply, called Tensor Cores. That's why matmul throughput can be up to 16× higher than non-matmul throughput, e.g. 312 TFLOPs vs 19.5 TFLOPs on A100.[^TC] And secondly, they can be extremely memory-bound. 

Let's take $\mathbf{P}=\operatorname{softmax}(\mathbf{S}) \in \mathbb{R}^{L \times L}$ from attention. For a large sequence length $L$, tensor $\mathbf{S}$ has to reside on HBM, since it would be too large to fit on SRAM. The simplest implementation of softmax requires us to

- Read $L^2$ floats of $\mathbf{S}$, compute $\exp(\mathbf{S})$, write $L^2$ floats back to HBM.
- Get sum of $\exp(\mathbf{S})$ along row axis: read $L^2$, write $L$.
- Divide $\exp(\mathbf{S})$ by denominator: read $L^2+L$, write $L^2$.

In total we need to move back and forth $5L^2+L$ floats. It would be much faster if we could just read $L^2$ floats of $\mathbf{S}$ and write $L^2$ floats of $\mathbf{P}$, making this operation more than 2.5× faster.

That's where **kernel fusion** comes into play: instead of writing computation output $y=f(x)$ to low-bandwidth global memory only to read it again to get $z=g(y)$, we can implement kernel which performs multiple computations at once $z=(g \circ f)(x)$ without extra memory accesses. XLA compiler in Jax can perform simple fusions, but a programmer can also write custom CUDA kernels with [Triton](https://triton-lang.org/main/index.html) or [Pallas](https://jax.readthedocs.io/en/latest/pallas/design.html).

### Memory-efficient attention

#### Online softmax

Function $\mathbf{y} = \operatorname{softmax}(\mathbf{x})$ is defined as

$$\mathbf{y}_i = \frac{e^{\mathbf{x}_i}}{\sum_j e^{\mathbf{x}_j}}.$$

```python
@jit
def naive_softmax(logits):
    exp_logits = jnp.exp(logits)
    return exp_logits / exp_logits.sum()
```

The naive implementation of softmax scans $\mathbf{x}$ 2 times - one to calculate normalization term and another to compute output vector $\mathbf{y}$. Unfortunately, on real hardware, such implementation has a serious flaw: for $\mathbf{x}_i \geq 89$ exponentiation results in infinity for bf16 and fp32. And here's a trick to avoid overflow: notice that for any constant $m$:

$$
\begin{aligned}
\operatorname{softmax}(\mathbf{x})_i & = \frac{e^{\mathbf{x}_i}}{\sum_j e^{\mathbf{x}_j}} \\ & = \frac{e^{\mathbf{x}_i}}{\sum_j e^{\mathbf{x}_j}} \cdot \frac{e^{-m}}{e^{-m}}\\ & = \frac{e^{\mathbf{x}_i-m}}{\sum_j e^{\mathbf{x}_j-m}} \\ &= \operatorname{softmax}(\mathbf{x}-m)_i .
\end{aligned}
$$

If we set $m(\mathbf{x}) = \max_i \mathbf{x}_i$ and $\ell(\mathbf{x}) = \sum_j e^{\mathbf{x}_j - m(\mathbf{x})}$ to compute

$$\mathbf{y}_i = \operatorname{softmax}(\mathbf{x}-m(\mathbf{x}))_i = \frac{e^{\mathbf{x}_i - m(\mathbf{x})}}{\ell(\mathbf{x})}$$

we implement a numerically stable version of softmax, which is sometimes called **safe softmax**. 

```python
@jit
def safe_softmax(logits):
    exp_logits = jnp.exp(logits - logits.max())
    return exp_logits / exp_logits.sum()
```

But stability comes with a price in a efficiency since we do one more pass over $\mathbf{x}$ now to calculate $m(\mathbf{x})$. This results in 4 memory accesses per vector element overall (3 loads and 1 store) and we want to improve on that.

For two vectors $\mathbf{x}^1, \mathbf{x}^2$ we can decompose statistics of concatenated vector $\mathbf{x}=[\mathbf{x}^1, \mathbf{x}^2]$ as

- $m(\mathbf{x}) = \max \big(m(\mathbf{x}^1), m(\mathbf{x}^2)\big)$
- $\ell(\mathbf{x}) = e^{m(\mathbf{x}^1) - m(\mathbf{x})} \ell(\mathbf{x}^1) + e^{m(\mathbf{x}^2) - m(\mathbf{x})} \ell(\mathbf{x}^2)$

Based on this property [Milakov and Gimelshein (2018)](https://arxiv.org/pdf/1805.02867) presented **online softmax**, which calculates both $m(\mathbf{x})$ and $\ell(\mathbf{x})$ in one pass: initialize $m_0=-\infty$ and $\ell_0 = 0$, then for each iteration $i = 1, \dots, L$:

- $m_{i} \leftarrow \max(m_{i-1}, \mathbf{x}_i),$
- $\ell_{i} \leftarrow \ell_{i-1} e^{m_{i-1}-m_i} + e^{\mathbf{x}_i - m_i}.$

This algorithm keeps the maximum value 

$$m_{i} = m([\mathbf{x}_{1}, \dots, \mathbf{x}_{i}])$$

and the normalization term 

$$\ell_{i}=\ell([\mathbf{x}_{1}, \dots, \mathbf{x}_{i}])$$

as it iterates over elements of the input array. At each iteration it needs to adjust the normalizer to the new maximum $m_{i}$ and only then add new value to $\ell_{i}$.

There also exists a parallel version to fully utilize devices:

$$
\begin{bmatrix}
  m(\mathbf{x}) \\
  \ell(\mathbf{x})
\end{bmatrix} = \begin{bmatrix}
  \mathbf{x}_1 \\
  1
\end{bmatrix} \oplus  \begin{bmatrix}
  \mathbf{x}_2 \\
  1
\end{bmatrix} \oplus \cdots \oplus \begin{bmatrix}
  \mathbf{x}_L \\
  1
\end{bmatrix},
$$

where the binary operation $\oplus: \mathbb{R}^2 \times \mathbb{R}^2 \rightarrow \mathbb{R}^2$ is defined as

$$
\begin{bmatrix}
  m_i \\
  \ell_i
\end{bmatrix} \oplus  \begin{bmatrix}
  m_j \\
  \ell_j
\end{bmatrix} = \begin{bmatrix}
  \max(m_i, m_j) \\
  \ell_i e^{m_i - \max(m_i, m_j)} + \ell_j e^{m_j - \max(m_i, m_j)}
\end{bmatrix}.
$$

The operation $\oplus$ is associative and commutative, which enables parallel and efficient evaluation. 

```python
@jit
def online_softmax(logits):
    
    def reducer(x, y):
        m_i, l_i = x
        m_j, l_j = y
        m = jnp.maximum(m_i, m_j)
        l = l_i * jnp.exp(m_i - m) + l_j * jnp.exp(m_j - m)
        return (m, l)
    
    m, l = jax.lax.reduce(
        (logits, jnp.ones_like(logits)), 
        (-jnp.inf, 0.), 
        reducer, 
        (0,)
    )
    exp_logits = jnp.exp(logits - m)
    return exp_logits / l
```

One can run this little test script to evaluate the efficiency of each implementation:

```python
# create large random vector
logits = jax.random.uniform(random.PRNGKey(42), shape=(1_000_000,))

# one warmup run for each function to compile
naive_softmax(logits)
safe_softmax(logits)
online_softmax(logits)

print('Naive:')
%timeit naive_softmax(logits).block_until_ready()
print('\nSafe:')
%timeit safe_softmax(logits).block_until_ready()
print('\nOnline:')
%timeit online_softmax(logits).block_until_ready()
```

This is the output of the script running on TPU-v3:

```
Naive:
194 μs ± 15.4 μs per loop (mean ± std. dev. of 7 runs, 10,000 loops each)

Safe:
254 μs ± 17.8 μs per loop (mean ± std. dev. of 7 runs, 1,000 loops each)

Online:
199 μs ± 22.3 μs per loop (mean ± std. dev. of 7 runs, 1,000 loops each)
```

Check out the original paper for more details, including algorithm with softmax + top-k fusion.

#### Lazy softmax

Now let's get back to calculation of attention operation. Given query, key and value tensors $\mathbf{Q}, \mathbf{K}, \mathbf{V} \in \mathbb{R}^{L \times d}$ our aim is to compute (we omit masking and $\frac{1}{\sqrt{d}}$ normalization here and after for simplification)

$$\mathbf{S} = \mathbf{Q}\mathbf{K}^T, \quad \mathbf{P} = \operatorname{softmax}({\mathbf{S}}), \quad \mathbf{O} = \mathbf{PV}.$$

The straight-forward implementation would be

$$\mathbf{S}_{ij} = \mathbf{Q}_i^T\mathbf{K}_j, \quad \mathbf{P}_{ij}= \frac{e^{\mathbf{S}_{ij}}}{\sum_{l=1}^L e^{\mathbf{S}_{il}}},\quad \mathbf{O}_{i}= \sum_{l=1}^L \mathbf{P}_{il} \mathbf{V}_l  \quad \forall i, j = 1, \dots, L$$

The problem with implementation above is that it requires us to first compute and remember $\mathbf{S}_{ij}$ for all $j$, leading to linear time and memory complexity for each query, leading to the overall time and space complexity $\mathcal{O}(L^2)$. [Rabe and Staats (2022)](https://arxiv.org/pdf/2112.05682.pdf) suggested to move the division by normalization term to the very end of the attention operation using the distributive law:

$$\mathbf{O}_{i}= \frac{ \sum_{l=1}^L \mathbf{V}_l e^{\mathbf{S}_{il}} } {\sum_{l=1}^L e^{\mathbf{S}_{il}}} \quad \forall i = 1, \dots, L.$$

This implementation, called **lazy softmax**, can be computed with constant memory for each query: we start from vector $\mathbf{v}_0 \in \mathbb{R}^d$ and scalar $\ell_0$, both initialized with $0$, and when we process key/value pairs sequentially for $j=1, \dots, L$, we only update 

$$
\begin{aligned}
\mathbf{v}_j &\leftarrow \mathbf{v}_{j-1} + \mathbf{V}_j e^{\mathbf{S}_{ij}}, \\ \ell_j &\leftarrow \ell_{j-1} + e^{\mathbf{S}_{ij}}.
\end{aligned}
$$

After processing all keys and values, we divide $\frac{\mathbf{v}_L}{\ell_L}$ to get the final result.

One can notice that such algorithm has the same numerical problem as the naive implementation of softmax: incremental computation of the sum of exponentiated scores (and values). The standard safe-softmax trick cannot be applied here as the maximum may depend on the last score in the sequence. The subtraction cannot be delayed either, since the scores must be exponentiated before they can be added to the cumulative sum.

To resolve this problem, authors introduce an additional scalar $m$ as in online softmax, which keeps track of the maximum score that the incremental algorithm has seen so far, and they renormalize the sums of exponentiated values as needed: 

$$
\begin{aligned}
m_j &\leftarrow \max(m, \mathbf{S}_{ij}), \\ 
\mathbf{v}_j &\leftarrow \mathbf{v}_{j-1} e^{m_{j-1}-m_j} + \mathbf{V}_j e^{\mathbf{S}_{ij} - m_j}, \\
\ell_j &\leftarrow \ell_{j-1} + e^{m_{j-1}-m_j}.
\end{aligned}
$$

Authors also exploited massive parallelism and provided [code in Jax](https://github.com/google-research/google-research/tree/master/memory_efficient_attention) for memory-efficient parallel algorithm. Notice that they use `jax.checkpoint` decorator in `summarize_chunk` function. The reason is that during forward pass this algorihtm saves memory by summarizing parts of the attention matrix sequentially, allowing it to forget the parts of the attention matrix it has summarized already. A naive application of differentiation would have to store all those intermediate results and algorithm would lose its memory advantage entirely. So authors propose to apply gradient checkpointing to the function that summarizes the individual chunks. The intermediate results can thus be forgotten during the forward pass and recomputed during backpropagation.

Applying checkpointing to the standard attention algorithm would not achieve these results. The standard attention algorithm with checkpointing would forget the attention matrix after it is formed; query chunk attention algorithm never forms the full attention matrix at all.

### FlashAttention

FlashAttention might be the most popular implementation of attention mechanism nowadays. While it actually does more FLOPs than standard attention, it runs up to 3× faster just by making attention algorithm IO-aware — accounting for reads and writes between levels of GPU memory. Remember, how we discussed GPU architecture in the first section of this post and that moving tensors from SRAM can be 10× faster on modern GPU than moving them from HBM.

Standard attention forward pass looks like that:

- Load $\mathbf{Q}$, $\mathbf{K}$ by blocks from HBM, compute $\mathbf{S}=\mathbf{QK^T}$, write $\mathbf{S}$ to HBM.
- Read $\mathbf{S}$ from HBM, compute $\mathbf{P} = \operatorname{softmax}(\mathbf{S})$, write $\mathbf{P}$ to HBM.
- Load $\mathbf{P}$ and $\mathbf{V}$ by blocks from HBM, compute $\mathbf{O} = \mathbf{PV}$, write $\mathbf{O}$ to HBM.
- Return $\mathbf{O}$

[Dao et al. (2022)](https://arxiv.org/pdf/2205.14135) modified it with two techniques to reduce the amount of HBM accesses to sub-quadratic in $L$:

1. **Tiling**: inputs $\mathbf{Q}$, $\mathbf{K}$, $\mathbf{V}$ are divided into blocks to fit to SRAM, then online softmax is computed to get attention scores for each block and the results are combined all together. With tiling the whole attention mechanism can be implemented in one CUDA kernel:
	
	- Load inputs from HBM
	- Perform all attention computation steps: $\mathbf{QK}^T$ matrix multiply, softmax, optionally masking and dropout, $\mathbf{PV}$ matrix multiply
	- And only then write the result back to HBM

2. **Recomputation**:

	Training requires to store intermediate outputs, such as $\mathbf{S}, \mathbf{P} \in \mathbb{R}^{L \times L}$, to compute gradients with respect to $\mathbf{Q}$, $\mathbf{K}$, $\mathbf{V}$ during backward pass. The standard mechanism to reduce memory footprint during training is to use gradient checkpointing: forgetting these outputs in forward pass and recalculating them in backward pass. However, this implementation has to trade speed for memory. 
	
	Authors propose to use selective gradient checkpointing and to store only the output $\mathbf{O}$ and the softmax normalization statistics $(m, \ell)$ to recompute the attention matrices $\mathbf{S}$ and $\mathbf{P}$ easily in the backward pass from blocks of $\mathbf{Q}$, $\mathbf{K}$, $\mathbf{V}$ in SRAM.

**FlashAttention** forward pass:

- Set block sizes $B_{\mathbf{KV}} = \lceil \frac{M}{4d} \rceil$, $B_{\mathbf{Q}} = \min \big( \lceil \frac{M}{4d} \rceil, d \big)$, where $M$ is an on-chip SRAM size.
- Initialize $\color{#E86456}{\mathbf{O} \in \mathbb{R}^{L \times d}}$, $\color{#E86456}{\ell \in \mathbb{R}^{L}}$ both $0$-valued and $\color{#E86456}{m \in \mathbb{R}^L}$ with values set to $\mathbf{-\infty}$, all stored in HBM.
- Split $\color{#E86456}{\mathbf{Q}}$ into $T_{\mathbf{Q}} = \lceil \frac{L}{B_{\mathbf{Q}}} \rceil$ blocks and split $\color{#E86456}{\mathbf{K}, \mathbf{V}}$ into $T_{\mathbf{KV}} = \lceil \frac{L}{B_{\mathbf{KV}}} \rceil$ blocks along sequence axis.
- Split $\color{#E86456}{\mathbf{O}, \ell, m}$ into $T_{\mathbf{Q}}$ blocks along sequence axis.
- For $j = 1, \dots, T_{\mathbf{KV}}$:
	- Load $\color{#65AD69}{\mathbf{K}_j, \mathbf{V}_j}$ from HBM to SRAM
	- For $i = 1, \dots, T_{\mathbf{Q}}$:
		- Load $\color{#65AD69}{\mathbf{Q}_i, \mathbf{O}_i, \ell_i, m_i}$ from HBM to SRAM.
		- Compute unnormalized attention scores 
		
		  $$\color{#65AD69}{\mathbf{S}_{ij} = \mathbf{Q}_i \mathbf{K}^T_j \in \mathbb{R}^{B_{\mathbf{Q}} \times B_{\mathbf{KV}}}}.$$
		  
		- Compute
		
		 $$
		 \color{#65AD69}{
		 \begin{aligned}
		 \tilde{m}_{ij} & = \operatorname{rowmax}(\mathbf{S}_{ij}) \in \mathbb{R}^{B_{\mathbf{Q}}}, \\
		 \tilde{\mathbf{P}}_{ij} & = \exp ( \mathbf{S}_{ij} - \tilde{m}_{ij}) \in \mathbb{R}^{B_{\mathbf{Q}} \times B_{\mathbf{KV}}}, \\
		 \tilde{\ell}_{ij} & = \operatorname{rowsum}(\tilde{\mathbf{P}}_{ij}) \in \mathbb{R}^{B_{\mathbf{Q}}}
		 \end{aligned}}
		 $$
		 
		- Renew statistics
		
		 $$
		 \color{#65AD69}{
		 \begin{aligned}
		 m_i^{\text{new}} & = \max(m_i, \tilde{m}_{ij}) \in \mathbb{R}^{B_{\mathbf{Q}}}, \\
		 \ell_{i}^{\text{new}} & = e^{m_i-m_i^{\text{new}}} \ell_i + e^{\tilde{m}_{ij} - m_i^{\text{new}}}\tilde{\ell}_{ij}  \in \mathbb{R}^{B_{\mathbf{Q}}}
		 \end{aligned}}
		 $$
		 
		- Write 
		$$
		\color{#E86456}{\mathbf{O}_i} \color{#EDA137}{ \leftarrow } \color{#65AD69}{\operatorname{diag}(\ell_i^{\text{new}})^{-1} \big( \operatorname{diag}(\ell_i) e^{m_i-m_i^{\text{new}}} \mathbf{O}_i + e^{\tilde{m}_{ij} - m_i^{\text{new}}} \tilde{\mathbf{P}}_{ij} \mathbf{V}_j \big) }
		$$
		to HBM.
		- Write $\color{#E86456}{\ell_i} \color{#EDA137}{ \leftarrow } \color{#65AD69}{ \ell_{i}^{\text{new}}}$, $\color{#E86456}{m_i} \color{#EDA137}{ \leftarrow } \color{#65AD69}{m_{i}^{\text{new}}}$ to HBM.
- Return $\color{#E86456}{\mathbf{O}}$.

In the end FlashAttention algorithm returns $\mathbf{O} = \operatorname{softmax}(\mathbf{QK}^T)\mathbf{V}$ using $\mathcal{O}(L^2 d)$ FLOPs and $\mathcal{O}(L)$ additional memory beyond inputs and outputs. In terms of memory accesses, it requires $\mathcal{O}(L^2d^2M^{-1})$ HBM accesses, where $d \leq M \leq Ld$, compared to $\mathcal{O}(Ld + L^2)$ for standard dot-product attention.

<div id="flash_attn" class="svg-container" align="center"></div> 

<script>

function text_id_(svg, text, x, y, opacity=1, size=14, text_id="fattn") {
	svg.append('text')
	  .attr("id", text_id)
	  .attr('x', x)
	  .attr('y', y)
	  .text(text)
	  .style("font-size", size + "px")
	  .attr('opacity', opacity)
	  .style("fill", "currentColor")
	  .attr("font-family", "Arvo");
}

function draw_fa_stage1_text(svg, x_start, y_start, rct_sz, shift, opacity) {
	text_id_(svg, "/", x_start + 25.2 * rct_sz, y_start + 3.3 * shift, opacity);
	text_id_(svg, "ℓ", x_start + 25.6 * rct_sz, y_start + 3.3 * shift, opacity);	
	text_id_(svg, "1", x_start + 25.6 * rct_sz + 7, y_start + 3.3 * shift + 3, opacity, size=8);
	
	text_id_(svg, "K", x_start + 4 * rct_sz, y_start - shift, opacity);
	text_id_(svg, "1", x_start + 4 * rct_sz + 11, y_start - shift + 3, opacity, size=8);
	text_id_(svg, "T", x_start + 4 * rct_sz + 10, y_start - shift - 8, opacity, size=8);
	
	text_id_(svg, "S", x_start + 4 * rct_sz, y_start + 7.5 * shift, opacity);
	text_id_(svg, "11", x_start + 4 * rct_sz + 9, y_start + 7.5 * shift + 3, opacity, size=8);
	
	text_id_(svg, "V", x_start + 19 * shift + 7, y_start + shift, opacity);
	text_id_(svg, "1", x_start + 19 * shift + 18, y_start + shift + 3, opacity, size=8);
	
	text_id_(svg, "P̃", x_start + 13.2 * rct_sz, y_start + 7.5 * shift, opacity);	
	text_id_(svg, "11", x_start + 13.2 * rct_sz + 9, y_start + 7.5 * shift + 3, opacity, size=8);
}

function draw_fa_stage2_text(svg, x_start, y_start, rct_sz, shift, opacity) {
	text_id_(svg, "/", x_start + 25.2 * rct_sz, y_start + 6.3 * shift, opacity);
	text_id_(svg, "ℓ", x_start + 25.6 * rct_sz, y_start + 6.3 * shift, opacity);	
	text_id_(svg, "2", x_start + 25.6 * rct_sz + 7, y_start + 6.3 * shift + 3, opacity, size=8);
	
	text_id_(svg, "⋅", x_start + 31.7 * rct_sz, y_start + 4.8 * shift, opacity);
	text_id_(svg, "ℓ", x_start + 32.5 * rct_sz, y_start + 4 * shift, opacity);	
	text_id_(svg, "1", x_start + 32.5 * rct_sz + 7, y_start + 4 * shift + 3, opacity, size=8);
	text_id_(svg, "ℓ", x_start + 32.5 * rct_sz, y_start + 5.7 * shift, opacity);	
	text_id_(svg, "2", x_start + 32.5 * rct_sz + 7, y_start + 5.7 * shift + 3, opacity, size=8);
	text_id_(svg, "_", x_start + 32.5 * rct_sz, y_start + 4.4 * shift, opacity, size=18);
	text_id_(svg, "+", x_start + 33.7 * rct_sz, y_start + 4.8 * shift, opacity);
	
	text_id_(svg, "K", x_start + 7.5 * rct_sz, y_start - shift, opacity);
	text_id_(svg, "2", x_start + 7.5 * rct_sz + 11, y_start - shift + 3, opacity, size=8);
	text_id_(svg, "T", x_start + 7.5 * rct_sz + 10, y_start - shift - 8, opacity, size=8);
	
	text_id_(svg, "S", x_start + 7.5 * rct_sz, y_start + 7.5 * shift, opacity);
	text_id_(svg, "12", x_start + 7.5 * rct_sz + 9, y_start + 7.5 * shift + 3, opacity, size=8);
	
	text_id_(svg, "P̃", x_start + 16.7 * rct_sz, y_start + 7.5 * shift, opacity);	
	text_id_(svg, "12", x_start + 16.7 * rct_sz + 9, y_start + 7.5 * shift + 3, opacity, size=8);
	
	text_id_(svg, "V", x_start + 19 * shift + 7, y_start + 8.6 * shift, opacity);
	text_id_(svg, "2", x_start + 19 * shift + 18, y_start + 8.6 * shift + 3, opacity, size=8);
}

function draw_flash_attn_text(svg, x_start, y_start, rct_sz, shift) {
	text_(svg, "SRAM", x_start + 363, y_start - 32, size=11);
	text_(svg, "HBM", x_start + 416, y_start - 32, size=11);

	text_(svg, "Q", x_start + 7, y_start + 2 * shift);
	text_(svg, "1", x_start + 18, y_start + 2 * shift + 3, size=8);
	
	text_(svg, "O", x_start + 27.5 * shift + 8, y_start + 2 * shift);
	text_(svg, "1", x_start + 27.5 * shift + 19, y_start + 2 * shift + 3, size=8);
	
	text_(svg, "→", x_start + 10.5 * rct_sz, y_start + 4.8 * shift);
	
	text_(svg, "⋅", x_start + 19.8 * rct_sz, y_start + 4.9 * shift + 3, size=30);
	text_(svg, "=", x_start + 27 * rct_sz, y_start + 4.8 * shift, size=18);
}

function flash_attn() {
	var svg = d3.select("#flash_attn")
				  .append("svg")
				  .attr("width", 600)
				  .attr("height", 205);
	const x_start = 90, y_start = 50;
	const shift = 14, rct_sz = 12;
	const dim = 2;
	const b_r = 3, b_c = 2;
	const t_r = 1, t_c = 2;
	const shade = 0.08;
	
	function draw_flash_attn_cells(stage) {
	   svg.selectAll('rect').remove();

	   rect(svg, x_start + 410, y_start - 45, 40, 20, colors[2], opacity=0.3);
	   rect(svg, x_start + 360, y_start - 45, 40, 20, colors[0], opacity=0.3);
	   
	   rect(svg, x_start - 5, y_start + 3 * shift - 5, dim * shift + 8, t_r * (b_r + 1) * shift - 5, colors[2], opacity=0.3);
	   
	   for (var i = 0; i < t_r; i += 1) {
			vector_rect(svg, x_start, y_start + (3 + i * (b_r + 1)) * shift, dim, b_r, shift, rct_sz, matrix_colors[0]);
		}
		
	   rect(svg, x_start + 3 * shift - 5, y_start - 5, t_c * (b_c + 1) * shift - 5, dim * shift + 8, colors[2], opacity=0.3);
	   
		for (var i = 0; i < t_c; i += 1) {
		   var opacity = (i == stage - 1) ? 1 : shade;
			vector_rect(svg, x_start + (3 + i * (b_c + 1)) * shift, y_start, b_c, dim, shift, rct_sz, matrix_colors[1], opacity);
		}
		
	   rect(svg, x_start + 3 * shift - 5, y_start + 3 * shift - 5, t_c * (b_c + 1) * shift - 5, t_r * (b_r + 1) * shift - 5, colors[0], opacity=0.3);
	   
	   for (var i = 0; i < t_c; i += 1) {
		   var opacity = (i == stage - 1) ? 1 : shade;
			for (var j = 0; j < t_r; j += 1) {
				vector_rect(svg, x_start + (3 + i * (b_c + 1)) * shift, y_start + (3 + j * (b_r + 1)) * shift, b_c, b_r, shift, rct_sz, matrix_colors[2], opacity);
			}
		}
		
	   rect(svg, x_start + 11 * shift - 5, y_start + 3 * shift - 5, t_c * (b_c + 1) * shift - 5, t_r * (b_r + 1) * shift - 5, colors[0], opacity=0.3);
	   
	   for (var i = 0; i < t_c; i += 1) {
		   var opacity = (i == stage - 1) ? 1 : shade;
		   for (var j = 0; j < t_r; j += 1) {
				vector_rect(svg, x_start + (11 + i * (b_c + 1)) * shift, y_start + (3 + j * (b_r + 1)) * shift, b_c, b_r, shift, rct_sz, matrix_colors[2], opacity);
			}
		}
	   
	   rect(svg, x_start + 19 * shift - 5, y_start + 2 * shift - 5, dim * shift + 8, t_c * (b_c + 1) * shift - 5, colors[2], opacity=0.3);
		
		for (var i = 0; i < t_c; i += 1) {
		   var opacity = (i == stage - 1) ? 1 : shade;
			vector_rect(svg, x_start + 19 * shift, y_start + (2 + i * (b_c + 1)) * shift, b_c, dim, shift, rct_sz, matrix_colors[3], opacity);
		}
	  	   
	   rect(svg, x_start + 25 * shift - 5, y_start + 3 * shift - 5, t_c * (b_c + 2) * shift - 5, t_r * (b_r + 1) * shift - 5, colors[2], opacity=0.3);
	   
	   for (var i = 0; i < t_c; i += 1) {
		   var opacity = (i <= stage - 1) ? 1 : shade;
		   for (var j = 0; j < t_r; j += 1) {
				vector_rect(svg, x_start + (25 + i * (b_c + 3)) * shift, y_start + (3 + j * (b_r + 1)) * shift, dim, b_r, shift, rct_sz, matrix_colors[4], opacity);
			}
		}
		
		d3.selectAll("#fattn").remove();
		
		var opacity = (stage == 1) ? 1 : shade;
		draw_fa_stage1_text(svg, x_start, y_start, rct_sz, shift, opacity);
		
		opacity = (stage == 2) ? 1 : shade;
		draw_fa_stage2_text(svg, x_start, y_start, rct_sz, shift, opacity);
	}

	var l_x = d3.scaleLinear()
	    .domain([1, 2])
	    .range([0, 320])
	    .clamp(true);
	    
	createSlider(svg, draw_flash_attn_cells, l_x, x_start + 50, 185, "i", "currentColor", 1, roundN, 1);
	
	draw_flash_attn_cells(1);
	draw_flash_attn_text(svg, x_start, y_start, rct_sz, shift);
}

flash_attn();
</script>

![](.)
*Schematic diagram of how FlashAttention forward pass is performed, when $\mathbf{Q}$ is partitioned into $T_{\mathbf{Q}}=1$ block of size $B_{\mathbf{Q}} \times d$ with $B_{\mathbf{Q}}=3$ and $\mathbf{K}/\mathbf{V}$ are partitioned into $T_{\mathbf{KV}} = 2$ blocks of size $B_{\mathbf{KV}} \times d$ with $B_{\mathbf{KV}}=2$ each. Here $\ell_1=\sum e^{\mathbf{S}_{11}}$, $\ell_2=\ell_1 + \sum e^{\mathbf{S}_{12}}$. The step with subtracting $m$ in softmax is omitted for simplification.*

Authors of FlashAttention also compared it to query chunk attention algorithm, stating three major differences:

1. Memory-efficient attention focuses on memory footprint, while FlashAttention focuses on reducing HBM accesses.
2. Memory-efficient attention summarizes each block with its temporary output along with the softmax normalization statistics. FlashAttention instead incrementally updates the output after processing each block, so only one copy of the output is needed.
3. Memory-efficient attention uses gradient checkpointing to recompute the attention matrix and the temporary output of each block. FlashAttention
only recomputes the attention matrix and does not recompute the temporary output of each block. 

"FlashAttention" is an amazingly written paper and it's definitely worth reading to anyone who's training large language models. There are more details in the paper about FlashAttention backward pass, theoretical proofs and comparisons to other optimization algorithms.

#### FlashAttention + Parallelism

FlashAttention significantly speeds up attention computation and reduces memory usage from quadratic to linear in sequence length. While it works for most cases, it's not optimized for the case of long sequences with small batch size and/or small number of attention heads, due to insufficient parallelism.

The first version of FlashAttention kernel uses one thread block per one attention head leading to overall $Bh$ thread blocks ($B$ is batch size, $h$ is number of attention heads). Each thread block is scheduled to run on a SM, and such scheduling is only efficient when $Bh$ is as large as number of SMs (e.g. 108 SMs for A100 GPU) for all of the compute resources to be used effectively. If one trains LLM with [modern parallelism techniques](https://astralord.github.io/posts/exploring-parallel-strategies-with-jax/) batch size is reduced by a factor of DP and number of heads is reduced by a factor of TP.

Tri Dao, author of the original FlashAttention, [applied](https://hazyresearch.stanford.edu/blog/2023-01-12-flashattention-long-sequences) additional parallelization along sequence axis to make better use of the multiprocessors on the GPU. In such regime in forward pass one attention head is now processed by multiple thread blocks and each block takes care of its own segment of rows of the attention matrix. As the rows of the attention matrix don’t depend on each other, we don’t need to communicate between the blocks.

In the backward pass each thread block now takes care of a segment of columns of the attention matrix. Parallelizing by columns is faster than parallelizing by rows due to the reduced communication between the workers (parallelizing by columns requires aggregating the gradient of the query, while parallelizing by rows requires aggregating the gradient of the key and value).

#### FlashAttention-2

A new version of FlashAttention ([Tri Dao, 2023](https://arxiv.org/pdf/2307.08691)) included parallelization across different thread blocks to increase occupancy for long sequences. Besides that two numbers were reduced:

- the number of non-matmul FLOPs,
- the number of communication through SRAM.

The first tweak is to rewrite the online softmax trick to reduce the number of rescaling operations as well as bound-checking and causal masking, without changing the output (remember that matmul throughput can be few orders higher than non-matmul throughput).

The second tweak is an optimal work partitioning between warps, a group of threads working together. In the first FlashAttention $\mathbf{K}$ and $\mathbf{V}$ were divided across 4 or 8 warps per thread block. Each warp multiplies to get a slice of $\mathbf{QK}^T$, then they need to multiply with a slice of $\mathbf{V}$ and communicate to add up the result. However, this is inefficient since all warps need to write their intermediate results out to shared memory, synchronize, then add up the intermediate results.

In FlashAttention-2, $\mathbf{Q}$ is divided instead across 4 warps while keeping $\mathbf{K}$ and $\mathbf{V}$ accessible by all warps. After each warp performs matrix multiply to get a slice of $\mathbf{QK}^T$, they just need to multiply with their shared slice of $\mathbf{V}$ to get their corresponding slice of the output. There is no need for communication between warps. The reduction in SRAM reads/writes yields speedup

The second version also introduced support for larger head dimensions ($d = 128 \rightarrow 256$) and support for MQA and GQA.

#### FlashAttention-3

The most recent version, [FlashAttention-3 (2024)](https://tridao.me/publications/flash3/flash3.pdf) focused on optimizations specific to H100 GPU, as previous versions only achieved up to 35% utilization with Hopper architecture. Authors exploited a new feature: **tensor memory accelerator (TMA)** - a new chunk of hardware that can do asynchronous address generation and fetch memory. The main techniques which were used for speed up:

1. Overlap overall computation and data movement 
2. Interleave block-wise matmul and softmax operations 
3. Block quantization and incoherent processing that leverages hardware
support for FP8 low-precision

### Ring Attention

Even with Flash Attention, the memory complexity is linear in $L$ so scaling the sequence length is limited by the memory capacity. We could scale context with number of devices $N$, split inputs into $N$ parts, perform computations in parallel, then gather the results. However, attention requires for $\mathbf{Q}$ to access all elements of $\mathbf{K}, \mathbf{V}$ matrices.[^OISWA] Sending large matrices between devices can add a huge communication overhead (e.g. A100 throughput is 600GB/s with NVLink and only 64GB/s with PCIe).

**Ring Attention** ([Lie et al. (2023)](https://arxiv.org/pdf/2310.01889)) addresses this problem and explores the idea of hiding communication overhead behind computations in an extremely large context scenario. The algorithm is the following:

- Split input sequence into $N$ blocks along sequence axis, such that each device stores one input block of size $C = \lceil \frac{L}{N} \rceil$. Compute $\mathbf{Q}_i$, $\mathbf{K}_i$, $\mathbf{V}_i$ for its input block on each $i$-th device.
- For $\text{iter} = 0, \dots, N - 1$:
	- For each $i$-th device in parallel:
		- Let $j = (\text{iter} + i) \bmod N$.
		- Compute memory-efficient attention incrementally using local $\mathbf{Q}_i$, $\mathbf{K}_j$, $\mathbf{V}_j$ blocks. 
		- *Simultaneously*, send $\mathbf{K}_j, \mathbf{V}_j$ blocks to the next device and receive new blocks from the previous device.

		

<div id="ring_attn" class="svg-container" align="center"></div> 

<script>

function text_id_2_(svg, text, x, y, size=14, text_id="fattn") {
	svg.append('text')
	  .attr("id", text_id)
	  .attr('x', x)
	  .attr('y', y)
	  .attr('orig_x', x)
	  .attr('orig_y', y)
	  .text(text)
	  .style("fill", "currentColor")
	  .style("font-size", size + "px")
	  .attr("font-family", "Arvo");
}

function draw_ring_attn_text_device(svg, x_start, y_start, device) {
	text_(svg, "Q", x_start, y_start);
	text_(svg, device, x_start + 11, y_start + 3, size=8);
	text_id_2_(svg, "K", x_start + 52, y_start - 45, size=14, text_id="kv" + (device - 1));
	text_id_2_(svg, device, x_start + 63, y_start - 42, size=8, text_id="kv" + (device - 1));
	text_id_2_(svg, "V", x_start + 104, y_start, size=14, text_id="kv" + (device - 1));
	text_id_2_(svg, device, x_start + 115, y_start + 3, size=8, text_id="kv" + (device - 1));
	text_(svg, "⋅", x_start + 74, y_start, size=18);
	text_(svg, "GPU " + device, x_start + 40, y_start + 37);
}

function draw_ring_attn_text(svg, x_start, y_start) {
	draw_ring_attn_text_device(svg, x_start + 190, y_start + 43, 1);
	draw_ring_attn_text_device(svg, x_start + 380, y_start + 143, 2);
	draw_ring_attn_text_device(svg, x_start + 190, y_start + 243, 3);
	draw_ring_attn_text_device(svg, x_start, y_start + 143, 4);
}

function labeled_rect(svg, x, y, w, h, color, label="kv") {
	svg.append('rect')
	  .attr("id", label)
	  .attr('x', x)
	  .attr('y', y)
	  .attr("orig_x", x)
	  .attr("orig_y", y)
	  .attr('width', w)
	  .attr('height', h)
	  .attr('stroke', 'black')
	  .attr('stroke-width', 1)
	  .attr("rx", Math.min(3, Math.min(w, h) / 3))
	  .attr('fill', color);
}

function labeled_vector_rect(svg, x, y, w, h, shift, rct_sz, color, label="kv") {
	for (var i = 0; i < w; i += 1) {
		for (var j = 0; j < h; j += 1) {
			labeled_rect(
				svg,
				x + i * shift, 
				y + j * shift, 
				rct_sz, 
				rct_sz, 
				color,
				label);
		}
	}
}

function get_positions(iter, x_start, y_start) {
	if (iter == 0) {
		return [x_start + 215, y_start + 25];
	}
	if (iter == 1) {
		return [x_start + 405, y_start + 125];
	}
	if (iter == 2) {
		return [x_start + 215, y_start + 225];
	}
	return [x_start + 25, y_start + 125];
}

function ring_attn() {
	var svg = d3.select("#ring_attn")
				  .append("svg")
				  .attr("width", 600)
				  .attr("height", 350);
	const x_start = 50, y_start = 25;
	const shift = 14, rct_sz = 12;
	const dim = 1;
	
	function draw_ring_attn_init_cells(x_st, y_st, label) {
		labeled_vector_rect(svg, x_st, y_st, dim, 2, shift, rct_sz, matrix_colors[0], "q");
		labeled_vector_rect(svg, x_st + 20, y_st - 20, 2, dim, shift, rct_sz, matrix_colors[1], label);
		labeled_vector_rect(svg, x_st + 20, y_st, 2, 2, shift, rct_sz, matrix_colors[2], "s");
		labeled_vector_rect(svg, x_st + 60, y_st, dim, 2, shift, rct_sz, matrix_colors[3], label);
		
		svg.append('rect')
		  .attr('x', x_st - 30)
		  .attr('y', y_st - 40)
		  .attr('width', 130)
		  .attr('height', 70)
		  .attr('stroke', 'currentColor')
		  .attr('stroke-width', 1)
		  .attr("rx", 3)
		  .attr('fill', 'none');
		  
	}
	
	var xAxis = d3.scaleLinear()
	    .domain([0, 1000])
	    .range([0, 1000])
	    .clamp(true);

	function drag_ring_attn_cells(iter) {
		for (var i = 0; i < 4; i += 1) {
			var [x_st_new, y_st_new] = get_positions((iter + i) % 4, x_start, y_start);
			var [x_st, y_st] = get_positions(i, x_start, y_start);
			var x_shift = x_st_new - x_st;
			var y_shift = y_st_new - y_st;
			
			d3.selectAll("#kv" + i).transition()
	        .delay(0)
	        .duration(1500)
	        .attr("x", function(d, i) {
	        		return x_shift + xAxis(d3.select(this).attr("orig_x"));
	         })
	        .attr("y", function(d, i) {
	        		return y_shift + xAxis(d3.select(this).attr("orig_y"));
	         });
	    }
   }

	var l_x = d3.scaleLinear()
	    .domain([0, 3])
	    .range([0, 415])
	    .clamp(true);
	    
	createSlider(svg, drag_ring_attn_cells, l_x, x_start + 50, 330, "iter", "currentColor", 0, roundN, 4);
	
	var x_st = 0, y_st = 0;
	for (var i = 0; i < 4; i += 1) {
		[x_st, y_st] = get_positions(i, x_start, y_start);
		draw_ring_attn_init_cells(x_st, y_st, "kv" + i);
		rect(svg, x_st - 30, y_st - 40, 130, 70, colors[i], 0.08);
	}
	draw_ring_attn_text(svg, x_start, y_start);
}

ring_attn();
</script>

![](.)
*GPUs are arranged in a ring, each holding a portion of $\mathbf{Q}$. During the Ring Attention process, the GPUs pass along blocks of $\mathbf{K}, \mathbf{V}$ to each other.*

Local attention compute at each iteration requires $\mathcal{O}(C^2d)$ operations, while each device needs to send $\mathbf{K}_j, \mathbf{V}_j \in \mathbb{R}^{C \times d}$ tensors or $\mathcal{O(Cd)}$ bytes. Thus the lower bound of chunk to effectively hide communication overhead will be

$$ C = \big  \lceil \frac{L}{N} \big \rceil \geq \frac{\operatorname{compute performance}}{\operatorname{communication bandwidth}}.$$

Authors also shared implementation [code in JAX](https://github.com/forhaoliu/ringattention/tree/main).

#### Stripe Attention

[Brandon et al. 2023](https://arxiv.org/pdf/2311.09431) studied Ring Attention performance for causal transformer and found out that it leads to highly imbalanced workload due to triangular structure of causal attention computations. They propose a simple extension called **Stripe Attention** to fix this and achieved near 1.5× speedup.

The problem of Ring Attention is that on all but the first iteration, the workload of some devices is entirely necessary (unmasked), while the workload of others is entirely unnecessary (masked) for the final output (and thus there is no need for them to be computed). The total latency is determined by the maximum latency of any participating device per iteration. As a result, regardless of per device optimizations, the latency per iteration would be the same as the time taken to compute a fully unmasked workload. As a result, Ring Attention will run as fast as a workload with no attention masking, despite in principle needing to compute only half the operations.
 
<div id="stripe_attn" class="svg-container" align="center"></div> 

<script>

function stripe_attn() {
	var svg = d3.select("#stripe_attn")
				  .append("svg")
				  .attr("width", 600)
				  .attr("height", 300);
	const x_start = 10, y_start = 25;
	const shift = 14, rct_sz = 12;
	
	text_(svg, "Ring Attention scores", x_start + 55, y_start - 12, 11);
	text_(svg, "Stripe Attention scores", x_start + 400, y_start - 12, 11);
	
	function ring_causal(iter, x_st, y_st) {
		causal_mask(svg, x_st, y_st, 16, 16, shift, rct_sz, matrix_colors[7], 16, 0, 1);
		
		for (var i = 0; i < 4; i += 1) {
			var k = (i + iter) % 4;
			vector_rect(svg, x_st + 4 * shift * k, y_st + 4 * shift * i, 4, 4, shift, rct_sz, colors[i], 0.5);
		}	
	}
	
	function stripe_causal(iter, x_st, y_st) {
		causal_mask(svg, x_st, y_st, 16, 16, shift, rct_sz, matrix_colors[7], 16, 0, 1);
		
		for (var j = 0; j < 4; j += 1) {
			for (var i = 0; i < 16; i += 1) {
				var k = (i + 4 * j + iter) % 16;
				rect(svg, x_st + shift * k, y_st + shift * i, rct_sz, rct_sz, colors[i % 4], 0.5);
			}
		}
	}
	
	function stripe_attn(iter) {	
	   svg.selectAll('rect').remove();
		ring_causal((4 - iter) % 4, x_start, y_start);
		stripe_causal((4 - iter) % 4, x_start + 350, y_start);
	}
	
	stripe_attn(0);
	
	var l_x = d3.scaleLinear()
	    .domain([0, 3])
	    .range([0, 415])
	    .clamp(true);
	    
	    
	createSlider(svg, stripe_attn, l_x, x_start + 85, 280, "iter", "currentColor", 0, roundN, 4);
}

stripe_attn();
</script>

![](.)
*Workload distribution of Ring Attention (left) and Stripe Attention. The square matrices represent the set of all possible pairwise query/key interactions; row indices correspond to queries, and column indices correspond to keys. All cells above the diagonal are masked out by the causal mask and can be skipped. The colors indicate which devices are responsible for which parts of the computation. We can see that some devices in Ring Attention are responsible for workloads which are entirely masked out, whereas Striped Attention maintains a balanced workload across all devices*

Rather than partitioning the tokens into contiguous blocks like in Ring Attention, Stripe Attention partitions them into sets of evenly spaced **stripes** based on their residues modulo $N$, so that $i$-th token would reside on $i \bmod N$ device. In practice, we can achieve this partitioning scheme by permuting the input tokens before the model’s first embedding layer, and then partitioning the permuted sequence into contiguous blocks as in Ring Attention. After partitioning, Stripe and Ring Attention algorithms proceeds almost identically.

Another recent improvement, called **Tree Attention** ([Shyam et al. 2024](https://arxiv.org/pdf/2408.04093)), leveraged tree-reduction topology to reduce communication costs for decoding across multiple GPUs and achieved *asymptotically* 8× speedup compared to Ring Attention.


## Serving frameworks

### PagedAttention / vLLM

KV cache takes up a significant amount of memory. In a straightforward implementation we can store the KV cache of a request in contiguous memory space like all the other tensors. However, unlike the tensors in the traditional deep learning workloads, KV cache dynamically grows and shrinks over time as the model generates new tokens, and its lifetime and length are not known a priori.

To store the KV cache of a request in contiguous space, we have to pre-allocate a contiguous chunk of memory with the request's maximum length, while the request's actual length can be much shorter. Another memory inefficiency can be observed when we use advanced decoding techniques such as beam search or parallel sampling to generate multiple outputs per request. In these scenarios, the request consists of multiple sequences that can partially share their KV cache. However, memory sharing is not possible when KV cache is stored in separate contiguous spaces.

To address the above limitations, [Kwon et al., 2023](https://arxiv.org/pdf/2309.06180) propose **PagedAttention**, an algorithm inspired by the OS solution to memory fragmentation and sharing: *virtual memory with paging*. PagedAttention divides the request's KV cache into blocks, each of which can contain the attention keys and values of a fixed number of tokens. In PagedAttention, the blocks for the KV cache are not necessarily stored in contiguous space and we can manage the KV cache in a more flexible way. 

The key idea is to represent KV cache as a series of *logical KV blocks*, filled from left to right as new tokens are generated, that can be divided into non-contiguous *physical KV blocks* in GPU memory, allocated by a *block engine*. The KV *block manager* maintains *block tables* — the mapping between logical and physical KV blocks of each request. Separating logical and physical KV blocks allows vLLM to dynamically grow the KV cache memory without reserving it for all positions in advance.

<div id="paged_attn" class="svg-container" align="center"></div> 

<script>

function block(svg, x, y, w, h, color, opacity=1.0) {
	svg.append('rect')
	  .attr('x', x)
	  .attr('y', y)
	  .attr('width', w)
	  .attr('height', h)
	  .attr('stroke', 'black')
	  .attr('stroke-width', 1)
	  .attr("rx", 0)
	  .attr('fill', color)
	  .attr('opacity', opacity);
}

function vector_block(svg, x, y, w, h, shift_w, shift_h, rct_sz_w, rct_sz_h, color, opacity=1.0) {
	for (var i = 0; i < w; i += 1) {
		for (var j = 0; j < h; j += 1) {
			block(svg,
				x + i * shift_w, 
				y + j * shift_h, 
				rct_sz_w, 
				rct_sz_h, 
				color,
				opacity);
		}
	}
}

function paged_attn() {
	var svg = d3.select("#paged_attn")
				  .append("svg")
				  .attr("width", 525)
				  .attr("height", 300);
	const x_start = 0, y_start = 25;
	const shift_w = 45, shift_h = 24;
	const rct_sz_w = 43, rct_sz_h = 22;
	
	const a_text = ["Four", "score", "and", "seven", 
	                "years", "ago", "our", "fathers",
	                "brought"];
	const b_text = ["It", "was", "the", "best", "of", "times"];
	
	function block_text() {
		text_(svg, "Logical KV blocks", x_start + 95, y_start - 15, 11);
		for (var i = 0; i < 3; i += 1) {
			text_(svg, "Block " + i, x_start, y_start + 15 + i * shift_h, 11);
		}
		text_(svg, "Request A", x_start + 115, y_start + 4 * shift_h - 10, 10);
		
		for (var i = 0; i < 3; i += 1) {
			text_(svg, "Block " + i, x_start, y_start + 15 + (i + 5) * shift_h, 11);
		}
		text_(svg, "Request B", x_start + 115, y_start + 9 * shift_h - 10, 10);
		text_(svg, "Physical KV blocks", x_start + 365, y_start - 15, 11);
		for (var i = 0; i < 9; i += 1) {
			text_(svg, "Block " + i, x_start + 275, y_start + 15 + i * shift_h, 11);
		}
	}
	
	block_text();
	
	function generate(token) {
		svg.selectAll('rect').remove();
		d3.selectAll("#vllm").remove();
		
		var x_ph = x_start + 50;
		vector_block(svg, x_ph, y_start, 4, 3, shift_w, shift_h, rct_sz_w, rct_sz_h, 'none', 0.5);	
		
		vector_block(svg, x_ph, y_start + 5 * shift_h, 4, 3, shift_w, shift_h, rct_sz_w, rct_sz_h, 'none', 0.5);	
		x_ph = x_start + 325;
		vector_block(svg, x_ph, y_start, 4, 9, shift_w, shift_h, rct_sz_w, rct_sz_h, 'none', 0.5);	
		
		if (token < 4) {
			vector_block(svg, x_ph, y_start + 7 * shift_h, token, 1, shift_w, shift_h, rct_sz_w, rct_sz_h, colors[1], 0.5);
			vector_block(svg, x_ph - 275, y_start, token, 1, shift_w, shift_h, rct_sz_w, rct_sz_h, colors[1], 0.5);
			for (var i = 0; i < token; i += 1) {
				text_id_(svg, a_text[i], x_ph - 273 + i * shift_w, y_start + 15, 1, 10, "vllm");
				text_id_(svg, a_text[i], x_ph + 2 + i * shift_w, y_start + 7 * shift_h + 15, 1, 10, "vllm");
			}
		}
		else {
			vector_block(svg, x_ph, y_start + 7 * shift_h, 4, 1, shift_w, shift_h, rct_sz_w, rct_sz_h, colors[1], 0.5);
			vector_block(svg, x_ph - 275, y_start, 4, 1, shift_w, shift_h, rct_sz_w, rct_sz_h, colors[1], 0.5);
			for (var i = 0; i < 4; i += 1) {
				text_id_(svg, a_text[i], x_ph - 273 + i * shift_w, y_start + 15, 1, 10, "vllm");
				text_id_(svg, a_text[i], x_ph + 2 + i * shift_w, y_start + 7 * shift_h + 15, 1, 10, "vllm");
			}
			if (token <= 8) {
				vector_block(svg, x_ph, y_start + shift_h, token - 4, 1, shift_w, shift_h, rct_sz_w, rct_sz_h, colors[1], 0.5);
				vector_block(svg, x_ph - 275, y_start + shift_h, token - 4, 1, shift_w, shift_h, rct_sz_w, rct_sz_h, colors[1], 0.5);
				for (var i = 4; i < token; i += 1) {
					text_id_(svg, a_text[i], x_ph - 273 + (i - 4) * shift_w, y_start + shift_h + 15, 1, 10, "vllm");
					text_id_(svg, a_text[i], x_ph + 2 + (i - 4) * shift_w, y_start + shift_h + 15, 1, 10, "vllm");
				}
			}
			else {
				vector_block(svg, x_ph, y_start + shift_h, 4, 1, shift_w, shift_h, rct_sz_w, rct_sz_h, colors[1], 0.5);
				vector_block(svg, x_ph - 275, y_start + shift_h, 4, 1, shift_w, shift_h, rct_sz_w, rct_sz_h, colors[1], 0.5);
				for (var i = 4; i < 8; i += 1) {
					text_id_(svg, a_text[i], x_ph - 273 + (i - 4) * shift_w, y_start + shift_h + 15, 1, 10, "vllm");
					text_id_(svg, a_text[i], x_ph + 2 + (i - 4) * shift_w, y_start + shift_h + 15, 1, 10, "vllm");
				}
				vector_block(svg, x_ph - 275, y_start + 2 * shift_h, token - 8, 1, shift_w, shift_h, rct_sz_w, rct_sz_h, colors[1], 0.5);
				vector_block(svg, x_ph, y_start + 3 * shift_h, token - 8, 1, shift_w, shift_h, rct_sz_w, rct_sz_h, colors[1], 0.5);
				text_id_(svg, a_text[i], x_ph + 2, y_start + 3 * shift_h + 15, 1, 10, "vllm");
				text_id_(svg, a_text[i], x_ph - 273, y_start + 2 * shift_h + 15, 1, 10, "vllm");
			}
		}
		
		if (token > 3) {
			if (token < 7) {
				vector_block(svg, x_ph - 275, y_start + 5 * shift_h, token - 3, 1, shift_w, shift_h, rct_sz_w, rct_sz_h, colors[0], 0.5);
				vector_block(svg, x_ph, y_start + 5 * shift_h, token - 3, 1, shift_w, shift_h, rct_sz_w, rct_sz_h, colors[0], 0.5);
				for (var i = 0; i < token - 3; i += 1) {
					text_id_(svg, b_text[i], x_ph - 273 + i * shift_w, y_start + 5 * shift_h + 15, 1, 10, "vllm");
					text_id_(svg, b_text[i], x_ph + 2 + i * shift_w, y_start + 5 * shift_h + 15, 1, 10, "vllm");
				}
			}
			else {
				vector_block(svg, x_ph - 275, y_start + 5 * shift_h, 4, 1, shift_w, shift_h, rct_sz_w, rct_sz_h, colors[0], 0.5);
				vector_block(svg, x_ph, y_start + 5 * shift_h, 4, 1, shift_w, shift_h, rct_sz_w, rct_sz_h, colors[0], 0.5);
				for (var i = 0; i < 4; i += 1) {
					text_id_(svg, b_text[i], x_ph - 273 + i * shift_w, y_start + 5 * shift_h + 15, 1, 10, "vllm");
					text_id_(svg, b_text[i], x_ph + 2 + i * shift_w, y_start + 5 * shift_h + 15, 1, 10, "vllm");
				}
				
				vector_block(svg, x_ph - 275, y_start + 6 * shift_h, token - 7, 1, shift_w, shift_h, rct_sz_w, rct_sz_h, colors[0], 0.5);
				vector_block(svg, x_ph, y_start + 2 * shift_h, token - 7, 1, shift_w, shift_h, rct_sz_w, rct_sz_h, colors[0], 0.5);
				for (var i = 4; i < token - 3; i += 1) {
					text_id_(svg, b_text[i], x_ph - 273 + (i - 4) * shift_w, y_start + 6 * shift_h + 15, 1, 10, "vllm");
					text_id_(svg, b_text[i], x_ph + 2 + (i - 4) * shift_w, y_start + 2 * shift_h + 15, 1, 10, "vllm");
				}
			}
		}
	}
	
	generate(1);
	
	var l_x = d3.scaleLinear()
	    .domain([1, 9])
	    .range([0, 415])
	    .clamp(true);
	    
	createSlider(svg, generate, l_x, x_start + 85, 280, "Tokens", "currentColor", 0, roundN, 9, 30, text_size=11);
}

paged_attn();
</script>

![](.)
*Storing the KV cache of two requests with PagedAttention. The logical blocks of the two sequences are mapped to different physical blocks within the space reserved by the block engine in GPU workers. Block manager maps the first three logical blocks to 7, 1 and 3 physical blocks for request A and first two blocks to 5 and 2 for request B.*

In parallel sampling multiple responses share the same input prompt, allowing the KV cache of the prompt to be shared as well. Thus the logical blocks for the prompts of all sequences are mapped to the same physical block, which stores a *reference count*. At the generation phase the first physical block which contains the response is replicated with reduced reference count and sampling continues with separate blocks. By sharing physical KV blocks across multiple samples, memory usage can be greatly reduced, especially for long input prompts.

Beam search decoding with PagedAttention is more advanced, since also blocks across diferent candidates can be shared and the sharing pattern changes dynamically as the decoding process progresses. As candidates no longer among top, their logical blocks are freed and the reference counts of corresponding physical blocks are reduced.

Additionaly to the algorithm authors released [**vLLM**](https://github.com/vllm-project/vllm), a high-throughput distributed LLM serving engine on top of PagedAttention. Their evaluations on various models and workloads show that vLLM improves the LLM serving throughput by 2-4× compared to the other existing serving systems, without affecting the model accuracy.

### RadixAttention

While vLLM is optimized for independent requests - like a chatbot answering unrelated questions from different users, another system, called [**SGLang (Structured Generation Language for LLMs)**](https://github.com/sgl-project/sglang), proposed by [Zheng et al. (2024)](https://arxiv.org/pdf/2312.07104), is designed for complex multi-step LLM workflows like AI agentts or applications that need to call the LLM multiple times with related prompts.

The core innovation of SGLang's core is **RadixAttention**. In traditional LLM serving each request computes its own KV cache. Even if two requests share a common prefix like "You are a helpful assistant. User: Hello!", both compute this prefix separately, which leads to massive waste of computation and memory. Instead of treating each generation call independently, SGLang identifies and reuses shared computational work like common prompt prefixes or KV cache values across multiple calls, dramatically improving efficiency. 

SGLang retains the cache for prompts and generation results in a [radix tree](https://en.wikipedia.org/wiki/Radix_tree), enabling efficient prefix search, reuse, insertion, and eviction. The radix tree structure can be thought as a smart family tree for prompts, where each path stores the computed KV cache, so shared prefixes are computed once and reused.

<div id="radix_attention" class="svg-container" align="center"></div> 

<script>

function draw_sampling_text_2(svg, x_start, y_start, rct_sz, shift) {
	text_(svg, "Q", x_start + 8, y_start + 2 * shift);
}

function radix_attention() {
	var svg = d3.select("#radix_attention")
				  .append("svg")
				  .attr("width", 550)
				  .attr("height", 370);
	const x_start = 50, y_start = 10;
	const shift = 30, rct_sz = 16;
	var init_timepoint = 9, dim = 2;
	
	function line(svg, x0, y0, x1, y1) {
		svg.append("path")
		   .datum([{x: x0, y: y0},
		           {x: x1, y: y1}])
		   .attr("fill", "none")
		   .attr("stroke-width", 1)
		   .attr("stroke", "currentColor")
		   .attr("d",  d3.line()
		       .x(function(d) { return d.x; })
		       .y(function(d) { return d.y; }));
	}
	
	function line_evicted(svg, x0, y0, x1, y1) {
		svg.append("path")
		   .datum([{x: x0, y: y0},
		           {x: x1, y: y1}])
		   .attr("fill", "none")
		   .attr("stroke-width", 1)
		   .attr("stroke", "Salmon")
	  		.style('stroke-dasharray', ('2,3'))
		   .attr("d",  d3.line()
		       .x(function(d) { return d.x; })
		       .y(function(d) { return d.y; }));
	}
	
	function text_id_3_(svg, text, x, y, size=14, color="currentColor", text_id="fattn") {
		svg.append('text')
		  .attr("id", text_id)
		  .attr('x', x)
		  .attr('y', y)
		  .attr('orig_x', x)
		  .attr('orig_y', y)
		  .text(text)
		  .style("fill", color)
		  .style("font-size", size + "px")
		  .attr("font-family", "Arvo");
	}
	
	function rect_evicted(svg, x, y, w, h) {
		svg.append('rect')
		  .attr('x', x)
		  .attr('y', y)
		  .attr('width', w)
		  .attr('height', h)
		  .attr('stroke', "Salmon")
		  .attr('stroke-width', 1)
	  	  .style('stroke-dasharray', ('2,3'))
		  .attr("rx", Math.min(3, Math.min(w, h) / 3))
		  .attr('fill', matrix_colors[1])
		  .attr('opacity', 1.0);
		  
		text_id_3_(svg, 'x', x + w * 0.28, y + 0.75 * h, size=14, color="Salmon", text_id="sglang");
	}
		
	function draw_request_evolution(timepoint) {
	   svg.selectAll('rect').remove();
	   svg.selectAll('path').remove();
	   svg.selectAll('#sglang').remove();
	   
	   rect(svg, x_start, y_start, rct_sz, rct_sz, matrix_colors[7]);
	   
	   if (timepoint == 1) {
	   		return;
	   }
	   
	   if (timepoint == 2 || timepoint == 3) {
	   	   line(svg, x_start + rct_sz / 2, y_start + rct_sz, x_start + rct_sz / 2, y_start + 3 * shift);
		   text_id_2_(svg, 'You are a helpful assistant.', x_start + rct_sz / 2 + 5, y_start + rct_sz + 20, size=10, text_id="sglang");
		   text_id_2_(svg, 'User: Hello!', x_start + rct_sz / 2 + 5, y_start + rct_sz + 35, size=10, text_id="sglang");
		   text_id_2_(svg, 'Assistant: Hi!', x_start + rct_sz / 2 + 5, y_start + rct_sz + 50, size=10, text_id="sglang");
		   if (timepoint == 2) {
			   rect(svg, x_start, y_start + 3 * shift, rct_sz, rct_sz, matrix_colors[4]);
		   } else {
			   rect(svg, x_start, y_start + 3 * shift, rct_sz, rct_sz, matrix_colors[0]);
		   	   line(svg, x_start + rct_sz / 2, y_start + 3 * shift + rct_sz, x_start + rct_sz / 2, y_start + 6 * shift);
				rect(svg, x_start, y_start + 6 * shift, rct_sz, rct_sz, matrix_colors[4]);
			   text_id_2_(svg, 'User: Solve this problem …', x_start + rct_sz / 2 + 5, y_start + 3 * shift + rct_sz + 30, size=10, text_id="sglang");
			   text_id_2_(svg, 'Assistant: Sure! …', x_start + rct_sz / 2 + 5, y_start + 3 * shift + rct_sz + 45, size=10, text_id="sglang");
		   }
		   return;
	   }
	   
	   text_id_2_(svg, 'You are a helpful assistant.', x_start + rct_sz / 2 + 5, y_start + rct_sz + 25, size=10, text_id="sglang");
	   text_id_2_(svg, 'User: Hello!', x_start + rct_sz / 2 + 5, y_start + rct_sz + 75, size=10, text_id="sglang");
	   text_id_2_(svg, 'Assistant: Hi!', x_start + rct_sz / 2 + 5, y_start + rct_sz + 90, size=10, text_id="sglang");
	   
		if (timepoint < 8 && timepoint > 3) {
		   text_id_2_(svg, 'User: What can you do?', x_start + 3 * shift + rct_sz / 2 + 5, y_start + rct_sz + 75, size=10, text_id="sglang");
		   text_id_2_(svg, 'Assistant: I can …', x_start + 3 * shift + rct_sz / 2 + 5, y_start + rct_sz + 90, size=10, text_id="sglang");
		}
		
	   if (timepoint == 4) {
	   	   line(svg, x_start + rct_sz / 2, y_start + rct_sz, x_start + rct_sz / 2, y_start + 2 * shift);
			rect(svg, x_start, y_start + 2 * shift, rct_sz, rct_sz, matrix_colors[0]);
		   text_id_2_(svg, 'User: Solve this problem …', x_start + rct_sz / 2 + 5, y_start + 3 * shift + rct_sz + 50, size=10, text_id="sglang");
		   text_id_2_(svg, 'Assistant: Sure! …', x_start + rct_sz / 2 + 5, y_start + 3 * shift + rct_sz + 65, size=10, text_id="sglang");
			
	   	   line(svg, x_start + rct_sz / 2, y_start + 2 * shift + rct_sz, x_start + rct_sz / 2, y_start + 4 * shift);
			rect(svg, x_start, y_start + 4 * shift, rct_sz, rct_sz, matrix_colors[7]);
			
	   	   line(svg, x_start + rct_sz / 2, y_start + 4 * shift + rct_sz, x_start + rct_sz / 2, y_start + 6 * shift);
			rect(svg, x_start, y_start + 6 * shift, rct_sz, rct_sz, matrix_colors[7]);
			
	   	   line(svg, x_start + rct_sz, y_start + 2 * shift + rct_sz / 2, x_start + 3 * shift + rct_sz / 2, y_start + 2 * shift + rct_sz / 2);
	   	   line(svg, x_start + 3 * shift + rct_sz / 2, y_start + 2 * shift + rct_sz / 2, x_start + 3 * shift + rct_sz / 2, y_start + 4 * shift);
			rect(svg, x_start + 3 * shift, y_start + 4 * shift, rct_sz, rct_sz, matrix_colors[4]);
		}
	   
	   if (timepoint == 5) {
	   	   line(svg, x_start + rct_sz / 2, y_start + rct_sz, x_start + rct_sz / 2, y_start + 2 * shift);
			rect(svg, x_start, y_start + 2 * shift, rct_sz, rct_sz, matrix_colors[0]);
	   	   line(svg, x_start + rct_sz / 2, y_start + 2 * shift + rct_sz, x_start + rct_sz / 2, y_start + 4 * shift);
			rect(svg, x_start, y_start + 4 * shift, rct_sz, rct_sz, matrix_colors[7]);
	   	   line_evicted(svg, x_start + rct_sz / 2, y_start + 4 * shift + rct_sz, x_start + rct_sz / 2, y_start + 6 * shift);
			rect_evicted(svg, x_start, y_start + 6 * shift, rct_sz, rct_sz);
			
	   	   line(svg, x_start + rct_sz, y_start + 2 * shift + rct_sz / 2, x_start + 3 * shift + rct_sz / 2, y_start + 2 * shift + rct_sz / 2);
	   	   line(svg, x_start + 3 * shift + rct_sz / 2, y_start + 2 * shift + rct_sz / 2, x_start + 3 * shift + rct_sz / 2, y_start + 4 * shift);
			rect(svg, x_start + 3 * shift, y_start + 4 * shift, rct_sz, rct_sz, matrix_colors[0]);
	   	   line(svg, x_start + 3 * shift + rct_sz / 2, y_start + 4 * shift + rct_sz, x_start + 3 * shift + rct_sz / 2, y_start + 6 * shift);
			rect(svg, x_start + 3 * shift, y_start + 6 * shift, rct_sz, rct_sz, matrix_colors[4]);
			
		   text_id_2_(svg, 'User: Write a story …', x_start + 3 * shift + rct_sz / 2 + 5, y_start + 3 * shift + rct_sz + 50, size=10, text_id="sglang");
		   text_id_2_(svg, 'Assistant: Sure! …', x_start + 3 * shift + rct_sz / 2 + 5, y_start + 3 * shift + rct_sz + 65, size=10, text_id="sglang");
		   text_id_3_(svg, 'evicted', x_start + rct_sz / 2 + 5, y_start + 3 * shift + rct_sz + 50, size=10, color="Salmon", text_id="sglang");
		}
		
		if (timepoint > 5) {
		   text_id_2_(svg, 'Question 1: …', x_start + 8 * shift + rct_sz / 2 + 5, y_start + rct_sz + 15, size=10, text_id="sglang");
		   text_id_2_(svg, 'Answer 1: …', x_start + 8 * shift + rct_sz / 2 + 5, y_start + rct_sz + 30, size=10, text_id="sglang");
		   text_id_2_(svg, 'Question 2: …', x_start + 8 * shift + rct_sz / 2 + 5, y_start + rct_sz + 45, size=10, text_id="sglang");
		   text_id_2_(svg, 'Answer 2: …', x_start + 8 * shift + rct_sz / 2 + 5, y_start + rct_sz + 60, size=10, text_id="sglang");
		   text_id_2_(svg, 'Question 3: …', x_start + 8 * shift + rct_sz / 2 + 5, y_start + rct_sz + 75, size=10, text_id="sglang");
		  }
		   
	   if (timepoint < 8 && timepoint > 5) {
		   text_id_2_(svg, 'User: Write a story …', x_start + 3 * shift + rct_sz / 2 + 5, y_start + 3 * shift + rct_sz + 50, size=10, text_id="sglang");
		   text_id_2_(svg, 'Assistant: Sure! …', x_start + 3 * shift + rct_sz / 2 + 5, y_start + 3 * shift + rct_sz + 65, size=10, text_id="sglang");
		   		   
	   	   line(svg, x_start + rct_sz / 2, y_start + rct_sz, x_start + rct_sz / 2, y_start + 2 * shift);
			rect(svg, x_start, y_start + 2 * shift, rct_sz, rct_sz, matrix_colors[7]);
	   	   line(svg, x_start + rct_sz / 2, y_start + 2 * shift + rct_sz, x_start + rct_sz / 2, y_start + 4 * shift);
			rect(svg, x_start, y_start + 4 * shift, rct_sz, rct_sz, matrix_colors[7]);
			
	   	   line(svg, x_start + rct_sz, y_start + 2 * shift + rct_sz / 2, x_start + 3 * shift + rct_sz / 2, y_start + 2 * shift + rct_sz / 2);
	   	   line(svg, x_start + 3 * shift + rct_sz / 2, y_start + 2 * shift + rct_sz / 2, x_start + 3 * shift + rct_sz / 2, y_start + 4 * shift);
			rect(svg, x_start + 3 * shift, y_start + 4 * shift, rct_sz, rct_sz, matrix_colors[7]);
	   	   line(svg, x_start + 3 * shift + rct_sz / 2, y_start + 4 * shift + rct_sz, x_start + 3 * shift + rct_sz / 2, y_start + 6 * shift);
			rect(svg, x_start + 3 * shift, y_start + 6 * shift, rct_sz, rct_sz, matrix_colors[7]);
			
	   	   line(svg, x_start + rct_sz, y_start + rct_sz / 2, x_start + 8 * shift + rct_sz / 2, y_start + rct_sz / 2);
	   	   line(svg, x_start + 8 * shift + rct_sz / 2, y_start + rct_sz / 2, x_start + 8 * shift + rct_sz / 2, y_start + 4 * shift);
	   	   
	   	   if (timepoint == 6) {
				rect(svg, x_start + 8 * shift, y_start + 4 * shift, rct_sz, rct_sz, matrix_colors[4]);
			} else {
				rect(svg, x_start + 8 * shift, y_start + 4 * shift, rct_sz, rct_sz, matrix_colors[0]);
	   	   		line(svg, x_start + 8 * shift + rct_sz / 2, y_start + 4 * shift + rct_sz, x_start + 8 * shift + rct_sz / 2, y_start + 6 * shift);
				rect(svg, x_start + 8 * shift, y_start + 6 * shift, rct_sz, rct_sz, matrix_colors[7]);
	   	   		line(svg, x_start + 8 * shift + rct_sz, y_start + 4 * shift + rct_sz / 2, x_start + 14 * shift + rct_sz / 2, y_start + 4 * shift + rct_sz / 2);
	   	   		line(svg, x_start + 11 * shift + rct_sz / 2, y_start + 4 * shift + rct_sz / 2, x_start + 11 * shift + rct_sz / 2, y_start + 6 * shift);
				rect(svg, x_start + 11 * shift, y_start + 6 * shift, rct_sz, rct_sz, matrix_colors[4]);
	   	   		line(svg, x_start + 14 * shift + rct_sz / 2, y_start + 4 * shift + rct_sz / 2, x_start + 14 * shift + rct_sz / 2, y_start + 6 * shift);
				rect(svg, x_start + 14 * shift, y_start + 6 * shift, rct_sz, rct_sz, matrix_colors[4]);
			}
		}
		
		if (6 < timepoint) {
		   text_id_2_(svg, 'What …', x_start + 8 * shift + rct_sz / 2 + 5, y_start + 3 * shift + rct_sz + 50, size=10, text_id="sglang");
		   text_id_2_(svg, 'Answer 3: …', x_start + 8 * shift + rct_sz / 2 + 5, y_start + 3 * shift + rct_sz + 65, size=10, text_id="sglang");
		   if (timepoint < 9) {
			   text_id_2_(svg, 'When …', x_start + 11 * shift + rct_sz / 2 + 5, y_start + 3 * shift + rct_sz + 50, size=10, text_id="sglang");
			   text_id_2_(svg, 'Answer 3: …', x_start + 11 * shift + rct_sz / 2 + 5, y_start + 3 * shift + rct_sz + 65, size=10, text_id="sglang");
			   text_id_2_(svg, 'How …', x_start + 14 * shift + rct_sz / 2 + 5, y_start + 3 * shift + rct_sz + 50, size=10, text_id="sglang");
			   text_id_2_(svg, 'Answer 3: …', x_start + 14 * shift + rct_sz / 2 + 5, y_start + 3 * shift + rct_sz + 65, size=10, text_id="sglang");
			}
		}
			
	   if (timepoint == 8) {
	   	   line(svg, x_start + rct_sz / 2, y_start + rct_sz, x_start + rct_sz / 2, y_start + 2 * shift);
			rect(svg, x_start, y_start + 2 * shift, rct_sz, rct_sz, matrix_colors[0]);
	   	   line(svg, x_start + rct_sz / 2, y_start + 2 * shift + rct_sz, x_start + rct_sz / 2, y_start + 4 * shift);
			rect(svg, x_start, y_start + 4 * shift, rct_sz, rct_sz, matrix_colors[0]);
	   	   line(svg, x_start + rct_sz / 2, y_start + 4 * shift + rct_sz, x_start + rct_sz / 2, y_start + 6 * shift);
			rect(svg, x_start, y_start + 6 * shift, rct_sz, rct_sz, matrix_colors[4]);
			
	   	   line_evicted(svg, x_start + rct_sz, y_start + 2 * shift + rct_sz / 2, x_start + 5 * shift + rct_sz / 2, y_start + 2 * shift + rct_sz / 2);
	   	   line_evicted(svg, x_start + 5 * shift + rct_sz / 2, y_start + 2 * shift + rct_sz / 2, x_start + 5 * shift + rct_sz / 2, y_start + 4 * shift);
			rect_evicted(svg, x_start + 5 * shift, y_start + 4 * shift, rct_sz, rct_sz);
	   	   line_evicted(svg, x_start + 5 * shift + rct_sz / 2, y_start + 4 * shift + rct_sz, x_start + 5 * shift + rct_sz / 2, y_start + 6 * shift);
			rect_evicted(svg, x_start + 5 * shift, y_start + 6 * shift, rct_sz, rct_sz);
			
	   	   line(svg, x_start + rct_sz, y_start + rct_sz / 2, x_start + 8 * shift + rct_sz / 2, y_start + rct_sz / 2);
	   	   line(svg, x_start + 8 * shift + rct_sz / 2, y_start + rct_sz / 2, x_start + 8 * shift + rct_sz / 2, y_start + 4 * shift);
   	   
			rect(svg, x_start + 8 * shift, y_start + 4 * shift, rct_sz, rct_sz, matrix_colors[7]);
   	   		line(svg, x_start + 8 * shift + rct_sz / 2, y_start + 4 * shift + rct_sz, x_start + 8 * shift + rct_sz / 2, y_start + 6 * shift);
			rect(svg, x_start + 8 * shift, y_start + 6 * shift, rct_sz, rct_sz, matrix_colors[7]);
   	   		line(svg, x_start + 8 * shift + rct_sz, y_start + 4 * shift + rct_sz / 2, x_start + 14 * shift + rct_sz / 2, y_start + 4 * shift + rct_sz / 2);
   	   		line(svg, x_start + 11 * shift + rct_sz / 2, y_start + 4 * shift + rct_sz / 2, x_start + 11 * shift + rct_sz / 2, y_start + 6 * shift);
			rect(svg, x_start + 11 * shift, y_start + 6 * shift, rct_sz, rct_sz, matrix_colors[7]);
   	   		line(svg, x_start + 14 * shift + rct_sz / 2, y_start + 4 * shift + rct_sz / 2, x_start + 14 * shift + rct_sz / 2, y_start + 6 * shift);
			rect(svg, x_start + 14 * shift, y_start + 6 * shift, rct_sz, rct_sz, matrix_colors[7]);
		   text_id_3_(svg, 'evicted', x_start + 5 * shift + rct_sz / 2 + 5, y_start + shift + rct_sz + 50, size=10, color="Salmon", text_id="sglang");
		   
		   text_id_2_(svg, 'User: Solve this question …', x_start + rct_sz / 2 + 5, y_start + 3 * shift + rct_sz + 50, size=10, text_id="sglang");
		   text_id_2_(svg, 'Assistant: Sure! …', x_start + rct_sz / 2 + 5, y_start + 3 * shift + rct_sz + 65, size=10, text_id="sglang");
		   
		}
			
	   if (timepoint == 9) {
	   	   line(svg, x_start + rct_sz / 2, y_start + rct_sz, x_start + rct_sz / 2, y_start + 2 * shift);
			rect(svg, x_start, y_start + 2 * shift, rct_sz, rct_sz, matrix_colors[7]);
	   	   line(svg, x_start + rct_sz / 2, y_start + 2 * shift + rct_sz, x_start + rct_sz / 2, y_start + 4 * shift);
			rect(svg, x_start, y_start + 4 * shift, rct_sz, rct_sz, matrix_colors[7]);
	   	   line_evicted(svg, x_start + rct_sz / 2, y_start + 4 * shift + rct_sz, x_start + rct_sz / 2, y_start + 6 * shift);
			rect_evicted(svg, x_start, y_start + 6 * shift, rct_sz, rct_sz);
			
	   	   line(svg, x_start + rct_sz, y_start + rct_sz / 2, x_start + 8 * shift + rct_sz / 2, y_start + rct_sz / 2);
	   	   line(svg, x_start + 8 * shift + rct_sz / 2, y_start + rct_sz / 2, x_start + 8 * shift + rct_sz / 2, y_start + 4 * shift);
			rect(svg, x_start + 8 * shift, y_start + 4 * shift, rct_sz, rct_sz, matrix_colors[0]);
   	   		line(svg, x_start + 8 * shift + rct_sz / 2, y_start + 4 * shift + rct_sz, x_start + 8 * shift + rct_sz / 2, y_start + 6 * shift);
			rect(svg, x_start + 8 * shift, y_start + 6 * shift, rct_sz, rct_sz, matrix_colors[0]);
   	   		line_evicted(svg, x_start + 8 * shift + rct_sz, y_start + 4 * shift + rct_sz / 2, x_start + 14 * shift + rct_sz / 2, y_start + 4 * shift + rct_sz / 2);
   	   		line_evicted(svg, x_start + 11 * shift + rct_sz / 2, y_start + 4 * shift + rct_sz / 2, x_start + 11 * shift + rct_sz / 2, y_start + 6 * shift);
			rect_evicted(svg, x_start + 11 * shift, y_start + 6 * shift, rct_sz, rct_sz);
   	   		line_evicted(svg, x_start + 14 * shift + rct_sz / 2, y_start + 4 * shift + rct_sz / 2, x_start + 14 * shift + rct_sz / 2, y_start + 6 * shift);
			rect_evicted(svg, x_start + 14 * shift, y_start + 6 * shift, rct_sz, rct_sz);
		   text_id_3_(svg, 'evicted', x_start + rct_sz / 2 + 5, y_start + 3 * shift + rct_sz + 50, size=10, color="Salmon", text_id="sglang");
		   text_id_3_(svg, 'evicted', x_start + 9 * shift + rct_sz / 2 + 5, y_start + 2 * shift + rct_sz + 50, size=10, color="Salmon", text_id="sglang");
		   
		   for (var i = 0; i < 4; i += 1) {
	   	   		line(svg, x_start + 8 * shift + rct_sz / 2, y_start + 6 * shift + rct_sz, x_start + 8 * shift + rct_sz / 2, y_start + 7 * shift + rct_sz);
	   	   		line(svg, x_start + 5 * shift + rct_sz / 2, y_start + 7 * shift + rct_sz, x_start + 11 * shift + rct_sz / 2, y_start + 7 * shift + rct_sz);
	   	   		line(svg, x_start + (5 + 2 * i) * shift + rct_sz / 2, y_start + 7 * shift + rct_sz, x_start + (5 + 2 * i) * shift + rct_sz / 2, y_start + 9 * shift);
				rect(svg, x_start + (5 + 2 * i) * shift, y_start + 9 * shift, rct_sz, rct_sz, i == 0 ? matrix_colors[7] : matrix_colors[4]);
		   }
		   
		   text_id_2_(svg, 'This is …', x_start + 5 * shift + rct_sz / 2 + 3, y_start + 6 * shift + rct_sz + 55, size=10, text_id="sglang");
		   text_id_2_(svg, 'Let us …', x_start + 7 * shift + rct_sz / 2 + 3, y_start + 6 * shift + rct_sz + 55, size=10, text_id="sglang");
		   text_id_2_(svg, 'We can …', x_start + 9 * shift + rct_sz / 2 + 3, y_start + 6 * shift + rct_sz + 55, size=10, text_id="sglang");
		   text_id_2_(svg, 'To solve …', x_start + 11 * shift + rct_sz / 2 + 3, y_start + 6 * shift + rct_sz + 55, size=10, text_id="sglang");
		}
	
		init_timepoint = timepoint;
	}

	draw_request_evolution(init_timepoint);

	var l_x = d3.scaleLinear()
	    .domain([1, 9])
	    .range([0, 320])
	    .clamp(true);

	createSlider(svg, draw_request_evolution, l_x, x_start + 40, 350, "Step", "currentColor", init_timepoint, roundN, 9);
}

radix_attention();

</script>

![](.)
*Examples of RadixAttention operations with an LRU eviction policy, illustrated across nine time points. The nodes are color-coded to reflect different states: green for newly added nodes, blue for cached nodes accessed during the time point, and red for nodes that have been evicted*

When GPU memory fills up, SGLang removes least recently used branches. By evicting leaves first, it is possible to re-use their common ancestors until those ancestors become leaves and are also evicted. Also they propose cache-aware scheduling policy: instead of processing requests first-come-first-served, SGLang reorders them to maximize cache hits.

## Conclusion

This essay has covered a wide range of transformer inference optimization techniques, from high-level algorithmic improvements like KV caching and MQA/GQA/MLA, to low-level hardware optimizations such as CUDA kernel fusion and vLLM. The key takeaway is clear: LLMs are resource-intensive beasts, and taming them requires a diverse toolkit.

We've seen that successful ML engineering isn't just about understanding algorithms or hardware - it's about grasping how they work together. An algorithmic breakthrough might fall flat without the right hardware implementation, while a deep dive into GPU architecture could unlock unexpected performance gains.

For ML engineers, the message is simple: stay curious and be ready to learn across the entire stack. The most effective optimizations often come from connecting the dots between different areas of expertise. As LLMs continue to grow and evolve, so too will the tricks we use to run them efficiently.

---
[^GTL]: Full names given to GPU architectures are: **T**uring, **V**olta, **A**mpere, **H**opper, **B**lackwell.

[^MMM]: In general, the number of flops for a matrix-matrix multiplication $\mathbf{AB}$ with $\mathbf{A} \in \mathbb{R}^{n \times m}$ and $\mathbf{B} \in \mathbb{R}^{m \times k}$ is near $2mnk$: we perform $m$ matrix-vector multiplications, each of which can be represented as $n$ inner products, and each inner product requires $k-1$ additions and $k$ multiplications.

[^VD]: Dimension size of values $\mathbf{V}$ can be different from $d$, but usually it's not the case.

[^AIL]: A reasonable question might be: "What is the best way to utilize GPU to generate small sequences, e.g. $L \ll d$?" A possible solution is to enlarge batch processing, since one can compute that for $x \in \mathbb{R}^{B \times d}$ the arithmetic intensity is $$\frac{BL^2d + BLd^2}{BL^2h + BLd + d^2} \xrightarrow[d \gg L]{} BL $$

[^HP]: While it is clear that *hedgehog* comes from attention "spikyness" modelling, I still wonder what *porcupine* in the title refers to.

[^TC]: This isn't unique to GPUs - in fact, TPUs are even less general than GPUs.

[^OISWA]: If we don't use SWA.