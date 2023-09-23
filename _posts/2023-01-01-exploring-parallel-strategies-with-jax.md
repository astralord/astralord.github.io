---
layout: post
title: 'Introduction to Parallel Strategies'
date: 2023-01-01 11:00 +0800
categories: [ML Engineering]
tags: [data-parallel, model-parallel, pipeline-paralell, tensor-parallel, mixture-of-experts]
math: true
enable_d3: true
published: false
---

colors = ['#8ED3C7', '#D9D9D9', '#FDBFB9', '#FFF6B7',
          '#D9EFB5', '#FFFFDA', '#E6F5E2', '#FDB462']

> Training large language models like GPT-3 or LLaMA requires immense computational resources. With model sizes ballooning into the billions or even trillions of parameters, specialized parallelization techniques are essential to make training feasible. In this post, we’ll explore implementing some of these scaling strategies in Jax - a Python framework designed for high-performance numerical computing with support for accelerators like GPU and TPU. 

- Sources:
	- [Tim Dettmers Data](https://timdettmers.com/2014/10/09/deep-learning-data-parallelism/)
	- [Tim Dettmers Model](https://timdettmers.com/2014/11/09/model-parallelism-deep-learning/)
	- [Megatron-LM](https://arxiv.org/pdf/1909.08053.pdf) (Shoeybi et al. 2020)
	- [Another Megatron](https://arxiv.org/pdf/2201.11990.pdf)

### Sharding and Pipelining

Jax is a great fit for implementing parallel LLM training thanks to its high-level APIs for composing parallel functions and its seamless acceleration on GPU/TPU hardware. We’ll walk through code examples for data, model and pipeline parallelism in Jax while training a smaller "toy" LLM model for demonstration. The insights from this exercise will help us understand how state-of-the-art systems actually parallelize and distribute LLM training in practice.

> xmap is an experimental API. Expect rough edges and changes in the future!

<script src="https://d3js.org/d3.v4.min.js"></script>
<link href="https://fonts.googleapis.com/css?family=Arvo" rel="stylesheet">

<div id="lgnd" class="svg-container" align="center"></div> 

<script>

colors = ['#C7E9E3', '#ECECEC', '#FDBFB9', '#FFF6B7',
          '#D9EFB5', '#FFFFDA', '#E6F5E2', '#FEDAB1']

function cell(svg, x, y, color) {
	svg.append('rect')
	  .attr('x', x)
	  .attr('y', y)
	  .attr('width', 14)
	  .attr('height', 14)
	  .attr('stroke', 'black')
	  .attr('stroke-width', 1)
	  .attr("rx", 3)
	  .attr('fill', color);
}

function tensor(svg, x, y, w, h, color) {	for (var i = 0; i < w; i += 1) {
		for (var j = 0; j < h; j += 1) {
		   cell(svg, x + i * 16, y + j * 16, color);
		}
	}
}

function rplc_tensor(svg, x, y, w, h) {
   tensor(svg, x, y, w, h, 'lightgrey');
}

function shrd_tensor(svg, x, y, w, h, xsh, ysh) {
   xsplit = ~~(w / xsh);
   ysplit = ~~(h / ysh);
	for (var i = 0; i < ysh; i += 1) {
		for (var j = 0; j < xsh; j += 1) {
		   tensor(svg, 
		          x + j * 16 * xsplit, 
		          y + i * 20 * ysplit,
		          xsplit,
		          ysplit,
		          colors[i * xsh + j]);
		}
	}
}

function relu(svg, x, y) {
		  
	svg.append('circle')
		  .attr('cx', x)
		  .attr('cy', y)
		  .attr('r', 10)
		  .attr('stroke', 'black')
		  .attr('stroke-width', 1)
		  .attr('fill', 'none');
	
	svg.append("path")
	   .attr("stroke", "black")
	   .datum([{x: x - 9, y: y + 1},
	           {x: x, y: y + 1},  
	           {x: x + 7, y: y - 6}])
	   .attr("fill", "none")
	   .attr("stroke-width", 1)
	   .attr("stroke", "black")
	   .attr("d",  d3.line()
        .curve(d3.curveBasis)
	       .x(function(d) { return d.x; })
	       .y(function(d) { return d.y; }));
}

function aggregate(svg, x, y, h) {
	svg.append("path")
	   .attr("stroke", "black")
	   .datum([{x: x, y: y},
				 {x: x + 5, y: y},
	           {x: x + 5, y: y + h}, 
	           {x: x, y: y + h}])
	   .attr("fill", "none")
	   .attr("stroke-width", 1)
	   .attr("stroke", "black")
	   .attr("d",  d3.line()
	       .x(function(d) { return d.x; })
	       .y(function(d) { return d.y; }));
	       
	svg.append('text')
	  .attr('x', x + 5)
	  .attr('y', y + h / 2 + 2)
	  .text("⟶")
	  .style("font-size", "12px")
	  .attr("font-family", "Arvo");
}

function split(svg, x, y, h) {
	svg.append("path")
	   .attr("stroke", "black")
	   .datum([{x: x + 5, y: y},
				 {x: x, y: y},
	           {x: x, y: y + h}, 
	           {x: x + 5, y: y + h}])
	   .attr("fill", "none")
	   .attr("stroke-width", 1)
	   .attr("stroke", "black")
	   .attr("d",  d3.line()
	       .x(function(d) { return d.x; })
	       .y(function(d) { return d.y; }));
}

function right_arrow(svg, x, y) {
	svg.append('text')
	  .attr('x', x)
	  .attr('y', y)
	  .text("⟶")
	  .style("font-size", "11px")
	  .attr("font-family", "Arvo");
}

function grad_right_arrow(svg, x, y) {
	svg.append('text')
	  .attr('x', x + 2)
	  .attr('y', y - 8)
	  .text("∇")
	  .style("font-size", "9px")
	  .attr("font-family", "Arvo");
	right_arrow(svg, x, y);
}

function cdot(svg, x, y) {
	svg.append('text')
	  .attr('x', x)
	  .attr('y', y)
	  .text("⋅")
	  .style("font-size", "21px")
	  .attr("font-family", "Arvo");
}

function cdots(svg, x, y) {
	svg.append('text')
	  .attr('x', x)
	  .attr('y', y)
	  .text("⋅⋅⋅")
	  .style("font-size", "11px")
	  .attr("font-family", "Arvo");
}

function sync(svg, x, y) {
	svg.append('text')
	  .attr('x', x)
	  .attr('y', y)
	  .text("⇅")
	  .style("font-size", "11px")
	  .attr("font-family", "Arvo");
}

function legend() {

	var svg = d3.select("#lgnd")
				  .append("svg")
				  .attr("width", 600)
				  .attr("height", 90);
	
	svg.append('text')
	  .attr('x', 130)
	  .attr('y', 20)
	  .text("Sharded")
	  .style("font-size", "14px")
	  .attr("font-family", "Arvo");
	
	svg.append('text')
	  .attr('x', 415)
	  .attr('y', 20)
	  .text("Replicated")
	  .style("font-size", "14px")
	  .attr("font-family", "Arvo");
	  
	svg.append('line')
	  .attr('x1', 300)
	  .attr('y1', 0)
	  .attr('x2', 300)
	  .attr('y2', 90)
	  .style("stroke-width", 3)
	  .attr("opacity", 0.7)
	  .attr('stroke', 'black');
	
	shrd_tensor(svg, 127, 40, 4, 2, 4, 1);
	rplc_tensor(svg, 370, 40, 4, 2, 4, 1);
	rplc_tensor(svg, 470, 40, 4, 2, 4, 1);
	
	svg.append('rect')
	  .attr('x', 365)
	  .attr('y', 35)
	  .attr('width', 170)
	  .attr('height', 40)
	  .attr('stroke', 'black')
	  .attr("rx", 3)
	  .style('stroke-dasharray', ('2,3'))
	  .attr('fill', 'none');
}

legend();
</script>

![](.)
*Visualization of tensors locations. Each color stands for separate device. On the left side - tensor is split by 4 subtensors, each located on its own device. On the right side - tensor is copied on 2 devices.*

$$\operatorname{FFN}(x) = \max(0, x\mathbf{W}_1)\mathbf{W}_2$$

$$x \in \mathbb{R}^{n \times d}, \mathbf{W}_1 \in \mathbb{R}^{d \times h}, \mathbf{W}_2 \in \mathbb{R}^{h \times d}$$

```python
import jax
import jax.numpy as jnp
from jax import grad, jit, random
from jax.debug import visualize_array_sharding

@jit
def ffn(x, w1, w2):
  h = jnp.maximum(x @ w1, 0)
  return h @ w2
```

#### Data Parallelism 

**Data Parallel (DP)** is a relatively simple strategy: the training data is partitioned across distributed workers and fed to the model, which is replicated. The training process is done in parallel: each worker computes the model gradients on its independent data split. Gradients are then synchronized at the end of each training step before the weight update, so that all workers observe consistent model parameters throughout training. This way DP strategy allows scaling to large data batches.

<div id="data_prll" class="svg-container" align="center"></div> 

<script>
function data_parallel() {
	var svg = d3.select("#data_prll")
				  .append("svg")
				  .attr("width", 650)
				  .attr("height", 500);
	     
	svg.append('text')
	  .attr('x', 11)
	  .attr('y', 20)
	  .text("x")
	  .style("font-size", "17px")
	  .attr("font-family", "Arvo");
	 	     
	svg.append('text')
	  .attr('x', 63)
	  .attr('y', 23)
	  .text("W")
	  .style("font-size", "17px")
	  .attr("font-family", "Arvo");
	  
	svg.append('text')
	  .attr('x', 82)
	  .attr('y', 27)
	  .text("1")
	  .style("font-size", "8px")
	  .attr("font-family", "Arvo");
	     
	svg.append('text')
	  .attr('x', 216)
	  .attr('y', 23)
	  .text("W")
	  .style("font-size", "17px")
	  .attr("font-family", "Arvo");
	  
	svg.append('text')
	  .attr('x', 235)
	  .attr('y', 27)
	  .text("2")
	  .style("font-size", "8px")
	  .attr("font-family", "Arvo");
		  
	x_start = 10;
	y_shift = 100;
	k = 4;
	shrd_tensor(svg, x_start, 30, 2, 5 * k, 1, k);
	
	for (var i = 0; i < k; i += 1) 
	{
		cdot(svg, x_start + 36, 75 + i * y_shift);
		rplc_tensor(svg, x_start + 50, 54 + i * y_shift, 4, 2);
		right_arrow(svg, x_start + 130, 71 + i * y_shift);
	}
	
	svg.append('rect')
	  .attr('x', 55)
	  .attr('y', 50)
	  .attr('width', 72)
	  .attr('height', 85 * k)
	  .attr('stroke', 'black')
	  .attr("rx", 3)
	  .style('stroke-dasharray', ('2,3'))
	  .attr('fill', 'none');

	shrd_tensor(svg, x_start + 156, 30, 4, 5 * k, 1, k);
	
	for (var i = 0; i < k; i += 1) {
	   relu(svg, x_start + 237, 68 + i * y_shift);
	   cdot(svg, x_start + 252, 75 + i * y_shift);
	   rplc_tensor(svg, x_start + 267, 38 + i * y_shift, 2, 4);
	   right_arrow(svg, x_start + 316, 71 + i * y_shift);
	}
	
	svg.append('rect')
	  .attr('x', 272)
	  .attr('y', 30)
	  .attr('width', 40)
	  .attr('height', 95 * k)
	  .attr('stroke', 'black')
	  .attr("rx", 3)
	  .style('stroke-dasharray', ('2,3'))
	  .attr('fill', 'none');
	
	shrd_tensor(svg, x_start + 344, 30, 2, 5 * k, 1, k);
		
	for (var i = 0; i < k; i += 1) {
		cdots(svg, x_start + 387, 72 + i * y_shift);
		aggregate(svg, x_start + 412, 28 + i * y_shift, 82);
		cell(svg, x_start + 432, 62 + i * y_shift, colors[i]);
		grad_right_arrow(svg, x_start + 449, 71 + i * y_shift);
		cdots(svg, x_start + 472, 72 + i * y_shift);
		tensor(svg, x_start + 498, 38 + i * y_shift, 2, 4, colors[i]);
		
		if (i < k - 1) {
			sync(svg, x_start + 508, 123 + i * y_shift);
		}
		
		grad_right_arrow(svg, x_start + 547, 71 + i * y_shift);
		tensor(svg, x_start + 575, 54 + i * y_shift, 4, 2, colors[i]);
		
		if (i < k - 1) {
			sync(svg, x_start + 600, 123 + i * y_shift);
		}
	}
		     
	svg.append('text')
	  .attr('x', 150)
	  .attr('y', 460)
	  .text("Forward")
	  .style("font-size", "17px")
	  .attr("font-family", "Arvo");
	  
	svg.append('text')
	  .attr('x', 420)
	  .attr('y', 460)
	  .text("Loss")
	  .style("font-size", "17px")
	  .attr("font-family", "Arvo");
	     
	svg.append('text')
	  .attr('x', 510)
	  .attr('y', 460)
	  .text("Backward")
	  .style("font-size", "17px")
	  .attr("font-family", "Arvo");
}

data_parallel();
</script>

```python
x = jax.device_put(x, sharding.reshape(n_devices, 1))
w = jax.device_put(w, sharding.replicate())
```

The main problem with DP approach is that during the backward pass all the gradients must be transferred to the all other devices. For example, with $\mathbf{W}_1 \in \mathbb{R}^{d \times h}$ weight matrix in float32 one need to pass $4dh$ bytes to each device. The same amount is needed for $\mathbf{W}_2$. If we work in a multi-node setup, having $v$ GBit/s of network card bandwidth, we'll need 

$$t = \frac{64dh}{v \cdot 1024^3}$$

seconds to pass the gradients for FFN layer from one node to another (plus an additional overhead $\delta t$ that is neglected here). Imagine if we use only data parallelism for LLM pretraining and we need to give a rough estimation of the time required for backward calculations. In GPT-3 embedding size $d$ is 12⋅1024 and hidden size $h$ is 48⋅1024. [Microsoft built a supercomputer exclusively for OpenAI](https://news.microsoft.com/source/features/innovation/openai-azure-supercomputer/) with 10,000 GPUs and 400 Gbps of network connectivity between nodes. Plugging these numbers we get

$$ t = \frac{64 \cdot 12 \cdot 48}{400 \cdot 1024} = 90\mbox{ms}$$

just to transfer FFN gradients. As there are 96 FFN layers in GPT-3, it'll take about 9 seconds for this part of gradient synchronization. And this is just to pass data from one node to another, while there might be thousands of them. Easily we can see that data parallelism does not scale with size of the cluster.

The described strategy above is called **Distributed Data Parallel (DDP)** and it is different from [HuggingFace definition of data paralelism](https://huggingface.co/docs/transformers/perf_train_gpu_many#data-parallelism). Their version of DP helps to overcome slow intra-node connectivity by minimizing the amount of synchronized data and delegating a lot of data/gradient processing to one single GPU. This, in turn, results in under-utilization of other devices. 

Another common strategy for amortizing communication cost is **gradient accumulation**. Before synchronizing data and taking an optimizer step, we can run multiple forward and backward propagations and accumulate local gradients on each device in parallel. Additionally, performance can be improved by simultaneously synchronizing the computed gradients and computing gradients for other tensors.

#### Model Parallelism 

With the advent of large neural networks that do not fit on one device, the need to parallelize models has increased. This is especially true in the case of language models, whose number of parameters can exceed the already considerable amount of input data.

- Splitting the model across devices/nodes to distribute computation and memory. Can parallelize across multiple dimensions.

**Pipeline Parallel (PP)** - the model is split up vertically (layer-level) across multiple GPUs/TPUs, so that only one or several layers of the model are places on a single device. Each device processes in parallel different stages of the pipeline and working on a small chunk of the batch?

<div id="mdl_prll" class="svg-container" align="center"></div> 

<script>

function model_parallel() {
	var svg = d3.select("#mdl_prll")
				  .append("svg")
				  .attr("width", 600)
				  .attr("height", 200);
	
	tensor(svg, 1, 30, 2, 4, colors[0]);
	cdot(svg, 37, 67);
	tensor(svg, 51, 46, 3, 2, colors[0]);
	right_arrow(svg, 115, 63);
	tensor(svg, 141, 30, 3, 4, colors[0]);
	
	svg.append('text')
		  .attr('x', 161)
		  .attr('y', 105)
		  .text("↓")
		  .style("font-size", "11px")
		  .attr("font-family", "Arvo");

	tensor(svg, 141, 110, 3, 4, colors[1]);
	relu(svg, 206, 140);
	cdot(svg, 221, 147);
	tensor(svg, 236, 118, 1, 3, colors[1]);
	right_arrow(svg, 269, 143);
	cell(svg, 297, 134, colors[1]);
	right_arrow(svg, 330, 143);
	tensor(svg, 358, 118, 1, 3, colors[1]);
	
	svg.append('text')
		  .attr('x', 362)
		  .attr('y', 105)
		  .text("↑")
		  .style("font-size", "11px")
		  .attr("font-family", "Arvo");
}

model_parallel();
</script>


#### Tensor Parallelism

So far we've looked at two cases: data partitioning and vertical model split. Now, can we split the model weights the same way we did it with data? Sure, we can divide each tensor $\mathbf{W}$ into chunks distributed across multiple devices, so instead of having the whole tensor reside on a single device, each shard of the tensor resides on its designated GPU/TPU. During processing each part gets processed separately and in parallel on different devices and the results are synced at the end of the step. 

This is called **Tensor Parallel (TP)** or horizontal parallelism, as the splitting happens on horizontal level. One simple and efficient way to do it was proposed in [Megatron-LM](https://arxiv.org/pdf/1909.08053.pdf) paper. Let's represent matrices $\mathbf{W}_1$ and $\mathbf{W}_2$ as concatenation of $k$ sub-tensors along rows and columns respectively:

$$\mathbf{W}_1 = \begin{pmatrix} \color{#8ED3C7}{\mathbf{W}^1_1} & \color{#D9D9D9}{\mathbf{W}^2_1} & \cdots & \color{#FDB462}{\mathbf{W}^k_1} \end{pmatrix}, \quad \mathbf{W}_2 = \begin{pmatrix} \color{#8ED3C7}{\mathbf{W}^1_2} \\ \color{#D9D9D9}{\mathbf{W}^2_2} \\ \vdots \\ \color{#FDB462}{\mathbf{W}^k_2} \end{pmatrix}.$$

Then we can perform tensors multiplications on different devices independently:

$$x\mathbf{W}_1=\begin{pmatrix} x\color{#8ED3C7}{\mathbf{W}_1^1} & x\color{#D9D9D9}{\mathbf{W}_1^2} &\cdots & x\color{#FDB462}{\mathbf{W}^k_1} \end{pmatrix}$$

and for $z = \max(x\mathbf{W}_1, 0)$ we have

$$z\mathbf{W}_2=z\color{#8ED3C7}{\mathbf{W}_2^1} + z\color{#D9D9D9}{\mathbf{W}_2^2} + \cdots + z\color{#FDB462}{\mathbf{W}^k_2}.$$

Model weights on different devices do not overlap and the only communication between devices is in the end of FFN layer, when we need to sum all of the outputs. The size of synchronized weights is the same as the input splitted by $k$, hence if we operate with float32, we need to pass $\frac{4dn}{k}$ bytes to each GPU/TPU. Now with growing number of devices, the amount of data flowing between nodes doesn't grow as fast as with DP. The drawback here is that while we were operating on input data with batch size $k \times n$ with Data Parallel, with Tensor Parallel strategy input data is replicated and thus always of the same size $n$.

<div id="tnsr_prll" class="svg-container" align="center"></div> 

<script>
function tensor_parallel() {
	var svg = d3.select("#tnsr_prll")
				  .append("svg")
				  .attr("width", 600)
				  .attr("height", 500);
	x_start = 10;
	y_start = 60;
	y_shift = 100;
	
	shrd_tensor(svg, x_start, 10, 4, 2, 4, 1);
	shrd_tensor(svg, x_start + 500, 10, 2, 4, 1, 4);
	
	for (var i = 0; i < 4; i += 1) {
		rplc_tensor(svg, x_start, y_start + i * y_shift, 2, 5);
		cdot(svg, x_start + 36, y_start + 45 + i * y_shift);
		tensor(svg, x_start + 50, y_start + 24 + i * y_shift, 1, 2, colors[i]);
		right_arrow(svg, x_start + 78, y_start + 41 + i * y_shift);
	}
	
	svg.append('rect')
	  .attr('x', 5)
	  .attr('y', y_start - 10)
	  .attr('width', 40)
	  .attr('height', 100 * k)
	  .attr('stroke', 'black')
	  .attr("rx", 3)
	  .style('stroke-dasharray', ('2,3'))
	  .attr('fill', 'none');
	
	shrd_tensor(svg, x_start + 103, y_start, 1, 20, 1, 4);
	
	for (var i = 0; i < 4; i += 1) {
	   relu(svg, x_start + 136, y_start + 38 + i * y_shift);
		cdot(svg, x_start + 151, y_start + 45 + i * y_shift);
		tensor(svg, x_start + 167, y_start + 32 + i * y_shift, 2, 1, colors[i]);
		right_arrow(svg, x_start + 211, y_start + 41 + i * y_shift);	}
		
	shrd_tensor(svg, x_start + 236, y_start, 2, 20, 1, 4);
	
	for (var i = 0; i < 3; i += 1) {
	   sync(svg, x_start + 246, y_start + 93 + i * y_shift);	}
		
	for (var i = 0; i < 4; i += 1) {
	   cdots(svg, x_start + 284, y_start + 42 + i * y_shift);	}
		
	aggregate(svg, x_start + 318, y_start - 2, 382);
	cell(svg, x_start + 345, y_start + 182, 'lightgrey');
	grad_right_arrow(svg, x_start + 366, y_start + 191);
	
	for (var i = 0; i < 4; i += 1) {
	   cdots(svg, x_start + 389, y_start + 42 + i * y_shift);	}
	   
	split(svg, x_start + 389, y_start - 2, 382);
	
	for (var i = 0; i < 4; i += 1) {
		grad_right_arrow(svg, x_start + 422, y_start + 41 + i * y_shift);
		tensor(svg, x_start + 447, y_start + 24 + i * y_shift, 1, 2, colors[i]);
	}
}

tensor_parallel();
</script>

```python
x = jax.device_put(x, sharding.replicate())
w = jax.device_put(w, sharding.reshape(1, n_devices))
```

We can see that the amount memory transfer from each device is $O(dh)$ for DP versus $O(dn)$ for TP. Thus for DP is a preferable strategy for small networks (e.g. model can fit onto one GPU or at least onto one node), while TP works better with larger models and smaller batches.

TODO: check with https://irhum.github.io/blog/pjit/#partitioning-in-jax

#### Hybrid data and model tensor parallelism

Take the best of both worlds

Across multiple dimensions

<div id="cmbn_prll" class="svg-container" align="center"></div> 

<script>
function combined_parallel() {
	var svg = d3.select("#cmbn_prll")
				  .append("svg")
				  .attr("width", 600)
				  .attr("height", 200);	
				  
	x_start = 50;
	y_start = 60;
	y_shift = 100;
	
	rplc_tensor(svg, x_start, y_start, 2, 5);
	rplc_tensor(svg, x_start - 5, y_start + 5, 2, 5);
}

combined_parallel();
</script>


TAKEN FROM JAX DOCS:
When running on CPU you can always emulate an arbitrary number of devices with a nifty `--xla_force_host_platform_device_count` XLA flag. This is especially useful for debugging and testing locally or even for prototyping in Colab since a CPU runtime is faster to (re-)start.

```python
import os
# Use 8 CPU devices
os.environ["XLA_FLAGS"] = '--xla_force_host_platform_device_count=8' 

import matplotlib as mpl
from jax.sharding import PositionalSharding
from jax.debug import visualize_array_sharding
from jax import random

# create random tensor
key = random.PRNGKey(0)
tensor = random.normal(key, (16, 128))
sharding = PositionalSharding(jax.devices())
# shard tensor on 8 devices
tensor = jax.device_put(tensor, sharding.reshape(1, 8))
visualize_array_sharding(tensor, color_map=mpl.colormaps["Set3"])
```

![Tensor parallel on CPU]({{'/assets/img/tp_mac.png'|relative_url}})
*Tensor sharding on CPU*


| Setup | Model fits onto a single GPU | Model doesn’t fit onto a single GPU | Largest Layer not fitting into a single GPU | Total parameters
|---|---|---|---|---|
|Single GPU| No parallel| ZeRO + Offload CPU and optionally NVMe + Memory Centric Tiling | ZeRO - Enable Memory Centric Tiling (MCT). It allows you to run arbitrarily large layers by automatically splitting them and executing them sequentially. MCT reduces the number of parameters that are live on a GPU, but it does not affect the activation memory. As this need is very rare as of this writing a manual override of torch.nn.Linear needs to be done by the user. | 117M |
| Multi-GPU | DDP / ZeRO (may or may not be faster depending on the situation and configuration used) | PP / TP / ZeRO With very fast intra-node connectivity of NVLINK or NVSwitch all three should be mostly on par, without these PP will be faster than TP or ZeRO. The degree of TP may also make a difference. Best to experiment to find the winner on your particular setup. TP is almost always used within a single node. That is TP size <= gpus per node. | If not using ZeRO - must use TP, as PP alone won’t be able to fit. With ZeRO see the same entry for “Single GPU” above | 1.5B |
| Multi-node | When you have fast inter-node connectivity: ZeRO - as it requires close to no modifications to the model. PP+TP+DP - less communications, but requires massive changes to the model. When you have slow inter-node connectivity and still low on GPU memory: DP+PP+TP+ZeRO-1 | | 2048 tokens | 175B |


### Mixture-of-Experts (MoE)

#### Original MoE Transformer Encoder

#### GShard visualization

### Conclusion


Wasn't included:
Zero Redundancy Optimizer (ZeRO) - Also performs sharding of the tensors somewhat similar to TP, except the whole tensor gets reconstructed in time for a forward or backward computation, therefore the model does’t need to be modified. It also supports various offloading techniques to compensate for limited GPU memory.
