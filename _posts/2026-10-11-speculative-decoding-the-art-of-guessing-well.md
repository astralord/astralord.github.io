---
layout: post
title: 'Speculative Decoding: The Art of Guessing Well'
date: 2026-10-11 12:00 +0200
categories: [Large Language Models, Engineering]
tags: [jax, speculative decoding, eagle-3, multi-token prediction, dflash, dspark, block diffusion, lk losses]
math: true
enable_d3: true
published: true
---

> Large Language Models spend most of their decoding time not computing, but waiting: every new token requires reading all of the model weights from memory, only to multiply them by a single vector. What if the model could check several tokens for the price of one?
> 
> In this post we'll derive speculative decoding from first principles, build a general framework that covers every modern drafter, and walk through the architectures that became the industry standard, from EAGLE-3 and multi-token prediction to block-diffusion drafters like DFlash and DSpark. We'll finish with the question of how to train a drafter, and why minimizing KL divergence is not quite what we want.

This post is a sequel to [Transformers Inference Optimization Toolset](https://astralord.github.io/posts/transformer-inference-optimization-toolset/). We'll reuse its notation, its memory bandwidth arithmetic and its KV cache calculations, so if the phrase "decoding is memory-bound" sounds unfamiliar, it's worth skimming the first sections there.

Some speculative decoding techniques will be left out. We'll focus on **drafters attached to the target model**: small networks that read the target's hidden states and run next to it on the same GPU. A separate small LLM appears only as a baseline, drafting by string matching (n-grams, prompt lookup, suffix trees) goes to a footnote[^JAC], and serving systems appear only where they shape how drafters are built and used.

## Why decoding is slow

Let's recall that [text generation](https://astralord.github.io/posts/transformer-inference-optimization-toolset/#kv-cache) runs in two stages. During prefill the model ingests the whole prompt in one pass, and it is compute-bound. During decoding it produces one token per forward pass, and the [arithmetic intensity](https://astralord.github.io/posts/transformer-inference-optimization-toolset/#arithmetic-intensity-vs-opsbyte) of each pass is far below the `ops:byte` ratio of the GPU. At batch size 1, every linear layer multiplies its weight matrix by the hidden state of a single token, a vector of size $d$, so each weight is read from HBM, used for one multiplication and one addition, and left alone until the next step.

A real world example: take an 8B model in bf16 on H100 GPU. Just reading its weights once takes

$$\underset{\text{weights, bf16}}{16 \text{ GB}} \ / \ \underset{\text{H100 HBM bandwidth}}{3.35 \text{ TB/s}} \approx 4.8 \text{ ms} \ \Rightarrow \ \le 208 \ \tfrac{\text{tokens}}{\text{s}} \ \text{ for an 8B model at batch size 1},$$

no matter how fast the tensor cores are. Meanwhile the math for one token is one multiply-add per weight, i.e. $2 \cdot 8 \cdot 10^9 = 16$ GFLOPs (ignoring attention over the KV cache), which H100 does in about $0.016$ ms at its $989$ TFLOPs of dense bf16 compute. And here's the punchline: if we fed the model 8 tokens instead of one, we'd do 8 times more math and still pay nearly the same 4.8 ms, because the weights are read once either way. The pass stays memory-bound until the number of tokens scored in it reaches the `ops:byte` ratio, $989 / 3.35 \approx 295$.

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

<div id="memory_bound_decode" class="svg-container" align="center"></div> 

<script>

const colors = ['#65AD69', '#EDA137', '#E86456', '#348ABD'];
const green = colors[0], red = colors[2], blue = colors[3];
const matrix_colors = ['#C7E9E3', '#FDBFB9', '#FFF6B7', '#FEDAB1', 
                       '#D9EFB5', '#FFFFDA', '#E6F5E2', '#B3B3B3'];
const orange = colors[1];
const context_color = '#ECECEC', target_color = matrix_colors[3], drafter_color = blue;
const draft_token_color = matrix_colors[0], head_color = matrix_colors[1], fuse_color = matrix_colors[2];

function rect(svg, x, y, w, h, color, opacity=1.0) {
	return svg.append('rect')
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
	return svg.append('text')
	  .attr('x', x)
	  .attr('y', y)
	  .text(text)
	  .style("font-size", size + "px")
	  .style("fill", color)
	  .attr("font-family", "Arvo");
}

function line_(svg, x0, y0, x1, y1, color="currentColor", width=1, dash="none") {
	return svg.append("path")
	   .datum([{x: x0, y: y0},
	           {x: x1, y: y1}])
	   .attr("fill", "none")
	   .attr("stroke-width", width)
	   .attr("stroke", color)
	   .style('stroke-dasharray', dash)
	   .attr("d",  d3.line()
	       .x(function(d) { return d.x; })
	       .y(function(d) { return d.y; }));
}

function arrow_(svg, x0, y0, x1, y1, color="currentColor", width=1) {
	line_(svg, x0, y0, x1, y1, color, width);
	const angle = Math.atan2(y1 - y0, x1 - x0), size = 5;
	svg.append("path")
	   .datum([{x: x1 - size * Math.cos(angle - 0.4), y: y1 - size * Math.sin(angle - 0.4)},
	           {x: x1, y: y1},
	           {x: x1 - size * Math.cos(angle + 0.4), y: y1 - size * Math.sin(angle + 0.4)}])
	   .attr("fill", "none")
	   .attr("stroke-width", width)
	   .attr("stroke", color)
	   .attr("d",  d3.line()
	       .x(function(d) { return d.x; })
	       .y(function(d) { return d.y; }));
}

function arrow_head(svg, x, y, angle, color="currentColor") {
	const size = 5;
	svg.append("path")
	   .datum([{x: x - size * Math.cos(angle - 0.4), y: y - size * Math.sin(angle - 0.4)},
	           {x: x, y: y},
	           {x: x - size * Math.cos(angle + 0.4), y: y - size * Math.sin(angle + 0.4)}])
	   .attr("fill", "none")
	   .attr("stroke", color)
	   .attr("d",  d3.line()
	       .x(function(d) { return d.x; })
	       .y(function(d) { return d.y; }));
}

function bezier_(svg, x0, y0, c0x, c0y, c1x, c1y, x1, y1, color="currentColor", dash="none", head=false) {
	svg.append("path")
	   .attr("d", "M " + x0 + " " + y0 + " C " + c0x + " " + c0y + ", " + c1x + " " + c1y + ", " + x1 + " " + y1)
	   .attr("fill", "none")
	   .attr("stroke", color)
	   .style("stroke-dasharray", dash);
	if (head) {
		arrow_head(svg, x1, y1, Math.atan2(y1 - c1y, x1 - c1x), color);
	}
}

function link_h(svg, x0, y0, x1, y1, color="currentColor", dash="none", head=false) {
	const mx = (x0 + x1) / 2;
	bezier_(svg, x0, y0, mx, y0, mx, y1, x1, y1, color, dash, head);
}

function arc_arrow(svg, x0, y0, x1, y1, lift, color="currentColor") {
	const mx = (x0 + x1) / 2, my = Math.min(y0, y1) - lift;
	svg.append("path")
	   .attr("d", "M " + x0 + " " + y0 + " Q " + mx + " " + my + " " + x1 + " " + y1)
	   .attr("fill", "none")
	   .attr("stroke", color);
	const angle = Math.atan2(y1 - my, x1 - mx), size = 5;
	svg.append("path")
	   .datum([{x: x1 - size * Math.cos(angle - 0.4), y: y1 - size * Math.sin(angle - 0.4)},
	           {x: x1, y: y1},
	           {x: x1 - size * Math.cos(angle + 0.4), y: y1 - size * Math.sin(angle + 0.4)}])
	   .attr("fill", "none")
	   .attr("stroke", color)
	   .attr("d",  d3.line()
	       .x(function(d) { return d.x; })
	       .y(function(d) { return d.y; }));
}

function sub_text(svg, base, sub, x, y, size=12, color="currentColor") {
	const label = text_(svg, base, x, y, size, color);
	label.append("tspan").text(sub).attr("dy", size * 0.3).style("font-size", (size * 0.72) + "px");
	return label;
}

function token_box(svg, x, y, w, token, color, opacity=1.0, size=12) {
	rect(svg, x, y, w, 22, color, opacity);
	return text_(svg, token, x + w / 2, y + 15, size, "black")
	  .attr("text-anchor", "middle")
	  .attr('opacity', opacity);
}

function arvo_axis(axis) {
	axis.selectAll(".tick text").attr("font-family", "Arvo");
	return axis;
}

function trivialRound(x) { return x; }
function roundN(x) { return Math.round(x); }
function round2(x) { return Math.round(x * 100) / 100; }

function createSlider(svg_, parameter_update, x, loc_x, loc_y, letter, color, init_val, round_fun, x_ticks, shift_text=23, text_size=14) {
    const slider = svg_.append("g")
      .attr("class", "slider")
      .attr("transform", "translate(" + loc_x + "," + loc_y + ")");
    
    const drag = d3.drag()
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

   const handle = slider.insert("circle", ".track-overlay")
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

function tabs_(svg, labels, x, y, on_select, init=0) {
	const items = [];
	let shift = x;
	labels.forEach(function(label) {
		items.push(text_(svg, label, shift, y, 13).style("cursor", "pointer"));
		shift += label.length * 8 + 30;
	});
	function select(i) {
		items.forEach(function(item, j) {
			item.style("font-weight", i == j ? "bold" : "normal")
			    .style("text-decoration", i == j ? "underline" : "none")
			    .style("opacity", i == j ? 1.0 : 0.6);
		});
		on_select(i);
	}
	items.forEach(function(item, i) { item.on("click", function() { select(i); }); });
	select(init);
}

function memory_bound_decode() {
	const svg = d3.select("#memory_bound_decode")
				  .append("svg")
				  .attr("viewBox", "0 0 700 300")
				  .style("width", "100%")
				  .style("max-width", "700px")
				  .style("height", "auto");
	
	const x_start = 90, x_end = 610, y_start = 20, y_end = 200;
	const t_memory = 16e9 / 3.35e12 * 1e3;
	const t_compute = function(n) { return 2 * 8e9 * n / 989e12 * 1e3; };
	const step_time = function(n) { return Math.max(t_memory, t_compute(n)); };
	const knee = 989 / 3.35;
	
	const x = d3.scaleLinear().domain([1, 1024]).range([x_start, x_end]);
	const y = d3.scaleLinear().domain([0, 18]).range([y_end, y_start]);
	
	arvo_axis(svg.append("g")
	   .attr("transform", "translate(0," + y_end + ")")
	   .call(d3.axisBottom(x).tickValues([1, 128, 256, 384, 512, 640, 768, 896, 1024])));
	arvo_axis(svg.append("g")
	   .attr("transform", "translate(" + x_start + ", 0)")
	   .call(d3.axisLeft(y).ticks(4)));
	   
	text_(svg, "ms", x_start - 40, y_start + 5, 11);
	text_(svg, "n", x_end + 15, y_end + 4, 11);
	
	line_(svg, x_start, y(t_memory), x_end, y(t_memory), "currentColor", 1, "4,3");
	text_(svg, "reading 16 GB of weights: 4.8 ms", x_start + 210, y(t_memory) + 16, 11);
	line_(svg, x(1), y(t_compute(1)), x(1024), y(t_compute(1024)), orange, 1, "4,3");
	text_(svg, "compute: 2 · 8B · n FLOPs", x(560), y(t_compute(1024)) + 10, 11, orange);
	line_(svg, x(knee), y_end, x(knee), y(t_memory) - 30, "currentColor", 1, "2,3");
	text_(svg, "ops:byte ≈ 295", x(knee) - 45, y(t_memory) - 36, 11);
	
	const curve = [];
	for (let n = 1; n <= 1024; n += 8) {
		curve.push({x: n, y: step_time(n)});
	}
	svg.append("path")
	   .datum(curve)
	   .attr("fill", "none")
	   .attr("stroke", "currentColor")
	   .attr("stroke-width", 2)
	   .attr("d", d3.line()
	       .curve(d3.curveBasis)
	       .x(function(d) { return x(d.x); })
	       .y(function(d) { return y(d.y); }));
	
	const dot = svg.append("circle")
	   .attr("r", 5)
	   .attr("fill", orange)
	   .attr("stroke", "black");
	const pass_text = text_(svg, "", x_start + 20, y_start + 15, 13);
	const token_text = text_(svg, "", x_start + 20, y_start + 35, 13);
	
	function update(n) {
		const t = step_time(n);
		dot.attr("cx", x(n)).attr("cy", y(t));
		pass_text.text("one pass over " + n + (n == 1 ? " token: " : " tokens: ") + t.toFixed(2) + " ms");
		token_text.text("per token: " + (t / n).toFixed(3) + " ms");
	}
	update(8);
	
	const n_x = d3.scaleLinear()
	    .domain([1, 1024])
	    .range([0, 400])
	    .clamp(true);
	createSlider(svg, update, n_x, x_start + 80, 260, "n", "currentColor", 8, roundN, 5);
}

memory_bound_decode();

</script>

![](.)
*Time of one forward pass of an 8B model in bf16 on H100 versus the number of tokens $n$ scored in that pass: $\max \big( \frac{16 \text{ GB}}{3.35 \text{ TB/s}}, \frac{2 \cdot 8 \cdot 10^9 \cdot n}{989 \text{ TFLOPs}} \big)$, ignoring KV cache reads. Until $n \approx 295$ the pass is bound by reading the weights, so extra tokens are almost free.*

Batching requests is one way to fill this gap, and serving engines batch aggressively. But batching improves throughput, not the latency of each request: every user still gets one token per 4.8 ms. To make a single sequence faster, the extra tokens in the pass have to come from the same sequence, and we don't know them yet. Unless we guess them.

### Draft and verify

Suppose we have a cheap **drafter** $q$, a model that guesses the next tokens much faster than the **target** model $p$ we actually want to sample from. Let's start with greedy decoding, where the target always picks its most likely token. One round of **speculative decoding** looks like this:

1. The drafter proposes $\gamma$ tokens $x_{t+1}, \dots, x_{t+\gamma}$ that continue the verified context $x_{\leq t}$.
2. The target runs *once* on the context extended with all $\gamma$ guesses. Thanks to causal attention, this single pass gives its next-token prediction at each of the $\gamma + 1$ positions.
3. We keep the longest prefix of guesses on which the target's argmax agrees with the drafter. At the first disagreement we take the target's own token instead, and if all $\gamma$ guesses agree, the target's prediction after the last one comes for free.

Every token we keep is exactly the token greedy decoding would have produced, so the output doesn't change, it just arrives faster. Each round costs one target pass and yields at least one token and at most $\gamma + 1$. The KV cache entries for the accepted tokens are already written during the verification pass, and the entries of rejected tokens are simply dropped.

<div id="draft_verify" class="svg-container" align="center"></div> 

<script>

function draft_verify() {
	const svg = d3.select("#draft_verify")
				  .append("svg")
				  .attr("viewBox", "0 0 700 265")
				  .style("width", "100%")
				  .style("max-width", "700px")
				  .style("height", "auto");
	
	const context = ["The", "cat", "sat", "on", "the"];
	const draft = ["mat", "and", "purred", "softly"];
	const target = ["mat", "and", "slept", "on", "."];
	const num_accepted = 2;
	const x_start = 100, w = 52, step = 58;
	const draft_y = 40, target_y = 120, output_y = 205;
	const g = svg.append("g");
	let phase = 0;
	
	text_(svg, "drafter q", 5, draft_y + 15, 13, drafter_color);
	text_(svg, "target p", 5, target_y + 15, 13);
	text_(svg, "output", 5, output_y + 15, 13);
	
	function draw() {
		g.selectAll("*").remove();
		const shown = Math.min(phase, draft.length);
		
		for (let i = 0; i < context.length; i += 1) {
			token_box(g, x_start + i * step, draft_y, w, context[i], context_color);
		}
		
		for (let k = 0; k < shown; k += 1) {
			let color = draft_token_color;
			if (phase >= 6) {
				color = k < num_accepted ? green : red;
			}
			const x = x_start + (context.length + k) * step;
			token_box(g, x, draft_y, w, draft[k], color, phase >= 6 && k > num_accepted ? 0.4 : 1.0);
			if (phase < 5) {
				text_(g, "pass " + (k + 1), x + 10, draft_y - 8, 10, drafter_color);
			}
		}
		
		if (phase >= 5) {
			for (let k = 0; k <= draft.length; k += 1) {
				const x = x_start + (context.length + k) * step;
				token_box(g, x, target_y, w, target[k], target_color, phase >= 6 && k > num_accepted ? 0.4 : 1.0);
				if (k < draft.length) {
					line_(g, x + w / 2, draft_y + 24, x + w / 2, target_y - 2, "currentColor", 1, "2,3");
				}
			}
			g.append('rect')
			  .attr('x', x_start + context.length * step - 4)
			  .attr('y', target_y - 4)
			  .attr('width', (draft.length + 1) * step + 2)
			  .attr('height', 30)
			  .attr('stroke', 'currentColor')
			  .attr('stroke-width', 2)
			  .attr("rx", 3)
			  .style('stroke-dasharray', ('2,3'))
			  .attr('fill', 'none');
			text_(g, "one target pass scores all " + (draft.length + 1) + " positions", 
			      x_start + context.length * step, target_y + 44, 11);
		}
		
		if (phase >= 6) {
			for (let k = 0; k < draft.length; k += 1) {
				const x = x_start + (context.length + k) * step;
				text_(g, k < num_accepted ? "✓" : (k == num_accepted ? "✗" : ""), 
				      x + w / 2 + 6, (draft_y + target_y + 20) / 2, 14, k < num_accepted ? green : red);
			}
		}
		
		if (phase >= 7) {
			for (let i = 0; i < context.length; i += 1) {
				token_box(g, x_start + i * step, output_y, w, context[i], context_color);
			}
			for (let k = 0; k < num_accepted; k += 1) {
				token_box(g, x_start + (context.length + k) * step, output_y, w, draft[k], green);
			}
			token_box(g, x_start + (context.length + num_accepted) * step, output_y, w, target[num_accepted], target_color);
			text_(g, "+3 tokens", x_start + (context.length + num_accepted + 1) * step + 4, output_y + 10, 12);
			text_(g, "for 1 target pass", x_start + (context.length + num_accepted + 1) * step + 4, output_y + 25, 12);
		}
	}
	
	draw();
	d3.interval(function() {
		phase = (phase + 1) % 10;
		draw();
	}, 1100);
}

draft_verify();

</script>

![](.)
*One greedy round with $\gamma = 4$. The drafter proposes tokens one by one (light blue), then the target scores all of them in a single pass (orange). The longest agreeing prefix is accepted (green), the rest is rejected (red), and the target's own token at the first mismatch is appended: 3 new tokens for the price of one target pass.*

### Speculative sampling

Greedy matching is not enough when we sample with a temperature: the target now has a distribution $p$ rather than one correct answer, and we want our output to follow exactly that distribution. **Speculative sampling** ([Leviathan et al., 2023](https://arxiv.org/pdf/2211.17192); [Chen et al., 2023](https://arxiv.org/pdf/2302.01318)) generalizes the matching rule:

$$\text{accept } x \sim q \ \text{ w.p. } \min\Big(1, \frac{p(x)}{q(x)}\Big), \qquad \text{otherwise sample from } \ \frac{(p - q)_+}{\sum_x (p(x) - q(x))_+},$$

where $(\cdot)\_+ = \max(\cdot, 0)$. Intuitively, tokens that the drafter underestimates, $q(x) \leq p(x)$, are always kept, tokens it overestimates are thinned out, and the residual distribution tops up exactly the mass the drafter was missing.

How often is a drafted token accepted? Averaging the acceptance probability over the drafter's own samples gives the **acceptance rate**

$$\alpha = \mathbb{E}_{x \sim q}\Big[\min\Big(1, \frac{p(x)}{q(x)}\Big)\Big] = \sum_x \min\big(p(x), q(x)\big),$$

the overlap of the two distributions: probability mass that both models put on the same tokens is accepted, mass the drafter puts where the target doesn't is wasted, and mass it misses is what the residual has to supply. The missing part has a standard name: it's the **total variation distance** between the two distributions, the largest difference in probability they can assign to any set of tokens:

$$1 - \alpha = \operatorname{TV}(p, q) = \frac{1}{2}\sum_x \vert p(x) - q(x) \vert.$$

Leviathan et al. denote it $D_{LK}$. The closer the drafter is to the target in TV, the more of its guesses survive.

Now we can see why the output still follows $p$. The residual's normalizer is $\sum_x (p(x) - q(x))\_+ = 1 - \alpha$, the probability of a rejection, so the probability to output token $x$ is

$$
\begin{aligned}
\Pr(\text{output} = x) &= q(x)\min\Big(1, \frac{p(x)}{q(x)}\Big) + (1 - \alpha)\,\frac{(p(x) - q(x))_+}{1 - \alpha} \\ &= \min\big(p(x), q(x)\big) + \big(p(x) - q(x)\big)_+ \\ &= p(x).
\end{aligned}
$$

So whatever the drafter does, the output follows the target exactly, and a bad drafter can only make us slower. Greedy matching is the special case where both $p$ and $q$ are one-hot. For a draft of $\gamma$ tokens the rule is applied left to right and stops at the first rejection. Here's the whole verification step in JAX. When all $\gamma$ tokens are accepted, there's no drafter distribution at the next position, and treating it as zero turns the residual into the target's own distribution for the bonus token:

```python
@jit
def verify(key, p, q, draft):
    """p: [γ+1, V] target probs, q: [γ, V] drafter probs, draft: [γ] tokens."""
    gamma = draft.shape[0]
    key_u, key_r = jax.random.split(key)
    p_x = p[jnp.arange(gamma), draft]
    q_x = q[jnp.arange(gamma), draft]
    accepted = jax.random.uniform(key_u, (gamma,)) < jnp.minimum(1.0, p_x / q_x)
    n = jnp.argmin(jnp.append(accepted, False))                     # first rejection
    q_n = jnp.where(n < gamma, q[jnp.minimum(n, gamma - 1)], 0.0)   # zero after γ
    residual = jnp.maximum(p[n] - q_n, 0.0)
    token = jax.random.categorical(key_r, jnp.log(residual))
    return n, token  # keep draft[:n], then append token
```

### Sampling with shared noise

Speculative sampling has one property that can be inconvenient: its output depends on the drafter. With the same random seed, a different drafter makes different accept/reject decisions and samples different residuals, so the text changes, even though its distribution doesn't. To see an alternative, let's look at how sampling from an LLM is actually implemented.

Sampling the next token means sampling from $\operatorname{softmax}(\ell)$ over a vocabulary of $V$ tokens, where $\ell$ are the logits. The textbook way normalizes the logits, computes the cumulative sum of the probabilities and inverts it with a single uniform number. GPUs prefer the **Gumbel-max trick**: add independent Gumbel noise to the logits and take the argmax,

$$x = \arg\max_i \big(\ell_i + g_i\big), \qquad g_i = -\log(-\log u_i), \quad u_i \sim U(0, 1).$$

Why does this sample from the softmax? Let $\xi_i = e^{-g_i} = -\log u_i$, an exponential random variable with rate 1. Maximizing $\ell_i + g_i$ is the same as minimizing $\xi_i e^{-\ell_i}$, which is exponential with rate $e^{\ell_i}$. Think of independent exponential clocks: the probability that clock $i$ rings first is its rate divided by the sum of all rates, so

$$\Pr(x = i) = \Pr\Big(\xi_i e^{-\ell_i} < \min_{j \neq i} \xi_j e^{-\ell_j}\Big) = \frac{e^{\ell_i}}{\sum_j e^{\ell_j}} = \operatorname{softmax}(\ell)_i.$$

One elementwise pass and an argmax replace the normalization and the prefix sum over the vocabulary. vLLM's sampler, for example, runs exactly this race, taking the argmax of $p_i / \xi_i$, because `torch.multinomial` would force the GPU to synchronize with the CPU. For us the trick matters for another reason: it separates all the randomness from the model. Once the noise $g$ is fixed, sampling is a deterministic argmax, just like greedy decoding.

So let the drafter and the target share the noise. At draft position $k$ both use the same vector $g_k$: the drafter proposes $x_{t+k} = \arg\max_i \big(\log q_k(i) + g_{k,i}\big)$, the target computes its own $\arg\max_i \big(\log p_k(i) + g_{k,i}\big)$ in the verification pass, and we keep the longest prefix where the two agree, plus the target's token at the first disagreement. Verification becomes exact matching, as in the greedy case. This is **Gumbel coupling** ([Daliri et al., 2024](https://arxiv.org/pdf/2408.07978)), and its output is exactly what the target would have sampled with that noise on its own. It's **drafter-invariant**: for a fixed seed every drafter produces the same text, only faster or slower, which makes speculative decoding reproducible and much easier to test and debug.

<div id="gumbel_coupling" class="svg-container" align="center"></div> 

<script>

function gumbel_coupling() {
	const svg = d3.select("#gumbel_coupling")
				  .append("svg")
				  .attr("viewBox", "0 0 700 325")
				  .style("width", "100%")
				  .style("max-width", "700px")
				  .style("height", "auto");
	
	const tokens = ["mat", "floor", "sofa", "bed", "rug", "roof", "chair", "lap"];
	const p = [0.40, 0.20, 0.12, 0.10, 0.08, 0.05, 0.03, 0.02];
	const q = [0.22, 0.30, 0.05, 0.05, 0.18, 0.10, 0.05, 0.05];
	const x_start = 80, x_end = 450, y_start = 52, row = 28;
	const y_end = y_start + tokens.length * row;
	const x = d3.scaleLinear().domain([-6, 6]).range([x_start, x_end]).clamp(true);
	const alpha = d3.sum(p, function(v, i) { return Math.min(v, q[i]); });
	const shared_match = d3.sum(p, function(pj, j) {
		return 1 / d3.sum(p, function(pi, i) { return Math.max(pi / pj, q[i] / q[j]); });
	});
	const independent_match = d3.sum(p, function(v, i) { return v * q[i]; });
	
	let seed = 12;
	const uniform = function() { seed = (seed * 16807) % 2147483647; return seed / 2147483647; };
	const gumbel = function() { return -Math.log(-Math.log(uniform())); };
	const argmax = function(values) { return values.indexOf(d3.max(values)); };
	
	arvo_axis(svg.append("g")
	   .attr("transform", "translate(0," + y_end + ")")
	   .call(d3.axisBottom(x).ticks(6)));
	text_(svg, "score = log-probability + noise", x_start + 90, y_end + 35, 11);
	tokens.forEach(function(token, i) {
		text_(svg, token, x_start - 10, y_start + i * row + 15, 12).attr("text-anchor", "end");
	});
	
	const g = svg.append("g");
	const legend_x = 480;
	svg.append("circle").attr("cx", legend_x + 6).attr("cy", 59).attr("r", 5).attr("fill", orange).attr("stroke", "black");
	text_(svg, "target: log p + g", legend_x + 18, 63, 12);
	svg.append("circle").attr("cx", legend_x + 6).attr("cy", 81).attr("r", 5).attr("fill", drafter_color).attr("stroke", "black");
	const drafter_legend = text_(svg, "", legend_x + 18, 85, 12);
	const result_text = text_(svg, "", legend_x, 118, 13);
	const output_text = text_(svg, "", legend_x, 138, 12);
	const rate_text = text_(svg, "", legend_x, 176, 12);
	text_(svg, "expected match rate:", legend_x, 202, 12);
	const shared_text = text_(svg, "shared noise: " + shared_match.toFixed(3), legend_x + 10, 220, 12);
	const independent_text = text_(svg, "independent noise: " + independent_match.toFixed(3), legend_x + 10, 238, 12);
	text_(svg, "speculative sampling: " + alpha.toFixed(3), legend_x + 10, 256, 12);
	
	let mode = 0, draws = 0, matches = 0;
	let noise_p = [], noise_q = [];
	
	function draw_noise() {
		noise_p = tokens.map(gumbel);
		noise_q = mode == 0 ? noise_p.slice() : tokens.map(gumbel);
	}
	
	function scores() {
		return [p.map(function(v, i) { return Math.log(v) + noise_p[i]; }),
		        q.map(function(v, i) { return Math.log(v) + noise_q[i]; })];
	}
	
	function record() {
		const s = scores();
		draws += 1;
		matches += argmax(s[0]) == argmax(s[1]) ? 1 : 0;
	}
	
	function render() {
		g.selectAll("*").remove();
		const s = scores(), target = argmax(s[0]), draft = argmax(s[1]);
		tokens.forEach(function(token, i) {
			const y = y_start + i * row;
			if (i == target || i == draft) {
				const color = target == draft ? green : (i == draft ? red : orange);
				rect(g, x_start - 4, y + 1, x_end - x_start + 8, row - 4, color, 0.3);
			}
			if (mode == 0) {
				line_(g, x(s[0][i]), y + row / 2 - 1, x(s[1][i]), y + row / 2 - 1, "currentColor", 1, "2,2");
			}
			g.append("circle").attr("cx", x(s[1][i])).attr("cy", y + row / 2 - 1).attr("r", 5)
			   .attr("fill", drafter_color).attr("stroke", "black");
			g.append("circle").attr("cx", x(s[0][i])).attr("cy", y + row / 2 - 1).attr("r", 5)
			   .attr("fill", orange).attr("stroke", "black");
		});
		drafter_legend.text(mode == 0 ? "drafter: log q + g" : "drafter: log q + its own noise");
		shared_text.style("font-weight", mode == 0 ? "bold" : "normal");
		independent_text.style("font-weight", mode == 1 ? "bold" : "normal");
		if (target == draft) {
			result_text.text("draft “" + tokens[draft] + "” accepted");
			output_text.text("both argmaxes agree");
		} else {
			result_text.text("draft “" + tokens[draft] + "” rejected");
			output_text.text("output the target's “" + tokens[target] + "”");
		}
		rate_text.text("match rate: " + matches + " / " + draws + " = " + (matches / draws).toFixed(3));
	}
	
	tabs_(svg, ["shared noise", "independent noise"], 170, 22, function(i) {
		mode = i;
		draws = 0;
		matches = 0;
		draw_noise();
		record();
		render();
	});
	
	text_(svg, "▸ new noise", legend_x, 290, 13).style("cursor", "pointer").on("click", function() {
		draw_noise();
		record();
		render();
	});
	text_(svg, "▸ 1,000 draws", legend_x, 312, 13).style("cursor", "pointer").on("click", function() {
		for (let n = 0; n < 1000; n += 1) {
			draw_noise();
			record();
		}
		render();
	});
}

gumbel_coupling();

</script>

![](.)
*Gumbel-max sampling for the token after "The cat sat on the". Each row is a candidate token: its orange dot is the target's score $\log p + g$ and its blue dot is the drafter's score. The rightmost orange dot is the target's sample, the rightmost blue dot is the draft, and the draft is accepted when the two coincide. With shared noise both dots of a row move together, and the two argmaxes mostly agree; with independent noise they rarely do. Click the titles to switch, and draw new noise to watch the match rate converge.*

The price is a lower acceptance rate. Speculative sampling reaches the best possible agreement, $\alpha = 1 - \operatorname{TV}(p, q)$, because the verifier sees both distributions. Two models that share only the noise agree with probability

$$\Pr\big[\text{match}\big] = \sum_j \frac{1}{\sum_i \max\big(\frac{p(i)}{p(j)}, \frac{q(i)}{q(j)}\big)} \ \geq \ \frac{1 - \operatorname{TV}(p, q)}{1 + \operatorname{TV}(p, q)},$$

and no scheme without communication can guarantee more than the bound on the right in the worst case. Sharing the noise is what makes this work: with independent noise the two samples would match with probability only $\sum_i p(i)\, q(i)$. In the example above that's 0.679 with shared noise and 0.181 with independent noise, against $\alpha = 0.70$ for speculative sampling, and on real LLM distributions the gap between Gumbel coupling and $1 - \operatorname{TV}$ is often just as small. In JAX the whole draft-and-verify step for given distributions along the draft is a few argmaxes over the same noise:

```python
@jit
def gumbel_verify(key, log_p, log_q):
    """log_p: [γ+1, V] target log-probs, log_q: [γ, V] drafter log-probs along the draft.
    The same key, i.e. the same noise, is used by the drafter and the target."""
    g = jax.random.gumbel(key, log_p.shape)
    draft = jnp.argmax(log_q + g[:-1], axis=-1)              # what the drafter proposes
    target = jnp.argmax(log_p + g, axis=-1)                  # what the target samples
    n = jnp.argmin(jnp.append(draft == target[:-1], False))  # first mismatch
    return draft, n, target[n]  # keep draft[:n], then append target[n]
```

### How many tokens do we get?

Let's first assume that each drafted token is accepted independently with the same probability $\alpha$. A round stops at the first rejection, so the number of accepted draft tokens follows a geometric distribution truncated at $\gamma$, and together with the bonus token the expected number of tokens per round is

$$\mathbb{E}[\#\text{tokens}] = \frac{1 - \alpha^{\gamma + 1}}{1 - \alpha}.$$

In practice the drafter gets worse the further it guesses from the verified context, so acceptance depends on the position in the draft. If $\alpha_k$ is the average probability of accepting the $k$-th token given that all previous ones were accepted, then the **acceptance length** $\tau$, the expected number of tokens produced per round, is

$$\tau \approx 1 + \sum_{k=1}^{\gamma} \prod_{i=1}^{k} \alpha_i.$$

This is the number that papers report most often, and drafters compete on it. Now the walltime. Let's measure time in target forward passes and denote by $c$ the cost of one drafter pass relative to one target pass. A drafter that, like an LLM, predicts one token per forward pass runs $\gamma$ times per round, so a round costs $\gamma c$ for drafting plus one verification pass, which costs about as much as a regular decoding step (recall the first figure), and it produces $\tau$ tokens. Plain decoding produces one token per target pass, so the speedup is

$$\text{speedup} = \frac{\tau}{\gamma c + 1}.$$

A real world example: a decent drafter accepts $\alpha = 0.8$ of tokens, drafts $\gamma = 5$ of them, and costs $c = 0.05$ of a target pass. Then

$$\tau = \frac{1 - 0.8^6}{0.2} \approx 3.69, \qquad \text{speedup} = \frac{3.69}{5 \cdot 0.05 + 1} \approx 2.95\times.$$

A fifth of every round goes to drafting, and this share grows with $\gamma$. Both parts of the formula will keep coming back: better drafters raise $\tau$, and cheaper drafting shrinks the $\gamma c$ term.

## The verifier's side

The drafter proposes and the target verifies. Before we look at drafters themselves, let's take a closer look at the verifier, which can do more than check a single chain token by token. It can check several continuations at once, it can accept more of the same draft, and it stops being free once the batch grows.

### Token trees

A chain of guesses dies at the first wrong token. But the target can verify several continuations in the same pass: instead of a chain we can draft a **token tree** whose branches share prefixes, flatten it into one sequence and give the target a **tree attention mask**, in which every node attends only to the context and to its own ancestors ([Miao et al., 2023](https://arxiv.org/pdf/2305.09781)). Positional ids follow the node depth rather than the position in the flattened sequence. Verification then accepts the longest root-to-node path the target agrees with. For sampling, multi-draft versions of the rule keep the output lossless, and SpecTr ([Sun et al., 2023](https://arxiv.org/pdf/2310.15141)) derives the optimal one from optimal transport. The tree adds nodes to the verification pass, but no extra passes, and its expected acceptance length is

$$\tau(\mathcal{T}) = 1 + \sum_{v \in \mathcal{T}} \Pr\big(\text{path root} \to v \text{ accepted}\big).$$

Every node contributes the probability that the target walks all the way down to it, which is the tree version of $\sum_k \prod_{i \leq k} \alpha_i$. Given a budget of nodes, the best tree keeps the nodes with the highest path probabilities, and that's what the tree builders later in this post try to approximate.

<div id="token_tree" class="svg-container" align="center"></div> 

<script>

const tree_marginals = [
	[["mat", 0.55], ["floor", 0.25], ["sofa", 0.10]],
	[["and", 0.45], [",", 0.30], [".", 0.15]],
	[["purred", 0.40], ["slept", 0.30], ["then", 0.15]],
	[["softly", 0.45], ["again", 0.25], ["all", 0.10]]
];

function best_first_tree(marginals, budget) {
	const heap = [{score: marginals[0][0][1], path: [0]}];
	const tree = [];
	while (heap.length > 0 && tree.length < budget) {
		heap.sort(function(a, b) { return b.score - a.score; });
		const node = heap.shift();
		tree.push(node);
		const d = node.path.length;
		if (d < marginals.length) {
			heap.push({score: node.score * marginals[d][0][1], path: node.path.concat([0])});
		}
		const r = node.path[d - 1] + 1;
		if (r < marginals[d - 1].length) {
			heap.push({score: node.score / marginals[d - 1][r - 1][1] * marginals[d - 1][r][1],
			           path: node.path.slice(0, d - 1).concat([r])});
		}
	}
	return tree;
}

function draw_token_tree(svg, budget) {
	svg.selectAll("*").remove();
	const tree = best_first_tree(tree_marginals, budget);
	const key = function(path) { return path.join(","); };
	const nodes = {};
	tree.forEach(function(node, i) {
		nodes[key(node.path)] = {index: i, path: node.path, score: node.score, children: []};
	});
	const root = {path: [], children: []};
	tree.forEach(function(node) {
		const parent = node.path.length == 1 ? root : nodes[key(node.path.slice(0, -1))];
		parent.children.push(nodes[key(node.path)]);
	});
	
	const x_root = 10, dx = 92, y_top = 30, dy = 28, w = 72;
	let rows = 0;
	function place(node, row) {
		node.row = row;
		node.children.sort(function(a, b) { return a.path[a.path.length - 1] - b.path[b.path.length - 1]; });
		node.children.forEach(function(child, i) { place(child, i == 0 ? row : ++rows); });
	}
	place(root, 0);
	
	function draw_edges(node) {
		const x0 = x_root + node.path.length * dx;
		node.children.forEach(function(child) {
			link_h(svg, x0 + w, y_top + node.row * dy + 11, x0 + dx, y_top + child.row * dy + 11);
			draw_edges(child);
		});
	}
	draw_edges(root);
	
	token_box(svg, x_root, y_top, w, "the", context_color);
	let total = 0;
	tree.forEach(function(item) {
		const node = nodes[key(item.path)], d = node.path.length;
		const token = tree_marginals[d - 1][node.path[d - 1]][0];
		const x = x_root + d * dx, y = y_top + node.row * dy;
		rect(svg, x, y, w, 22, draft_token_color);
		text_(svg, token, x + 5, y + 15, 12, "black");
		text_(svg, node.score.toFixed(2).substring(1), x + w - 4, y + 15, 9, "black").attr("text-anchor", "end");
		total += node.score;
	});
	const n = tree.length, cell = 14, m_x = 500, m_y = y_top + 20;
	text_(svg, "τ ≈ 1 + Σ q(path) = " + (1 + total).toFixed(2), m_x - 60, m_y + n * cell + 42, 13);
	text_(svg, "ctx", m_x - 2, m_y - 8, 10);
	for (let i = 0; i < n; i += 1) {
		const node = tree[i], d = node.path.length;
		text_(svg, tree_marginals[d - 1][node.path[d - 1]][0], m_x - 6, m_y + i * cell + 10, 10)
		   .attr("text-anchor", "end");
		rect(svg, m_x, m_y + i * cell, cell - 2, cell - 2, context_color);
		for (let j = 0; j < n; j += 1) {
			const other = tree[j].path;
			const ancestor = other.length <= node.path.length && key(node.path.slice(0, other.length)) == key(other);
			rect(svg, m_x + (j + 1) * cell, m_y + i * cell, cell - 2, cell - 2, ancestor ? orange : 'white');
		}
	}
	text_(svg, "tree attention mask", m_x, m_y + n * cell + 20, 11);
}

function token_tree() {
	const svg = d3.select("#token_tree")
				  .append("svg")
				  .attr("viewBox", "0 0 700 200")
				  .style("width", "100%")
				  .style("max-width", "700px")
				  .style("height", "auto");
	draw_token_tree(svg.append("g"), 7);
}

token_tree();

</script>

![](.)
*A token tree of 7 drafted nodes continuing "...sat on the" (left) and its attention mask (right): each row is a node, colored cells mark what it attends to. The numbers in the nodes are the drafter's path probabilities; if they were calibrated, their sum plus one would be the expected acceptance length.*

### Block verification

Speculative sampling checks a chain one token at a time and stops at the first rejection. That's a myopic decision. Suppose the drafter overestimates $x_{t+1}$ but underestimates $x_{t+2}$: the pair as a whole may be as likely under the target as under the drafter, yet token-wise verification often throws both away. **Block verification** ([Sun et al., 2024](https://arxiv.org/pdf/2403.10444)) decides on the whole draft jointly. Let $p_k$ and $q_k$ be the target's and the drafter's distributions at position $k$ of the draft. The verifier keeps a running weight of how much of each prefix the target still covers,

$$w_k = \min\Big(1,\ w_{k-1}\frac{p_k(x_{t+k})}{q_k(x_{t+k})}\Big), \qquad w_0 = 1,$$

so that a ratio below one on one token can be compensated by ratios above one later on (note that $\min(1, ab) \geq \min(1, a) \min(1, b)$). Each prefix survives with probability

$$\Pr\big(x_{t+1:t+k} \text{ survives}\big) = \frac{\sum_x \big(w_k\, p_{k+1}(x) - q_{k+1}(x)\big)_+}{\sum_x \big(w_k\, p_{k+1}(x) - q_{k+1}(x)\big)_+ + 1 - w_k},$$

the longest surviving prefix, of length $n$, is accepted, and the next token is sampled from the normalized residual $\big(w_n\, p_{n+1} - q_{n+1}\big)\_+$. With $q_{\gamma+1} \equiv 0$ the bonus token falls out for free once again, since then the whole draft survives with probability $w_\gamma$. Compared to `verify`, only the choice of the prefix length and the residual change:

```python
@jit
def block_verify(key, p, q, draft):
    """Same inputs as verify(), but the prefix is accepted jointly."""
    gamma = draft.shape[0]
    key_u, key_r = jax.random.split(key)
    ratio = p[jnp.arange(gamma), draft] / q[jnp.arange(gamma), draft]

    def step(w, ratio_k):
        w = jnp.minimum(w * ratio_k, 1.0)
        return w, w

    _, w = jax.lax.scan(step, 1.0, ratio)
    w = jnp.concatenate([jnp.ones(1), w])                          # w_0, ..., w_γ
    mass = jnp.maximum(w[:-1, None] * p[:-1] - q, 0.0).sum(-1)     # residual mass
    mass = jnp.append(mass, w[-1])                                  # zero draft after γ
    survival = jnp.where(w < 1.0, mass / (mass + 1.0 - w), 1.0)
    survived = jax.random.uniform(key_u, (gamma,)) <= survival[1:]
    n = jnp.max(jnp.where(survived, jnp.arange(1, gamma + 1), 0))  # longest prefix
    q_n = jnp.where(n < gamma, q[jnp.minimum(n, gamma - 1)], 0.0)
    residual = jnp.maximum(w[n] * p[n] - q_n, 0.0)
    token = jax.random.categorical(key_r, jnp.log(residual))
    return n, token
```

Sun et al. prove that, for a given draft, block verification maximizes the expected number of accepted tokens among all lossless verification rules, so it never does worse than the token-wise rule. Take a toy example where at every position $p(a) = \frac{1}{3}$, $p(b) = \frac{2}{3}$ and $q(a) = \frac{2}{3}$, $q(b) = \frac{1}{3}$, with $\gamma = 2$. Token-wise verification accepts $\frac{10}{9}$ tokens per round on average, block verification $\frac{11}{9}$. With PaLM-2 models the gain was about 8% more tokens per round and 5–8% in walltime, with the same draft, the same target pass and a few extra vector operations. It works with any drafter, and at the end of the post we'll see how to train a drafter for it. Traversal Verification ([Weng et al., 2025](https://arxiv.org/pdf/2505.12398)) brings the same idea to token trees, checking whole root-to-leaf sequences from the leaves up instead of discarding a subtree as soon as its root is rejected.

### Verification bottlenecks

So far we've assumed that verification costs as much as one decoding step. That holds only while the verification pass stays memory-bound, i.e. while the total number of tokens it scores across the batch stays below the `ops:byte` ratio from the first figure. With a batch of $B$ requests and a tree of $N$ drafted nodes per request, the pass scores $B(N + 1)$ tokens. Denote by $T(n)$ the time of a forward pass over $n$ tokens: the curve from the first figure, flat while the pass is memory-bound and linear once it becomes compute-bound. Then the speedup over plain decoding at the same batch size is

$$\text{speedup}(N, B) = \frac{\tau(N) \cdot T(B)}{T\big(B(N + 1)\big) + T_{\text{draft}}},$$

where $T_{\text{draft}}$ is the time spent drafting.

The numerator grows slowly: even with an optimal tree, the acceptance length grows only roughly logarithmically with the tree size, as observed in Sequoia ([Chen et al., 2024](https://arxiv.org/pdf/2402.12374)). The denominator grows linearly as soon as $B(N+1)$ passes the knee of that curve. So the best tree shrinks as the batch grows, roughly as $N^\ast \approx \frac{\text{ops:byte}}{B}$:

<div id="tree_cost" class="svg-container" align="center"></div> 

<script>

function tree_tau(alpha, n_max) {
	const ranks = 12, depth = 16;
	const pi = d3.range(ranks).map(function(r) { return alpha * Math.pow(1 - alpha, r); });
	const heap = [{score: pi[0], path: [0]}];
	const taus = [1];
	let total = 1;
	while (heap.length > 0 && taus.length <= n_max) {
		heap.sort(function(a, b) { return b.score - a.score; });
		const node = heap.shift();
		total += node.score;
		taus.push(total);
		const d = node.path.length;
		if (d < depth) {
			heap.push({score: node.score * pi[0], path: node.path.concat([0])});
		}
		const r = node.path[d - 1] + 1;
		if (r < ranks) {
			heap.push({score: node.score / pi[r - 1] * pi[r], path: node.path.slice(0, d - 1).concat([r])});
		}
	}
	return taus;
}

function tree_cost() {
	const svg = d3.select("#tree_cost")
				  .append("svg")
				  .attr("viewBox", "0 0 700 330")
				  .style("width", "100%")
				  .style("max-width", "700px")
				  .style("height", "auto");
	
	const x_start = 70, x_end = 480, y_start = 25, y_end = 240, n_max = 256, c = 0.05;
	const step_time = function(n) { return Math.max(16e9 / 3.35e12, 2 * 8e9 * n / 989e12); };
	const batches = [[1, green], [8, blue], [32, colors[1]], [128, red]];
	const x = d3.scaleLog().base(2).domain([1, n_max]).range([x_start, x_end]);
	const y = d3.scaleLinear().range([y_end, y_start]);
	
	arvo_axis(svg.append("g")
	   .attr("transform", "translate(0," + y_end + ")")
	   .call(d3.axisBottom(x).tickValues([1, 2, 4, 8, 16, 32, 64, 128, 256]).tickFormat(d3.format("d"))));
	const y_axis = svg.append("g").attr("transform", "translate(" + x_start + ", 0)");
	text_(svg, "speedup", x_start - 25, y_start - 10, 11);
	text_(svg, "drafted nodes per request, N", x_start + 120, y_end + 35, 11);
	const baseline = svg.append("line").attr("x1", x_start).attr("x2", x_end)
	   .attr("stroke", "currentColor").style("stroke-dasharray", "4,3");
	const baseline_text = text_(svg, "no speculation", x_start + 6, 0, 10);
	const curves = batches.map(function(b) {
		return svg.append("path").attr("fill", "none").attr("stroke", b[1]).attr("stroke-width", 2);
	});
	const dots = batches.map(function(b) {
		return svg.append("circle").attr("r", 5).attr("fill", b[1]).attr("stroke", "black");
	});
	const legend = batches.map(function(b, i) {
		rect(svg, 505, 52 + i * 30, 12, 12, b[1]);
		return text_(svg, "", 523, 63 + i * 30, 12);
	});
	text_(svg, "best tree per batch size B", 505, 35, 12);
	
	function update(alpha) {
		const taus = tree_tau(alpha, n_max);
		const all_points = batches.map(function(b) {
			return d3.range(1, n_max + 1).map(function(n) {
				return {n: n, s: taus[n] * step_time(b[0]) / (step_time(b[0] * (n + 1)) + c * step_time(b[0]))};
			});
		});
		const top = d3.max(all_points, function(points) { return d3.max(points, function(d) { return d.s; }); });
		y.domain([0, Math.max(4, Math.ceil(top * 1.1))]);
		arvo_axis(y_axis.call(d3.axisLeft(y).ticks(6)));
		baseline.attr("y1", y(1)).attr("y2", y(1));
		baseline_text.attr("y", y(1) + 13);
		batches.forEach(function(b, i) {
			const batch = b[0], points = all_points[i];
			curves[i].attr("d", d3.line()
			   .curve(d3.curveMonotoneX)
			   .x(function(d) { return x(d.n); })
			   .y(function(d) { return y(d.s); })(points));
			const best = points.reduce(function(a, p) { return p.s > a.s ? p : a; });
			dots[i].attr("cx", x(best.n)).attr("cy", y(best.s));
			legend[i].text("B = " + batch + ": N = " + best.n + ", " + best.s.toFixed(2) + "×");
		});
	}
	update(0.8);
	
	const alpha_x = d3.scaleLinear().domain([0.5, 0.9]).range([0, 330]).clamp(true);
	createSlider(svg, update, alpha_x, x_start + 60, 300, "α", "currentColor", 0.8, round2, 4);
}

tree_cost();

</script>

![](.)
*Idealized speedup of tree speculation for an 8B model on H100 versus the number of drafted nodes per request, for four batch sizes. The tree is built best-first from a drafter whose top candidate matches the target with probability $\alpha$ and whose $j$-th candidate matches with probability $\alpha(1-\alpha)^j$; verification time follows the curve of the first figure, and drafting costs $c = 0.05$ of a decoding step. Dots mark the best tree for each batch size.*

With $\alpha = 0.8$, a single request keeps gaining from bigger trees right up to the knee of the curve. Eight concurrent requests are best served by trees of about 36 nodes, 32 requests by about 8, and at 128 requests the best "tree" is a chain of two tokens. A 64-node tree there makes decoding almost four times slower than not speculating at all. That's why Sequoia chooses the tree size per hardware, 64 to 128 nodes on an A100 but 768 when the weights are offloaded and streamed to an L40, where each verification pass is so expensive that its flat part becomes enormous. In serving it's why TurboSpec ([Liu et al., 2024](https://arxiv.org/pdf/2406.14066)) adapts the amount of speculation to the load by tracking goodput, the rate of tokens that actually get accepted, since naive speculation can make a busy server slower.

Beyond FLOPs, trees are harder for the serving engine than chains:

- **Attention kernels.** The ancestor-only mask of a tree is neither causal nor a simple window, so it doesn't fit the fast paths of standard attention kernels such as FlashAttention-2, and tree verification often falls back to slower generic attention.
- **KV cache bookkeeping.** The verification pass writes keys and values for every node, but only the accepted path survives, so its entries have to be compacted into consecutive positions or remapped in a [paged cache](https://astralord.github.io/posts/transformer-inference-optimization-toolset/#pagedattention--vllm). A rejected chain is simply truncated.
- **Static shapes.** Engines capture decoding steps as CUDA graphs, pre-recorded sequences of kernel launches, and overlap CPU scheduling with GPU work, and both want the same shapes at every step. SGLang, for example, requires a chain (`--speculative-eagle-topk 1`) when its overlap scheduler is on, and that's the default.

So trees pay off most where latency matters and the batch is small, while high-throughput serving tends to use chains with a draft length that adapts to the load.

## Anatomy of a drafter

Drafters look very different on the surface, but every drafter in this post is one conditional distribution

$$q_\phi\big(x_{t+k} \mid x_{\le t},\; \mathbf{H},\; x_{t+1:t+k-1}\big), \qquad k = 1, \dots, \gamma,$$

where $x_{\leq t}$ is the verified context, $\mathbf{H}$ is whatever the drafter reads from the target (from its last hidden state to several of its layers or even its KV cache), and $x_{t+1:t+k-1}$ are the drafter's own guesses earlier in this round. The most consequential design decision is how a draft token depends on the guesses before it, because that decides how many sequential passes a round takes. Written as factorizations of the whole draft, there are three basic options:

$$
\begin{aligned}
\text{independent:} \quad & q(x_{t+1:t+\gamma}) = \textstyle\prod_{k} q_k(x_{t+k} \mid x_{\le t}, \mathbf{H}) \\
\text{autoregressive:} \quad & q(x_{t+1:t+\gamma}) = \textstyle\prod_{k} q(x_{t+k} \mid x_{\le t}, \mathbf{H}, x_{t+1:t+k-1}) \\
\text{parallel + causal head:} \quad & q(x_{t+1:t+\gamma}) = \textstyle\prod_{k} \operatorname{head}\big(\mathbf{z}_k, x_{t+1:t+k-1}\big)(x_{t+k})
\end{aligned}
$$

The independent line costs a single drafter pass, but no position knows what the others guessed. The autoregressive line is how an LLM generates text: full dependency, paid for with $\gamma$ sequential passes through the whole drafter. The last line is still autoregressive, but it splits the work: a parallel network computes states $\mathbf{z}\_{1:\gamma} = f_\phi(x_{\le t}, \mathbf{H})$ for all positions in one pass, and only a tiny head runs token by token, looking at the previous guess or at all of them. The next sections follow these lines family by family.

## Autoregressive drafters

The original speculative decoding papers drafted with a separate small LLM from the same family as the target, for example a 4B model drafting for a 70B Chinchilla in Chen et al. It's easy to deploy, since any smaller model with the same tokenizer will do, but it shares nothing with the target: it recomputes from scratch representations the target has already built, and it keeps its own full KV cache. The drafters in this section keep the autoregressive structure but read the target instead.

### EAGLE

**EAGLE** ([Li et al., 2024](https://arxiv.org/pdf/2401.15077)) drafts autoregressively at the level of hidden states, which it calls features, rather than tokens. Its input combines two sequences: the target's last hidden states $\mathbf{h}\_i$, right before the LM head, and the embeddings $e(x_{i+1})$ of the tokens one step ahead. Each hidden state is concatenated with the embedding of the token that was actually sampled from it, an FC layer reduces the pair back to the hidden size, and a single transformer layer predicts the next hidden state:

$$\mathbf{a}_{t+1} = \operatorname{Layer}_\phi\Big(\operatorname{FC}\big[\mathbf{h}_i;\, e(x_{i+1})\big]_{i \le t}\Big).$$

The target's LM head turns $\mathbf{a}\_{t+1}$ into the drafter's distribution $q := q(\cdot \mid \mathbf{a}\_{t+1})$ over the token $x_{t+2}$, and the drafter is trained both to match the target's next hidden state and to predict that token:

$$\mathcal{L} = \operatorname{SmoothL}_1\big(\mathbf{a}_{t+1}, \mathbf{h}_{t+1}\big) + w_{\text{cls}} \cdot \mathcal{L}_{\operatorname{CE}}\big(q,\, x_{t+2}\big),$$

with $w_{\text{cls}} = 0.1$. The shifted tokens matter: a hidden state $\mathbf{h}\_t$ alone doesn't tell the drafter which token was actually sampled from it. At later draft steps the target's hidden states aren't available yet, so the drafter feeds its own predictions $\mathbf{a}$ back in. **EAGLE-2** ([Li et al., 2024b](https://arxiv.org/pdf/2406.16858)) kept the drafter and replaced the static token tree with a dynamic one, grown wherever the drafter's confidence is high, since that confidence turned out to be well calibrated with acceptance.

### EAGLE-3

**EAGLE-3** ([Li et al., 2025](https://arxiv.org/pdf/2503.01840)) made two changes. First, instead of the last hidden state it fuses low-, mid- and high-level hidden states of the target, $\mathbf{H} = \mathbf{W}\_{\text{fuse}}\,\big[\mathbf{h}^{(l_1)};\, \mathbf{h}^{(l_2)};\, \mathbf{h}^{(l_3)}\big]$. Second, it dropped the hidden-state regression, which tied the drafter to imitating the target's hidden states, and trained on tokens only. Without it, the drafter's hidden states $\mathbf{a}$ at later steps no longer have to look like the target's, so they must be part of training. The **training-time test** unrolls the drafter for several steps: the fused target states $\mathbf{H}$ are available only for the verified context, and at every later position the drafter consumes its own hidden states from the previous step, exactly as at inference. The tokens themselves come from the training text:

$$\mathcal{L}_{\text{TTT}} = \sum_{k=1}^{\gamma} \mathcal{L}_{\operatorname{CE}}\Big(q_\phi\big(\cdot \mid x_{< t+k},\ \mathbf{H}_{\le t},\ \mathbf{a}_{t+1:t+k-1}\big),\; x_{t+k}\Big),$$

where $\mathbf{a}\_{t+1:t+k-1}$ are the drafter's own hidden states from the earlier steps. With these changes the drafter finally scaled with training data, and EAGLE-3 reported speedups of up to 6.5×. Its weak spot is that it's still sequential: a round costs $\gamma$ drafter passes before the tree is verified, and when verification isn't free, this overhead can eat the whole gain. In [vLLM's throughput tests on AMD GPUs](https://vllm.ai/blog/2026-08-23-speculative-decoding-amd-gpus), EAGLE-3 on Qwen3-8B with MATH500 stayed below plain decoding for every draft length from 1 to 7 tokens, from 0.44× to 0.88×.

### Multi-token prediction

Another way to get a drafter is to train it together with the target. **Multi-token prediction (MTP)** modules were popularized by DeepSeek-V3 ([DeepSeek-AI, 2024](https://arxiv.org/pdf/2412.19437)) as an auxiliary pretraining objective that densifies the training signal. The model gets $D$ extra modules applied in sequence. The $k$-th module takes the hidden state $\mathbf{a}^{(k-1)}\_i$ of position $i$ from the previous depth, starting from the target's last hidden state $\mathbf{a}^{(0)}\_i = \mathbf{h}\_i$, normalizes it, concatenates it with the normalized embedding $e(x_{i+k})$ of the token $k$ steps ahead, projects the pair back to the hidden size with its own FC layer, and passes the result through its own transformer layer:

$$\mathbf{a}^{(k)}_t = \operatorname{Layer}_k\Big(\operatorname{FC}_k\big[\operatorname{RMSNorm}(\mathbf{a}^{(k-1)}_i);\, \operatorname{RMSNorm}(e(x_{i+k}))\big]_{i \le t}\Big).$$

The shared LM head turns $\mathbf{a}^{(k)}\_t$ into the module's distribution $q_k := q(\cdot \mid \mathbf{a}^{(k)}\_t)$ over token $x_{t+k+1}$, so the $k$-th module learns to look $k+1$ tokens ahead. During pretraining the cross-entropies of all depths are averaged and added to the target's usual next-token loss with a small weight:

$$\mathcal{L} = \mathcal{L}_{\operatorname{CE}}\big(p,\, x_{t+1}\big) + \frac{w_{\text{MTP}}}{D} \sum_{k=1}^{D} \mathcal{L}_{\operatorname{CE}}\big(q_k,\, x_{t+k+1}\big).$$

At inference the modules are reused as a sequential drafter that comes with the model for free. DeepSeek-V4 serves with MTP-1, a single module drafting one token per round, and Qwen3.5 has native MTP layers. Kimi-K3 pre-trains one MTP layer and then fine-tunes it into an EAGLE-3-style drafter that fuses the outputs of its 1st, 4th and final layers ([Kimi Team, 2026](https://github.com/MoonshotAI/Kimi-K3/blob/main/k3_tech_report.pdf)), which is where the two families meet. We'll get back to Kimi-K3's training objective at the end of the post.

Every drafter in this section pays $\gamma$ drafter passes per round. The next section removes that cost.

## Parallel drafters

Every drafter in the previous section runs once per drafted token, and in the example from the first section that already cost a fifth of every round. A **parallel drafter** proposes all $\gamma$ tokens in a single pass. With $n_{\text{draft}}$ drafter passes per round the speedup becomes

$$\text{speedup} = \frac{\tau}{n_{\text{draft}} \cdot c + 1}, \qquad n_{\text{draft}} = \begin{cases} \gamma & \text{autoregressive drafter} \\ 1 & \text{parallel drafter} \end{cases}$$

and with the same numbers as before a parallel drafter reaches $\frac{3.69}{0.05 + 1} \approx 3.51\times$: almost 20% faster with the same acceptance rate and draft length. The gap grows with $\gamma$. An autoregressive drafter has an optimal draft length, beyond which extra guesses cost more than they bring, while for a parallel drafter longer drafts are nearly free, at least while verification stays memory-bound:

<div id="speedup_sliders" class="svg-container" align="center"></div> 

<script>

function speedup_sliders() {
	const svg = d3.select("#speedup_sliders")
				  .append("svg")
				  .attr("viewBox", "0 0 700 410")
				  .style("width", "100%")
				  .style("max-width", "700px")
				  .style("height", "auto");
	
	let alpha = 0.8, c = 0.05, gamma = 5;
	const tau = function(a, g) { return (1 - Math.pow(a, g + 1)) / (1 - a); };
	const sequential = function(a, g) { return tau(a, g) / (g * c + 1); };
	const block = function(a, g) { return tau(a, g) / (c + 1); };
	
	const x_start = 80, x_end = 560, y_start = 35, y_end = 240;
	const x = d3.scaleLinear().domain([1, 16]).range([x_start, x_end]);
	const y = d3.scaleLinear().domain([0, 12]).range([y_end, y_start]);
	
	arvo_axis(svg.append("g")
	   .attr("transform", "translate(0," + y_end + ")")
	   .call(d3.axisBottom(x).ticks(8)));
	arvo_axis(svg.append("g")
	   .attr("transform", "translate(" + x_start + ", 0)")
	   .call(d3.axisLeft(y).ticks(6)));
	text_(svg, "speedup", x_start - 25, y_start - 15, 11);
	text_(svg, "γ", x_end + 12, y_end + 4, 13);
	
	const curve = function() {
		return d3.line()
		  .curve(d3.curveMonotoneX)
		  .x(function(d) { return x(d.x); })
		  .y(function(d) { return y(Math.min(d.y, 12)); });
	};
	const tau_path = svg.append("path").attr("fill", "none").attr("stroke", "currentColor")
	   .attr("stroke-width", 1).style('stroke-dasharray', '2,3');
	const block_path = svg.append("path").attr("fill", "none").attr("stroke", "currentColor")
	   .attr("stroke-width", 2);
	const seq_path = svg.append("path").attr("fill", "none").attr("stroke", "currentColor")
	   .attr("stroke-width", 2).style('stroke-dasharray', '5,2');
	const tau_label = text_(svg, "τ", 0, 0, 12);
	const block_label = text_(svg, "parallel", 0, 0, 12);
	const seq_label = text_(svg, "autoregressive", 0, 0, 12);
	
	const block_dot = svg.append("circle").attr("r", 5).attr("fill", orange).attr("stroke", "black");
	const seq_dot = svg.append("circle").attr("r", 5).attr("fill", drafter_color).attr("stroke", "black");
	const tau_text = text_(svg, "", x_start + 20, y_start + 15, 13);
	const block_text = text_(svg, "", x_start + 20, y_start + 35, 13);
	const seq_text = text_(svg, "", x_start + 20, y_start + 55, 13);
	
	function update() {
		const gammas = d3.range(1, 17);
		tau_path.attr("d", curve()(gammas.map(function(g) { return {x: g, y: tau(alpha, g)}; })));
		block_path.attr("d", curve()(gammas.map(function(g) { return {x: g, y: block(alpha, g)}; })));
		seq_path.attr("d", curve()(gammas.map(function(g) { return {x: g, y: sequential(alpha, g)}; })));
		tau_label.attr("x", x(16) + 8).attr("y", y(Math.min(tau(alpha, 16), 12)) - 2);
		block_label.attr("x", x(16) + 8).attr("y", y(Math.min(block(alpha, 16), 12)) + 12);
		seq_label.attr("x", x(16) + 8).attr("y", y(sequential(alpha, 16)) + 4);
		block_dot.attr("cx", x(gamma)).attr("cy", y(block(alpha, gamma)));
		seq_dot.attr("cx", x(gamma)).attr("cy", y(sequential(alpha, gamma)));
		tau_text.text("τ = " + tau(alpha, gamma).toFixed(2));
		block_text.text("parallel: " + block(alpha, gamma).toFixed(2) + "×");
		seq_text.text("autoregressive: " + sequential(alpha, gamma).toFixed(2) + "×");
	}
	update();
	
	const alpha_x = d3.scaleLinear().domain([0.5, 0.95]).range([0, 360]).clamp(true);
	const c_x = d3.scaleLinear().domain([0, 0.2]).range([0, 360]).clamp(true);
	const gamma_x = d3.scaleLinear().domain([1, 16]).range([0, 360]).clamp(true);
	createSlider(svg, function(v) { alpha = v; update(); }, alpha_x, x_start + 60, 290, "α", "currentColor", alpha, round2, 5);
	createSlider(svg, function(v) { c = v; update(); }, c_x, x_start + 60, 335, "c", "currentColor", c, round2, 4);
	createSlider(svg, function(v) { gamma = v; update(); }, gamma_x, x_start + 60, 380, "γ", "currentColor", gamma, roundN, 15);
}

speedup_sliders();

</script>

![](.)
*Acceptance length $\tau$ (dotted) and walltime speedup of an autoregressive drafter (dashed) and a parallel drafter (solid) versus the draft length $\gamma$. The defaults reproduce the example from the first section: $\alpha = 0.8$, $c = 0.05$, $\gamma = 5$.*

The idea of guessing several future tokens at once is as old as speculative decoding itself[^JAC], but for a long time the guesses were too weak to compete with autoregressive drafters.

### Medusa

**Medusa** ([Cai et al., 2024](https://arxiv.org/pdf/2401.10774)) puts $\gamma$ cheap heads on top of the target's last hidden state $\mathbf{h}\_t$, and the $k$-th head guesses the token $k$ steps ahead:

$$q_k(\cdot \mid \mathbf{h}_t) = \operatorname{softmax}\Big(\mathbf{W}_2^{(k)}\big(\operatorname{SiLU}(\mathbf{W}_1^{(k)} \mathbf{h}_t) + \mathbf{h}_t\big)\Big), \quad k = 1, \dots, \gamma,$$

where $\mathbf{W}\_2^{(k)}$ is initialized from the target's LM head. All heads run in parallel on top of the target's own forward pass, so drafting is almost free, and the candidates from all heads are combined into a static token tree. The catch is the first line of our factorization: head $k$ never sees what heads $1, \dots, k-1$ guessed, and a single hidden state is little context to guess far ahead. If the first head hesitates between "New" and "Los", the second one can only spread its mass between "York" and "Angeles", and $\alpha_k$ decays quickly with $k$[^HYD].

### P-EAGLE

The most direct way to get a strong parallel drafter is to take a good autoregressive one and stop feeding it its own guesses. **P-EAGLE** ([Hui et al., 2026](https://arxiv.org/pdf/2602.01469)) does this to EAGLE-3. Every position after the first lacks both the previous draft token and the hidden state that would come with it, so P-EAGLE fills them with two learned vectors: a mask token embedding for the unknown token and a shared hidden state for the missing one.

The harder part is training. Each position of a training sequence of length $L$ now carries $\gamma$ prediction depths, so attention over $L\gamma$ positions needs $\mathcal{O}\big((L\gamma)^2\big)$ memory, which gets prohibitive for the long outputs of reasoning models. P-EAGLE precomputes the depth-aware attention mask once and slices it per example, and partitions long sequences into segments with gradient accumulation inside a sequence. The acceptance length stays on par with EAGLE-3, but with one drafter pass instead of five the end-to-end speedup over EAGLE-3 in vLLM is 1.10–1.36× on gpt-oss-120B, gpt-oss-20B and Qwen3-Coder-30B. PARD ([An et al., 2025](https://arxiv.org/pdf/2504.18583)) applies the same trick to a separate small LM, which learns to fill several placeholder positions per pass, so that a single drafter can serve a whole family of targets.

### DFlash

Placeholder positions predicted in one pass are what masked diffusion language models do: they are trained like masked language models to fill in masked positions all at once, and at generation time they refine their guesses over several such steps. Early work drafted with whole discrete diffusion models of this kind ([Christopher et al., 2024](https://arxiv.org/pdf/2408.05636)). **DFlash** ([Chen et al., 2026](https://arxiv.org/pdf/2602.06036)) calls itself a **block diffusion** drafter, but it's worth being careful with the name. It's trained like a masked diffusion model, yet at inference it takes a single step: the input block is the last token the target produced in the previous round followed by mask tokens, and one forward pass predicts all masked positions, with no refinement afterwards. In effect it's a masked block predictor, and iterative refinement will only come back later, in DEdit.

The block enters the drafter through the target's embedding table, which DFlash shares with the target and keeps frozen together with the LM head, so the input token and the mask tokens are embedded just like ordinary tokens. Inside the block, positions attend to each other bidirectionally. To predict well from masks, the drafter needs a strong grip on the target's context. DFlash takes hidden states from five target layers, spread uniformly between the second and the third-to-last, fuses them into $\mathbf{H}$ as EAGLE-3 does, with an extra RMSNorm, and injects them as extra keys and values into *every* draft layer:

$$\mathbf{z}^{(j+1)} = \operatorname{Attn}\Big(\mathbf{Q} = \mathbf{W}_Q^{(j)} \mathbf{z}^{(j)},\; \mathbf{K} = \big[\mathbf{W}_K^{(j)} \mathbf{H}_{\le t};\, \mathbf{W}_K^{(j)} \mathbf{z}^{(j)}\big],\; \mathbf{V} = \big[\mathbf{W}_V^{(j)} \mathbf{H}_{\le t};\, \mathbf{W}_V^{(j)} \mathbf{z}^{(j)}\big]\Big),$$

where $\mathbf{z}^{(j)}$ are the hidden states of the draft block at layer $j$, starting from the embeddings $\mathbf{z}^{(0)}$ of that token and the masks. EAGLE-3 sees the fused states only at its input, so their influence fades as they pass through the drafter. In DFlash every layer attends to the target's context directly, which lets the drafter be deeper (5 layers) and still cheap, because it runs only once. Training samples random positions in target-generated text as **anchors**, the last verified token of an imagined round, masks the block after each anchor, and applies cross-entropy with weights decaying exponentially along the block.

The most telling numbers come from [LMSYS](https://www.lmsys.org/blog/2026-06-15-next-generation-speculative-decoding-dflash-v2/): on Qwen3-4B with GSM8K, EAGLE-3 and DFlash both reach $\tau = 4.2$, yet the speedups are 2.1× and 3.3×. The acceptance length didn't move at all; $n_{\text{draft}}$ did, exactly the gap between the two curves at the start of this section.

Adoption followed quickly. The DFlash drafter for Qwen3.5-397B-A17B delivers higher throughput than the model's native MTP in every setting LMSYS benchmarked, with 1.5× the throughput of MTP at concurrency 1.

#### DDTree

A parallel drafter gives us something a sequential one doesn't: the full distribution at every position of the block, all at once. **DDTree** ([Ringel and Romano, 2026](https://arxiv.org/pdf/2604.12989)) turns these per-position marginals into a token tree without any training. It scores a candidate path $\rho$ by its log-probability under the drafter, $\log q(\rho) = \sum_i \log q_i(\rho_i)$. Because the marginals are independent, path scores simply add up, and the classic best-first enumeration finds the top paths for a given node budget. Each popped path pushes two successors, its best child one level deeper and its next-best sibling:

```python
import heapq

def build_tree(log_q, budget):
    """log_q: [γ, V] per-position log-probs from the parallel drafter.
    Returns `budget` paths, each a tuple of per-depth candidate ranks."""
    log_q = np.asarray(log_q)
    top = np.argsort(-log_q, axis=-1)[:, :budget]   # candidate tokens per depth
    lq = np.take_along_axis(log_q, top, axis=-1)    # and their log-probs
    heap = [(-lq[0, 0], (0,))]                      # (-score, ranks along path)
    tree = []
    while heap and len(tree) < budget:
        neg, path = heapq.heappop(heap)
        tree.append(path)
        d = len(path)
        if d < lq.shape[0]:                         # child: best token one level deeper
            heapq.heappush(heap, (neg - lq[d, 0], path + (0,)))
        r = path[-1] + 1
        if r < lq.shape[1]:                         # sibling: next-best token, same depth
            heapq.heappush(heap, (neg + lq[d - 1, r - 1] - lq[d - 1, r], path[:-1] + (r,)))
    return tree
```

Scores only decrease along both kinds of edges, so paths leave the heap in order and every parent leaves before its children. The token tree from the verifier's side was built exactly this way:

<div id="token_tree_budget" class="svg-container" align="center"></div> 

<script>

function token_tree_budget() {
	const svg = d3.select("#token_tree_budget")
				  .append("svg")
				  .attr("viewBox", "0 0 700 330")
				  .style("width", "100%")
				  .style("max-width", "700px")
				  .style("height", "auto");
	const g = svg.append("g");
	draw_token_tree(g, 7);
	
	const budget_x = d3.scaleLinear()
	    .domain([1, 12])
	    .range([0, 330])
	    .clamp(true);
	createSlider(svg, function(budget) { draw_token_tree(g, budget); }, budget_x, 130, 300, "budget", "currentColor", 7, roundN, 11, 60);
}

token_tree_budget();

</script>

![](.)
*Best-first token tree built from the per-position marginals of a parallel drafter, as in DDTree, and its attention mask. Move the slider to change the node budget.*

The tree is verified in one target pass with an ancestor-only attention mask, which FlashAttention-2 doesn't support, so DDTree falls back to PyTorch's generic scaled dot-product attention. On Qwen3-8B at temperature 0 it lifts the acceptance length on MATH-500 from DFlash's 7.79 to 10.73 tokens per round, and the speedup from 5.56× to 7.52×.

## Restoring dependency in the block

DFlash has the same catch as Medusa, one level up. Every position of the block is a marginal, so nothing ties the token at position $t+k$ to the one before it. When the target is unsure between two continuations, independent sampling happily mixes them:

<div id="marginals_vs_causal" class="svg-container" align="center"></div> 

<script>

function marginals_vs_causal() {
	const svg = d3.select("#marginals_vs_causal")
				  .append("svg")
				  .attr("viewBox", "0 0 700 290")
				  .style("width", "100%")
				  .style("max-width", "700px")
				  .style("height", "auto");

	const g = svg.append("g");
	const first = ["New", "Los"], second = ["York", "Angeles"], tail = ["last", "week"];
	const samples = [[[0, 0], [1, 0], [0, 1], [1, 1]],
	                 [[0, 0], [1, 1], [1, 1], [0, 0]]];
	const joint = [[[0.25, 0.25], [0.25, 0.25]],
	               [[0.5, 0.0], [0.0, 0.5]]];
	const grid_x = 115, grid_y = 112, cw = 76, ch = 36;
	const s_x = 330, s_y = 90, row_step = 42, w = 58, step = 64;

	text_(svg, "second token", grid_x + cw - 3, 72, 12).attr("text-anchor", "middle");
	text_(svg, "first token", 52, grid_y + ch - 3, 12).attr("text-anchor", "middle")
	   .attr("transform", "rotate(-90 52 " + (grid_y + ch - 3) + ")");
	text_(svg, "sampled drafts after “I flew from Paris to”", s_x, 72, 12);

	function draw(mode) {
		g.selectAll("*").remove();
		for (let a = 0; a < 2; a += 1) {
			text_(g, first[a], grid_x - 10, grid_y + a * ch + ch / 2 + 4, 12).attr("text-anchor", "end");
			for (let b = 0; b < 2; b += 1) {
				if (a == 0) {
					text_(g, second[b], grid_x + b * cw + cw / 2, grid_y - 10, 12).attr("text-anchor", "middle");
				}
				const prob = joint[mode][a][b];
				rect(g, grid_x + b * cw, grid_y + a * ch, cw - 6, ch - 6, a == b ? green : red, 0.15 + 1.3 * prob);
				text_(g, prob.toFixed(2), grid_x + b * cw + (cw - 6) / 2, grid_y + a * ch + (ch - 6) / 2 + 4, 12, "black")
				   .attr("text-anchor", "middle");
			}
		}
		const mixed = joint[mode][0][1] + joint[mode][1][0];
		text_(g, "mixed-up pairs: " + Math.round(100 * mixed) + "% of drafts", 40, grid_y + 2 * ch + 22, 12);
		text_(g, "target: New York or Los Angeles, 50 / 50", 40, grid_y + 2 * ch + 42, 11);

		let total = 0;
		samples[mode].forEach(function(sample, r) {
			const y = s_y + r * row_step;
			const coherent = sample[0] == sample[1];
			const tokens = [first[sample[0]], second[sample[1]]].concat(tail);
			tokens.forEach(function(token, k) {
				let color = green, opacity = 1.0;
				if (!coherent && k == 1) {
					color = red;
				} else if (!coherent && k > 1) {
					color = draft_token_color;
					opacity = 0.45;
				}
				token_box(g, s_x + k * step, y, w, token, color, opacity);
			});
			const produced = coherent ? 5 : 2;
			total += produced;
			text_(g, "+" + produced + " tokens", s_x + 4 * step + 8, y + 15, 12);
		});
		text_(g, "average: " + (total / 4).toFixed(1) + " tokens per round", s_x, s_y + 4 * row_step + 14, 13);
	}

	tabs_(svg, ["independent marginals", "with a causal head"], 200, 25, draw);
}

marginals_vs_causal();

</script>

![](.)
*A drafter that is unsure between "New York" and "Los Angeles" after "I flew from Paris to". Left: the probability it assigns to each pair of first two tokens, green for coherent pairs and red for mixed-up ones. Right: four sampled drafts. With independent marginals, half of the drafts mix the two cities, get rejected at the second token and lose the rest of the block (faded). A causal head conditions each token on the previous one and keeps every draft coherent. Click the titles to switch.*

Marginals fail in a second, more mundane way too. [Chen et al. (2026)](https://arxiv.org/pdf/2610.06011) call it the **repetition trap**: neighbouring positions often predict the same token twice, since nothing tells position $k$ that position $k-1$ has already taken it, and blocks that fall into the trap get shorter accepted prefixes. The drafters in this section put the dependency back without giving up the cheap parallel backbone.

### DSpark

**DSpark** ([Cheng et al., 2026](https://arxiv.org/pdf/2607.05147)) from DeepSeek adds a light causal head on top of a DFlash-style parallel backbone. Its default **Markov head** adds a low-rank transition bias to the backbone's logits $\mathbf{U}\_k$, so that the distribution at position $k$ depends on the token drafted just before it:

$$q_k(\cdot \mid x_{\le t+k-1}) = \operatorname{softmax}\big(\mathbf{U}_k + \mathbf{W}_1[x_{t+k-1}]\,\mathbf{W}_2\big),$$

where the lookup table $\mathbf{W}\_1 \in \mathbb{R}^{V \times 256}$ and $\mathbf{W}\_2 \in \mathbb{R}^{256 \times V}$ form a rank-256 transition matrix. The backbone still runs once; then tokens are sampled left to right, each one picking its row of the transition bias. Once the block is drafted, its logits for verification are a single extra matmul:

```python
def markov_logits(z, prev_tok, W_o, W1, W2):
    """z: [γ, d] backbone states, prev_tok: [γ] previous draft tokens,
    W1: [V, 256], W2: [256, V] - low-rank token transition."""
    return z @ W_o + W1[prev_tok] @ W2
```

Here `W_o` is the target's LM head. The sequential part is so light that scaling the draft from 4 to 16 tokens adds only 0.2–1.3% of latency. The alternative **RNN head** carries a gated recurrent state over the drafted tokens (the last line of our factorization), but in the paper it gives only marginal gains over the Markov head. Here are the parallel architectures we've met so far next to EAGLE-3:

<div id="drafter_architectures" class="svg-container" align="center"></div> 

<script>

function drafter_architectures() {
	const svg = d3.select("#drafter_architectures")
				  .append("svg")
				  .attr("viewBox", "0 0 700 400")
				  .style("width", "100%")
				  .style("max-width", "700px")
				  .style("height", "auto");

	const g = svg.append("g");
	const t_x = 30, t_w = 120, base_y = 296;
	const fuse_x = 195, fuse_w = 110, fuse_y = 212;
	const d_x = 380, slot_w = 46, slot_step = 52, d_w = 3 * slot_step + slot_w;
	const out_y = 52, in_y = 334;
	const layer_y = function(i) { return base_y - i * 26; };
	const slot_x = function(k) { return d_x + k * slot_step; };
	const slot_c = function(k) { return slot_x(k) + slot_w / 2; };
	const drafted = ["mat", "and", "purred", "softly"];

	function target_stack(taps, tap_labels) {
		text_(g, "target", t_x + t_w / 2, layer_y(7) - 10, 13).attr("text-anchor", "middle");
		for (let i = 0; i < 8; i += 1) {
			rect(g, t_x, layer_y(i), t_w, 18, target_color);
		}
		token_box(g, t_x + 10, in_y, t_w - 20, "… sat on the", context_color, 1.0, 11);
		arrow_(g, t_x + t_w / 2, in_y, t_x + t_w / 2, layer_y(0) + 20);
		taps.forEach(function(i, j) {
			link_h(g, t_x + t_w, layer_y(i) + 9, fuse_x, fuse_y + 13, "currentColor", "2,2");
			if (tap_labels) {
				text_(g, tap_labels[j], t_x + t_w + 5, layer_y(i) + 4, 10);
			}
		});
	}

	function box(x, y, w, h, label, color, size=12) {
		rect(g, x, y, w, h, color);
		text_(g, label, x + w / 2, y + h / 2 + size / 3, size, "black").attr("text-anchor", "middle");
	}

	function outputs(first_color) {
		drafted.forEach(function(token, k) {
			token_box(g, slot_x(k), out_y, slot_w, token, k == 0 && first_color ? first_color : draft_token_color, 1.0, 11);
		});
	}

	function inputs(tokens) {
		tokens.forEach(function(token, k) {
			const color = k == 0 ? context_color : (token == "[M]" ? 'white' : draft_token_color);
			token_box(g, slot_x(k), in_y, slot_w, token, color, 1.0, 11);
		});
	}

	function summary(lines) {
		lines.forEach(function(line, i) {
			text_(g, line, 350, 375 + i * 16, 12).attr("text-anchor", "middle");
		});
	}

	function block_backbone(top) {
		target_stack([1, 2, 3, 4, 5]);
		box(fuse_x, fuse_y, fuse_w, 26, "concat + norm", fuse_color);
		text_(g, "2nd to 3rd-to-last layer", fuse_x + fuse_w / 2, fuse_y + 44, 11).attr("text-anchor", "middle");
		const ys = [270, 242, 214, 186, top];
		ys.forEach(function(y) {
			rect(g, d_x, y, d_w, 20, drafter_color);
			link_h(g, fuse_x + fuse_w, fuse_y + 13, d_x - 2, y + 10, "currentColor", "none", true);
		});
		text_(g, "K, V", fuse_x + fuse_w + 18, ys[4] + 4, 10);
		inputs(["the", "[M]", "[M]", "[M]"]);
		for (let k = 0; k < 4; k += 1) {
			arrow_(g, slot_c(k), in_y, slot_c(k), ys[0] + 22);
		}
		text_(g, "5 layers", d_x + d_w + 8, 232, 11);
		return ys[4];
	}

	function draw_medusa() {
		target_stack([]);
		const head_y = 180;
		rect(g, d_x, in_y, d_w, 22, target_color);
		sub_text(g, "last hidden state h", "t", d_x + 38, in_y + 15, 12, "black");
		link_h(g, t_x + t_w, layer_y(7) + 9, d_x - 2, in_y + 11, "currentColor", "none", true);
		for (let k = 0; k < 4; k += 1) {
			arrow_(g, slot_c(k), in_y, slot_c(k), head_y + 26);
			box(slot_x(k), head_y, slot_w, 24, k == 0 ? "LM" : "head " + k, k == 0 ? target_color : head_color, 11);
			arrow_(g, slot_c(k), head_y, slot_c(k), out_y + 24);
		}
		outputs(target_color);
		summary(["reads the last hidden state · draft tokens are independent", "1 pass, heads run on top of the target's own forward pass"]);
	}

	function draw_eagle() {
		target_stack([1, 4, 7], ["low", "mid", "high"]);
		box(fuse_x, fuse_y, fuse_w, 26, "concat + FC", fuse_color);
		const layer_top = 214;
		link_h(g, fuse_x + fuse_w, fuse_y + 13, d_x - 2, layer_top + 13, "currentColor", "none", true);
		box(d_x, layer_top, d_w, 26, "decoder layer", drafter_color);
		box(d_x, 132, d_w, 24, "LM head (shared with the target)", target_color, 11);
		inputs(["the", "mat", "and", "purred"]);
		for (let k = 0; k < 4; k += 1) {
			arrow_(g, slot_c(k), in_y, slot_c(k), layer_top + 28);
			text_(g, "pass " + (k + 1), slot_c(k), out_y - 6, 10).attr("text-anchor", "middle");
		}
		arrow_(g, d_x + d_w / 2, layer_top, d_x + d_w / 2, 158);
		for (let k = 0; k < 4; k += 1) {
			arrow_(g, slot_c(k), 132, slot_c(k), out_y + 24);
		}
		outputs();
		summary(["reads low, mid and high layers at its input · draft tokens are autoregressive", "γ passes, each fed the previous draft token and its own hidden state"]);
	}

	function draw_dflash() {
		const top = block_backbone(158);
		for (let k = 0; k < 4; k += 1) {
			arrow_(g, slot_c(k), top, slot_c(k), out_y + 24);
		}
		outputs();
		summary(["reads 5 layers as keys and values in every layer · draft tokens are independent", "1 pass over the last verified token and a block of masks"]);
	}

	function draw_dspark() {
		const top = block_backbone(158);
		const head_y = 108;
		arrow_(g, d_x + d_w / 2, top, d_x + d_w / 2, head_y + 26);
		box(d_x, head_y, d_w, 24, "Markov head", head_color, 11);
		for (let k = 0; k < 4; k += 1) {
			arrow_(g, slot_c(k), head_y, slot_c(k), out_y + 24);
			if (k > 0) {
				arc_arrow(g, slot_c(k - 1), out_y - 2, slot_c(k) - 4, out_y - 2, 16);
			}
		}
		outputs();
		summary(["reads as DFlash · each draft token sees the previous one", "1 backbone pass, then a light head samples left to right"]);
	}

	const drawers = [draw_medusa, draw_eagle, draw_dflash, draw_dspark];
	tabs_(svg, ["Medusa", "EAGLE-3", "DFlash", "DSpark"], 190, 22, function(i) {
		g.selectAll("*").remove();
		drawers[i]();
	});
}

drafter_architectures();

</script>

![](.)
*Four drafters drawn with the same schema: the target on the left, what the drafter reads from it in the middle, and on the right the drafter with its inputs at the bottom and its draft tokens at the top. Medusa puts independent heads on the target's last hidden state, next to the target's own LM head. EAGLE-3 fuses a low, a middle and a high layer into the input of one decoder layer and drafts one token per pass. DFlash fuses five layers and injects them as keys and values into every draft layer, drafting a block of masked positions in one pass. DSpark keeps the DFlash backbone and adds a Markov head that samples the block left to right. Click the titles to switch.*

The second half of DSpark is about the verification budget. A **confidence head** $\pi_k = \sigma\big(\mathbf{w}^T[\mathbf{z}\_k; \mathbf{W}\_1[x_{t+k-1}]]\big)$ is trained to predict the acceptance rate of position $k$, given that the prefix before it survived. Its training target is exactly the acceptance rate from the first section, $\pi_k^\ast = 1 - \operatorname{TV}(p_k, q_k)$. The product $\prod_{i \leq k} \pi_{b,i}$ then estimates the chance that the first $k$ drafted tokens of request $b$ survive, and a scheduler picks the draft lengths $\gamma_b$ of all requests in the batch to maximize the expected number of accepted tokens per second:

$$\lbrace \gamma_b \rbrace^{*} = \arg\max_{\lbrace \gamma_b \rbrace} \ \underbrace{\sum_b \Big(1 + \sum_{k \le \gamma_b} \prod_{i \le k} \pi_{b,i}\Big)}_{\text{expected tokens per round}} \cdot \ \underbrace{\operatorname{SPS}\Big(\sum_b (\gamma_b + 1)\Big)}_{\text{rounds per second}},$$

where $\operatorname{SPS}$ (steps per second) is the engine's profiled number of rounds per second for a given number of tokens in the batch. Under light load every request drafts long; under heavy load verification becomes compute-bound (recall the first figure of the post), and only confident prefixes get budget. The problem is solved greedily by sorting the survival probabilities of all candidate positions.

DSpark lifts the acceptance length over DFlash by 16–18% on Qwen3-4B/8B/14B (e.g. from 5.33 to 6.17 on Qwen3-8B). In live DeepSeek-V4 serving it was compared against MTP-1: at matched throughput, per-user generation became 60–85% faster for V4-Flash and 57–78% faster for V4-Pro, and at a fixed per-user speed target aggregate throughput grew by about 50%. That makes it the first parallel drafter proven in frontier production, running with $\gamma \leq 5$.

### Domino

**Domino** ([Huang et al., 2026](https://arxiv.org/pdf/2605.29707)) gives the parallel backbone a causal head with full-prefix memory. A small GRU summarizes the embeddings of all tokens drafted so far into a state $\mathbf{s}\_{k-1}$, and a rank-256 correction $\mathbf{W}\_2\sigma(\mathbf{W}\_1[\mathbf{z}\_k; \mathbf{s}\_{k-1}])$ is added to the backbone's logits $\mathbf{U}\_k$, which is the last line of our factorization. The interesting part is training. A head that sees the prefix tempts the optimizer to leave the work to it and neglect the backbone, so Domino uses a base-anchored curriculum: it trains the backbone's own distribution $q_k^{\text{base}} := \operatorname{softmax}(\mathbf{U}\_k)$ first and gradually shifts the loss towards the corrected distribution $q_k$:

$$\mathcal{L} = (1 - \beta)\, \mathcal{L}_{\operatorname{CE}}\big(q_k,\, x_{t+k}\big) + \beta\, \mathcal{L}_{\operatorname{CE}}\big(q_k^{\text{base}},\, x_{t+k}\big),$$

with $\beta$ annealed from 1 to 0 over training. On Qwen3-8B with greedy decoding it raises the acceptance length over DFlash from 6.06 to 7.17 and the speedup from 4.66× to 5.49×.

#### DominoTree

DDTree's score assumes that the distribution at each depth doesn't depend on the path, which is no longer true once a causal head is involved. **DominoTree** ([Lin and Jang, 2026](https://arxiv.org/pdf/2607.08642)) is a training-free best-first tree that scores each node with Domino's correction recomputed along its own root-to-node path, restricted to the top-$M$ candidates per node to keep it cheap. Their ablation separates the two effects: applying the correction at all gives +10.1% acceptance length, and recomputing it along each candidate's path another +4.7%. Overall DominoTree reaches up to 7.3× speedup over autoregressive decoding on Qwen3-8B.

### DFlash2

**DFlash2** ([Inco AI, 2026](https://inco.ai/blog/dflash2/)) takes a different route to coherence. Instead of a sequential head that rewrites each position's full-vocabulary distribution, it selects among the candidates the parallel backbone already proposes. It starts from a diagnosis of DFlash. The correct token is among DFlash's top 16 candidates 99.5% of the time at the first position of the block, but only 87.8% at the seventh: draft quality decays towards the end of the block. Part of the reason is that the block barely attends to itself: the share of attention mass that a draft position spends on the other positions of the block falls from 30% in the first draft layer to 8% in the fifth. At the same time, an oracle that picks the best path through the top-16 candidates would reach an acceptance length of 6.79 instead of DFlash's 4.27, so most of the headroom is in selection.

DFlash2 addresses both with two cheap components. A **two-tap dynamic convolution**, placed before and after each attention and feed-forward sublayer, gives every position direct access to its predecessor:

$$\operatorname{Conv}(\mathbf{z})_k = \mathbf{m}_{k,0} \odot \mathbf{z}_k + \mathbf{m}_{k,1} \odot \mathbf{z}_{k-1},$$

where the coefficients combine a learned base kernel with a small correction computed from the current hidden state, and the first position reads the last verified token. A **selector** then scores every pair of adjacent candidates, $a$ at position $k-1$ and $b$ at position $k$, by adding a low-rank bilinear term to the drafter's logit $U_k(b)$:

$$S_k(a, b) = U_k(b) + \big\langle \mathbf{u}_a \odot \mathbf{v}_k,\ \mathbf{w}_b \big\rangle,$$

where $\mathbf{u}\_a$ and $\mathbf{w}\_b$ are low-rank embeddings of the two candidates and $\mathbf{v}\_k$ is computed from the backbone state $\mathbf{z}\_k$.

All pairs at all positions are scored in one shot, with no extra backbone or LM head pass. Starting from the last verified token, greedy decoding follows the best successor at each step, sampling draws from the same scores, and verification against the target keeps the output exact:

<div id="dflash2_selector" class="svg-container" align="center"></div> 

<script>

function dflash2_selector() {
	const svg = d3.select("#dflash2_selector")
				  .append("svg")
				  .attr("viewBox", "0 0 700 330")
				  .style("width", "100%")
				  .style("max-width", "700px")
				  .style("height", "auto");
	
	const candidates = [
		[["Los", 0.46], ["New", 0.44], ["San", 0.05], ["the", 0.03]],
		[["York", 0.41], ["Angeles", 0.38], ["Francisco", 0.08], ["City", 0.05]],
		[["last", 0.62], ["this", 0.20], ["on", 0.09], ["in", 0.05]],
		[["week", 0.70], ["month", 0.12], ["year", 0.08], ["Monday", 0.04]]
	];
	const pair_bonus = {
		"Los": {"Angeles": 3, "York": -3}, "New": {"York": 3, "City": 2, "Angeles": -3},
		"San": {"Francisco": 3}, "last": {"week": 1, "month": 1, "year": 1},
		"this": {"week": 1, "month": 1, "year": 1}
	};
	const target = ["Los", "Angeles", "last", "week"];
	const bonus = function(a, b) { return (pair_bonus[a] || {})[b] || 0; };
	
	const col_x = function(k) { return 150 + k * 135; };
	const row_y = function(r) { return 70 + r * 34; };
	const w = 92, h = 24, anchor_x = 30, anchor_y = row_y(1.5);
	
	text_(svg, "top 4 of 16 candidates per position, from one backbone pass", col_x(0), 52, 11);
	const lattice = svg.append("g"), path_layer = svg.append("g"), boxes = svg.append("g");
	token_box(boxes, anchor_x, anchor_y, 60, "to", context_color);
	text_(boxes, "verified", anchor_x + 4, anchor_y - 8, 10);
	candidates.forEach(function(column, k) {
		column.forEach(function(cand, r) {
			rect(boxes, col_x(k), row_y(r), w, h, draft_token_color);
			text_(boxes, cand[0], col_x(k) + 6, row_y(r) + 16, 12, "black");
			text_(boxes, cand[1].toFixed(2).substring(1), col_x(k) + w - 5, row_y(r) + 16, 9, "black").attr("text-anchor", "end");
		});
		if (k > 0) {
			candidates[k - 1].forEach(function(a, ra) {
				column.forEach(function(b, rb) {
					const strong = bonus(a[0], b[0]) > 0;
					lattice.append("path")
					   .attr("d", "M " + (col_x(k - 1) + w) + " " + (row_y(ra) + h / 2) + " C " + (col_x(k) - 20) + " " + (row_y(ra) + h / 2) + ", " +
					         (col_x(k - 1) + w + 20) + " " + (row_y(rb) + h / 2) + ", " + col_x(k) + " " + (row_y(rb) + h / 2))
					   .attr("fill", "none")
					   .attr("stroke", "currentColor")
					   .attr("opacity", strong ? 0.45 : 0.08);
				});
			});
		} else {
			column.forEach(function(b, rb) {
				link_h(lattice, anchor_x + 60, anchor_y + h / 2, col_x(0), row_y(rb) + h / 2, "currentColor");
			});
			lattice.selectAll("path").attr("opacity", 0.15);
		}
	});
	
	const strip_y = 250;
	text_(svg, "backbone", anchor_x, strip_y + 15, 12);
	for (let k = 0; k < 4; k += 1) {
		rect(svg, col_x(k) + 6, strip_y, w - 12, 22, drafter_color);
		sub_text(svg, "z", String(k + 1), col_x(k) + w / 2 - 6, strip_y + 15, 12, "black");
		if (k > 0) {
			arc_arrow(svg, col_x(k - 1) + w / 2 + 14, strip_y - 1, col_x(k) + w / 2 - 14, strip_y - 1, 14);
		}
	}
	text_(svg, "two-tap convolution: each position also reads its left neighbour", col_x(0) + 6, strip_y + 42, 11);
	const result_text = text_(svg, "", anchor_x, 318, 13);
	
	function draw(mode) {
		path_layer.selectAll("*").remove();
		const chosen = [];
		let previous = "to";
		candidates.forEach(function(column, k) {
			let best = 0;
			column.forEach(function(cand, r) {
				const score = function(c) { return Math.log(c[1]) + (mode == 1 ? bonus(previous, c[0]) : 0); };
				if (score(cand) > score(column[best])) { best = r; }
			});
			chosen.push(best);
			previous = column[best][0];
		});
		let accepted = 0;
		while (accepted < 4 && candidates[accepted][chosen[accepted]][0] == target[accepted]) { accepted += 1; }
		chosen.forEach(function(r, k) {
			const x0 = k == 0 ? anchor_x + 60 : col_x(k - 1) + w, y0 = k == 0 ? anchor_y + h / 2 : row_y(chosen[k - 1]) + h / 2;
			const color = k < accepted ? green : (k == accepted ? red : "currentColor");
			path_layer.append("path")
			   .attr("d", "M " + x0 + " " + y0 + " C " + (x0 + 30) + " " + y0 + ", " + (col_x(k) - 30) + " " + (row_y(r) + h / 2) + ", " +
			         col_x(k) + " " + (row_y(r) + h / 2))
			   .attr("fill", "none")
			   .attr("stroke", color)
			   .attr("stroke-width", 3)
			   .attr("opacity", k <= accepted ? 1.0 : 0.35);
			path_layer.append("rect")
			   .attr("x", col_x(k) - 2).attr("y", row_y(r) - 2).attr("width", w + 4).attr("height", h + 4)
			   .attr("rx", 4).attr("fill", "none").attr("stroke", color).attr("stroke-width", 2.5)
			   .attr("opacity", k <= accepted ? 1.0 : 0.35);
		});
		const words = chosen.map(function(r, k) { return candidates[k][r][0]; }).join(" ");
		result_text.text("draft: “" + words + "”, target accepts " + accepted + " of 4");
	}
	
	tabs_(svg, ["top-1 marginals (DFlash)", "selector path (DFlash2)"], 150, 22, draw);
}

dflash2_selector();

</script>

![](.)
*DFlash2's selection for the block after "I flew from Paris to", when the target continues with "Los Angeles last week". Each column holds the top candidates of one position from a single backbone pass, with their probabilities, and darker links mark pairs the selector likes. Taking the top candidate at every position mixes two cities and loses the block after the first token; the selector follows the best successor from the last verified token and keeps the draft coherent. Below, the two-tap convolution lets each backbone position read its left neighbour. Click the titles to switch.*

In terms of our factorization this is a parallel backbone with a light head that sees only the previous token, restricted to the 16 candidates of each position. Both components add 18.5M parameters and about 1.3% to the latency of a round. On Qwen3.5-4B the mean acceptance length goes from 4.92 with DFlash and 5.49 with DSpark to 5.97. For Qwen3.8-27B with blocks of 8, DFlash2 reaches 4.80 against 4.28 for the model's native MTP, and 2.7–3.4× the throughput of autoregressive decoding.

### Looping and editing the block

A parallel drafter is cheap enough that a second pass of it is affordable: with $c = 0.05$, two passes still cost a tenth of a target pass. So instead of adding a head, we can loop the same drafter over its own block, which is the Jacobi-style refinement from the beginning of the story[^JAC], now done by a drafter rather than by the target. Such drafters don't fit a single line of our factorization: every extra pass conditions the whole block on the previous guess.

**D-Loop** ([Chen et al., 2026](https://arxiv.org/pdf/2610.06011)), from the authors who named the repetition trap, runs the DFlash drafter twice. The first pass proposes a block, and the second conditions on a selected prefix of it and regenerates the suffix in parallel. The block becomes two semi-autoregressive halves without a single new parameter, and one shared drafter is trained for both roles with a prefix–suffix objective. On Qwen3-4B and Qwen3-8B it beats both DFlash and DSpark.

**DEdit** ([Yu et al., 2026](https://arxiv.org/pdf/2609.38510)) keeps DFlash's architecture but lets the drafter edit its whole draft. After the first unmasking pass, each editing pass takes the current draft tokens as input instead of masks and re-predicts every position at once, with bidirectional attention inside the block. Later guesses become context for repairing earlier errors, so a single wrong token no longer has to cut the accepted prefix short. Learning to repair without breaking correct tokens is the hard part. Its ProposalMix training builds editing inputs by mixing the drafter's own first-pass predictions with ground-truth tokens according to the drafter's confidence, which halves the edits that shorten the accepted prefix. With one proposal and two editing passes over a block of 32 positions, DEdit reaches an acceptance length of 7.58 on Qwen3-4B against 6.91 for DSpark with a block of 16, and average speedups of 5.72× and 5.97× on Qwen3-4B and Qwen3-8B with greedy decoding. Every editing pass costs a drafter pass, so the speedup grows less than the acceptance length, and restricting the editor to causal attention lowers acceptance: the future context is what makes editing work.
## Beyond a single round

So far every round has been self-contained: draft, verify once, throw away whatever was rejected, and start the next round from scratch. The newest drafters break these habits.

### Drafting past the verifier

With strong drafters, the target often accepts every drafted token, and the verification pass that follows each drafting stage turns out to be unnecessary. Adaptive draft lengths decide how much to draft before verifying. SpecDec++ ([Huang et al., 2024](https://arxiv.org/pdf/2405.19715)) trains an acceptance prediction head for an autoregressive drafter and stops drafting once the predicted chance of a rejection crosses a threshold, and DSpark's scheduler chooses lengths for a whole batch, but only within one block. Going beyond a block is harder for a parallel drafter: its next block needs target hidden states for tokens that haven't been verified yet. **DLoop** ([Gu et al., 2026](https://arxiv.org/pdf/2610.07659)) runs several drafting stages while the drafter stays confident, feeding it its own hidden states in place of the missing target ones, and then verifies all accumulated tokens in one pass. Loop-aware training exposes the drafter to these self-generated states so that it stays reliable in later stages. In terms of our speedup formula, DLoop trades a few more cheap drafter passes for fewer expensive verifications, and across EAGLE-3, DFlash, Domino, DSpark and MTP drafters it improves walltime by 5 to 41%. PEARL ([Liu et al., 2024](https://arxiv.org/pdf/2408.11850)) attacks the same idle time from the other side: it drafts the next tokens while the target is still verifying the current ones, so that neither model waits for the other.

### Recycling rejected work

Every verification pass also computes target hidden states for the rejected tokens, and the rejected part of a draft often still contains the right tokens a little later. Token Recycling ([Luo et al., 2024](https://arxiv.org/pdf/2408.08696)) was a training-free version of this idea: it stores candidate tokens from earlier steps in an adjacency matrix and grows the next draft tree from them. Two recent methods recycle much richer state into parallel drafters.

**ReTrace** ([Lin et al., 2026](https://arxiv.org/pdf/2608.29748)) shows that rejected positions of a DFlash draft often still align with the target's continuation, and conditions the next block on the rejected suffix instead of starting from fresh mask tokens. It keeps the hidden representations of the rejected suffix, aligns them with the new block, corrects them with signals from the same verification pass, and fuses them into the drafter's input embeddings through a gated residual connection, without an extra forward pass.

**Carryover drafting** ([Koo et al., 2026](https://arxiv.org/pdf/2609.14717)) recycles the target's side of the work: hidden states the verifier computed for rejected tokens become temporary KV context for the drafter, marked with a single learned embedding and replaced every round, so their size stays bounded by one block. The hard part is training, since standard parallel training never produces rejected states that look like the ones at inference. Carryover's draft–verify–draft training produces them while keeping all training positions in parallel. With DFlash and a DSpark-style drafter it raises the acceptance length by 6.5–14.7% and the vLLM speedup by 7.9–14.4%, and by up to 28.8% on translation.

Both stay lossless, because rejected tokens are never committed: they only inform the next draft. And both change what a drafter keeps between rounds, which brings us to memory.

## Drafters and the KV cache

So far we've counted only the weights a drafter reads. With long contexts and large batches the KV cache takes over. MagicDec ([Sadhukhan et al., 2024](https://arxiv.org/pdf/2408.11049)) showed that in this regime decoding is bound by reading the KV cache, which grows with both the sequence length and the batch size, and that speculative decoding keeps paying off even at batch sizes from 32 to 256, as long as the drafter's own KV cache stays small. So the drafter's memory is an architectural choice of its own.

DFlash and the drafters built on it keep their own KV cache: keys and values projected from $\mathbf{H}$ in every draft layer, for the whole context. A real world example: a DFlash-style drafter for Qwen3-8B has 5 layers with the target's attention layout of 8 KV heads of dimension 128, so for one request with 32k tokens of context in bf16 it needs

$$\underset{\mathbf{K/V}}{2} \cdot \underset{\text{bf16}}{2} \cdot \underset{\text{sequence length}}{32{,}768} \cdot \underset{\text{draft layers}}{5} \cdot \underset{\text{KV heads}}{8} \cdot \underset{\text{head dim}}{128} = \underset{\text{bytes per request}}{671{,}088{,}640}.$$

That's 0.67 GB per request, only 14% on top of the target's own 36-layer cache, but with 128 such requests in flight the drafter alone needs 86 GB, more than the whole HBM of an H100. For targets with hybrid attention the ratio is much worse: Qwen3.5-397B-A17B uses linear attention in most of its layers, so its own cache is small, and a drafter cache inflates the total by 1.8×. Besides memory, the drafter has to project and write keys and values for every newly verified token, and on Qwen3-8B this injection step gets 4.7× slower going from 4 to 128 concurrent requests.

<div id="drafter_memory" class="svg-container" align="center"></div> 

<script>

function drafter_memory() {
	const svg = d3.select("#drafter_memory")
				  .append("svg")
				  .attr("viewBox", "0 0 700 290")
				  .style("width", "100%")
				  .style("max-width", "700px")
				  .style("height", "auto");
	
	const x_start = 80, x_end = 560, y_start = 30, y_end = 240;
	const per_request = 2 * 2 * 32768 * 5 * 8 * 128 / 1e9;
	const x = d3.scaleLinear().domain([1, 128]).range([x_start, x_end]);
	const y = d3.scaleLinear().domain([0, 100]).range([y_end, y_start]);
	
	arvo_axis(svg.append("g")
	   .attr("transform", "translate(0," + y_end + ")")
	   .call(d3.axisBottom(x).tickValues([1, 16, 32, 48, 64, 80, 96, 112, 128])));
	arvo_axis(svg.append("g")
	   .attr("transform", "translate(" + x_start + ", 0)")
	   .call(d3.axisLeft(y).ticks(5)));
	text_(svg, "GB", x_start - 25, y_start - 12, 11);
	text_(svg, "concurrent requests, 32k context each", x_start + 120, y_end + 40, 11);
	
	line_(svg, x_start, y(80), x_end, y(80), "currentColor", 1, "4,3");
	text_(svg, "H100 HBM: 80 GB", x_start + 10, y(80) - 6, 11);
	line_(svg, x_start, y(16), x_end, y(16), orange, 1.5, "1,3");
	text_(svg, "8B target weights: 16 GB", x_start + 220, y(16) + 14, 11, orange);
	
	line_(svg, x(1), y(per_request), x(128), y(128 * per_request), drafter_color, 2);
	text_(svg, "drafter KV cache", x(128) + 8, y(128 * per_request) + 4, 12, drafter_color);
	
	[16 / per_request, 80 / per_request].forEach(function(n) {
		svg.append("circle").attr("cx", x(n)).attr("cy", y(n * per_request)).attr("r", 4)
		   .attr("fill", drafter_color).attr("stroke", "black");
		text_(svg, Math.round(n) + " requests", x(n) - 75, y(n * per_request) - 6, 11);
	});
}

drafter_memory();

</script>

![](.)
*Memory taken by the KV cache of a DFlash-style drafter (5 layers, 8 KV heads of dimension 128, bf16) for Qwen3-8B at 32k context versus the number of concurrent requests. It overtakes the target's weights at 24 requests and the whole HBM of an H100 at 119.*

There are three ways to keep the drafter's memory in check, and all of them appeared before parallel drafters.

**Shrink the drafter's cache.** MagicDec drafts with a sparse KV cache, and TriForce ([Sun et al., 2024](https://arxiv.org/pdf/2404.11912)) drafts with the target itself over a small, retrieval-based selection of its KV cache. SpecExtend ([Cha et al., 2025](https://arxiv.org/pdf/2505.20776)) decides which part of the context a small drafter keeps using the target's own attention scores, without any retraining, and QuantSpec ([Tiwari et al., 2025](https://arxiv.org/pdf/2502.10424)) drafts with the target over a hierarchical 4-bit quantized KV cache and 4-bit weights, accepting over 90% of the drafted tokens. Most recently, [Yuan et al. (2026)](https://arxiv.org/pdf/2608.30252) give a strong drafter a compressed memory that a small adaptor builds and updates incrementally, cutting drafter-side memory by over 70% at 32k contexts.

**Reuse the target's cache.** Self-speculative decoding drafts with a part of the target itself: Draft & Verify ([Zhang et al., 2023](https://arxiv.org/pdf/2309.08168)) skips some of its intermediate layers, and LayerSkip ([Elhoushi et al., 2024](https://arxiv.org/pdf/2404.16710)) trains the target with early exits and drafts with its first layers. Verification then reuses the cache and activations of those layers, so the drafter needs no memory of its own. GliDe ([Du et al., 2024](https://arxiv.org/pdf/2402.02082)) keeps a separate small drafter but lets it cross-attend to the target's KV cache, and LongSpec ([Yang et al., 2025](https://arxiv.org/pdf/2502.17421)) pairs this cross-attention with self-attention over a 512-token sliding window, so the drafter's own cache stays constant however long the context grows. Gemma 4's MTP assistants go all the way: their decoder layers have only query projections and read keys and values straight from the target's cache.

**Keep a recurrent state.** ReDrafter ([Cheng et al., 2024](https://arxiv.org/pdf/2403.09919)) drafts with a small RNN conditioned on the target's last hidden state, and Mamba drafters ([Choi et al., 2025](https://arxiv.org/pdf/2506.01206)) replace the drafter's transformer with a state space model. OWL ([Lee et al., 2025](https://arxiv.org/pdf/2510.07535)) drafts with an LSTM conditioned only on the state of the last token, and on long inputs, where EAGLE-3 slows decoding down to 0.81×, it reaches about 5 times EAGLE-3's acceptance length. Either way the drafter carries a fixed-size state instead of a cache that grows with the context.

### H-Spec

**H-Spec** ([Jiang et al., 2026](https://arxiv.org/pdf/2609.24197)) brings the last two ideas to a parallel drafter. Its starting point is the obvious fix. DFlash keeps its own keys and values only because it projects the target's hidden states into them, so why not let each draft layer attend to the keys and values of one target layer directly, in place, like GliDe? The authors tried exactly this with DFlash. The first two positions of the block stayed as good as before, but conditional acceptance at later positions fell by up to 16.5 percentage points. Keys and values are made for retrieval: each describes one position for future queries, and none of them summarizes the context as a whole, which is what the projected hidden states gave DFlash.

A summary of constant size is already there. Under causal attention, the target's hidden state $\mathbf{h}\_t$ at the last position has seen the whole context, and the target computes it anyway in every verification pass. The question is how a parallel drafter should consume one vector per round, and this is where state space models come in. Recall the recurrent form of [linear attention](https://astralord.github.io/posts/transformer-inference-optimization-toolset/#linear-attention) from the toolset post, which folds the whole prefix into a single matrix, $\mathbf{U}\_i = \mathbf{U}\_{i-1} + \phi(\mathbf{k}\_i)\mathbf{v}\_i^T$. At the core of **Mamba-2** ([Dao and Gu, 2024](https://arxiv.org/pdf/2405.21060)) is the same recurrence with a learned, input-dependent forgetting factor $a_i \in (0, 1)$:

$$\mathbf{U}_i = a_i\, \mathbf{U}_{i-1} + \mathbf{k}_i \mathbf{v}_i^T, \qquad \mathbf{o}_i = \mathbf{U}_i^T \mathbf{q}_i,$$

where $a_i$, $\mathbf{q}\_i$, $\mathbf{k}\_i$ and $\mathbf{v}\_i$ are all computed from the current input (Mamba calls them $\bar{A}$, $C$, $B$ and $x$). An attention layer has to keep a key and a value for every past token, so its memory grows with the context, and every new token reads all of it. A Mamba layer keeps only its state $\mathbf{U}$, whose size doesn't depend on the context length, and the forgetting factor lets it decide what to keep. This is how hybrid targets like Qwen3.5 get their small caches: most of their layers are linear-attention layers of this kind. The price is that the state is a lossy summary that can't retrieve an arbitrary token exactly, so such models still keep some full attention layers.

H-Spec splits the two jobs between two modules, and the target provides both inputs for free. Each of its four layers starts with a Mamba-2 module, and three of them follow it with attention. The Mamba modules don't scan the context at all. The last-position hidden states of five target layers are fused and projected into their initial state $\mathbf{U}\_0$, and they scan only the positions of the block. Since the recurrence is linear, the scan over the whole block runs as one parallel pass, and identical mask tokens still get different states at different positions. The attention modules match the target's KV layout and read the keys and values of three target layers in place, through a 2,048-token sliding window, for the token-level detail a summary can't hold. On top sits DSpark's Markov head. For Qwen3-8B the Mamba state has 48 heads × 64 × 4 = 12,288 numbers per layer and is rebuilt every round, so the drafter keeps nothing of its own between rounds: nothing grows with the context, and nothing has to be injected.

<div id="hspec_context" class="svg-container" align="center"></div> 

<script>

function hspec_context() {
	const svg = d3.select("#hspec_context")
				  .append("svg")
				  .attr("viewBox", "0 0 700 400")
				  .style("width", "100%")
				  .style("max-width", "700px")
				  .style("height", "auto");
	
	const words = ["The", "cat", "sat", "on", "the", "warm", "mat", "and", "then", "it", "purred"];
	const x0 = 30, step = 44, w = 40, cell = 12;
	const tok_y = 300, target_y = 214, own_y = 140, block_x = 500;
	const col_x = function(i) { return x0 + i * step; };
	const g = svg.append("g");
	let mode = 0, seq_len = 7;
	
	function column(x, y, n, color) {
		for (let j = 0; j < n; j += 1) {
			rect(g, x + (w - cell) / 2, y + j * (cell + 2), cell, cell, color);
		}
	}
	
	function draw() {
		g.selectAll("*").remove();
		const last_x = col_x(seq_len - 1);
		
		text_(g, "target KV cache, kept anyway", x0, target_y + 54, 11);
		for (let i = 0; i < seq_len; i += 1) {
			column(col_x(i), target_y, 3, target_color);
			token_box(g, col_x(i), tok_y, w, words[i], context_color, 1.0, 11);
		}
		text_(g, "draft block", block_x, tok_y + 38, 11);
		token_box(g, block_x, tok_y, w, words[seq_len], context_color, 1.0, 11);
		for (let k = 1; k < 4; k += 1) {
			token_box(g, block_x + k * step, tok_y, w, "[M]", "white", 1.0, 11);
		}
		
		if (mode == 0) {
			text_(g, "DFlash's own KV cache, written for every new token", x0, own_y - 8, 11, drafter_color);
			for (let i = 0; i < seq_len; i += 1) {
				column(col_x(i), own_y, 2, drafter_color);
				if (i < seq_len - 1) {
					line_(g, col_x(i) + w / 2, target_y - 2, col_x(i) + w / 2, own_y + 2 * cell + 4, "currentColor", 1, "2,2");
				}
			}
			arrow_(g, last_x + w / 2, target_y - 2, last_x + w / 2, own_y + 2 * cell + 4);
			bezier_(g, block_x, tok_y + 11, block_x - 60, tok_y + 11, last_x + w + 60, own_y + cell, last_x + w + 2, own_y + cell, "currentColor", "4,3", true);
			text_(g, "drafter memory grows with the context:", x0, 52, 12);
			text_(g, "20 KiB per token, 0.67 GB at 32k tokens (Qwen3-8B)", x0, 70, 12);
		} else {
			bezier_(g, block_x, tok_y + 11, block_x - 60, tok_y + 11, last_x + w + 60, target_y + 20, last_x + w + 2, target_y + 20, "currentColor", "4,3", true);
			
			rect(g, last_x, 156, w, 22, target_color);
			sub_text(g, "h", "t", last_x + w / 2 - 5, 171, 12, "black");
			arrow_(g, last_x + w / 2, target_y - 2, last_x + w / 2, 180);
			rect(g, 360, 72, 120, 24, fuse_color);
			text_(g, "fuse & project", 420, 88, 11, "black").attr("text-anchor", "middle");
			bezier_(g, last_x + w / 2, 156, last_x + w / 2, 126, 420, 126, 420, 98, "currentColor", "none", true);
			
			const grid_x = 522, grid_y = 60;
			text_(g, "Mamba state, fixed size", grid_x, grid_y - 8, 11, drafter_color);
			for (let r = 0; r < 4; r += 1) {
				for (let c = 0; c < 10; c += 1) {
					rect(g, grid_x + c * 12, grid_y + r * 12, 10, 10, drafter_color, 0.35 + 0.65 * ((r * 7 + c * 3) % 5) / 4);
				}
			}
			arrow_(g, 480, 84, grid_x - 4, 84);
			
			const scan_y = 262;
			arrow_(g, block_x + w / 2, grid_y + 50, block_x + w / 2, scan_y - 7);
			for (let k = 0; k < 4; k += 1) {
				const cx = block_x + k * step + w / 2;
				g.append("circle").attr("cx", cx).attr("cy", scan_y).attr("r", 6)
				   .attr("fill", drafter_color).attr("stroke", "black");
				line_(g, cx, tok_y, cx, scan_y + 6);
				if (k > 0) {
					arrow_(g, cx - step + 7, scan_y, cx - 7, scan_y);
				}
			}
			text_(g, "one parallel scan", block_x + 1.5 * step + w / 2, scan_y - 14, 10).attr("text-anchor", "middle");
			text_(g, "drafter memory stays constant:", x0, 52, 12);
			text_(g, "attention reads the target's cache in place,", x0, 70, 12);
			text_(g, "the Mamba state is rebuilt every round", x0, 88, 12);
		}
	}
	
	tabs_(svg, ["DFlash", "H-Spec"], 290, 22, function(i) { mode = i; draw(); });
	
	const l_x = d3.scaleLinear()
	    .domain([2, 10])
	    .range([0, 300])
	    .clamp(true);
	createSlider(svg, function(v) { seq_len = v; draw(); }, l_x, 200, 372, "L", "currentColor", seq_len, roundN, 8);
}

hspec_context();

</script>

![](.)
*How a parallel drafter conditions on a context of $L$ tokens. DFlash projects the target's hidden states into its own keys and values, so its cache grows with every token. H-Spec's attention reads the target's KV cache in place, and its Mamba layers start from a fixed-size state built from the target's last hidden state $\mathbf{h}\_t$, then scan the block in one parallel pass. Move the slider to grow the context, and click the titles to switch.*

Both halves matter. With the architecture fixed, dropping the target's keys and values cuts the acceptance length by 24–31%, and dropping the summary by 7–12%. At a matched parameter count, the hybrid backbone beats a Mamba-only drafter by 29–45% and an attention-only one by 13–15%. Against P-EAGLE, DFlash and DSpark retrained with the same recipe, H-Spec improves the acceptance length on Llama-3.1-8B, Qwen3-4B and Qwen3-8B by 13.3%, 5.0% and 8.7% and keeps its margin from 1k to 32k tokens of context. In vLLM it reaches 5–17% higher peak throughput at the lowest KV cache utilization, and its advantage grows with concurrency. The costs are a deeper computational graph, which makes the hybrid backbone draft slower than DFlash's below about 32 concurrent requests or 512 tokens of input, and attention layers that must copy the target's KV layout.

Serving engines have converged on a handful of these drafter families: vLLM's benchmark on AMD GPUs, for example, compares exactly four `method` values of `--speculative-config`, namely `mtp`, `eagle3`, `dflash` and `dspark`. Across all of them the architecture has been moving the same way, from drafting step by step towards a parallel backbone, a small causal head and deep conditioning on the target, all built with the serving engine in mind[^SV2]. What's left is the objective.

## Training the drafter

Let's get back to the formula from the first section: the speedup is governed by the acceptance length, and the acceptance length by $\alpha = 1 - \operatorname{TV}(p, q)$. Yet EAGLE-3, MTP, DFlash and Domino are all trained with cross-entropy, i.e. towards a KL objective. Newer drafters are moving away from it: DSpark and H-Spec put 90% of their loss on a total variation term, and Kimi-K3 drops KL altogether.

### KL is only a proxy

The usual recipe is cross-entropy on tokens generated by the target, or forward KL to the target's full distribution, which is the same objective with a denser signal. Both are perfectly reasonable: KL reaches zero exactly when $q = p$, and there $\alpha = 1$. But this only helps if the drafter family $\mathcal{Q}$ contains $p$. A drafter typically has 1–5% of the target's parameters and sees it only through a few hidden states, so in general it can't represent $p$ exactly. And once $q$ can't match $p$, the two objectives can pick different solutions:

$$q_{\text{KL}} = \arg\min_{q \in \mathcal{Q}} \operatorname{KL}(p \,\Vert\, q) \ \ne \ q_{\alpha} = \arg\max_{q \in \mathcal{Q}} \sum_x \min\big(p(x), q(x)\big).$$

Pinsker's inequality,

$$1 - \alpha \le \sqrt{\tfrac{1}{2} \operatorname{KL}(p \,\Vert\, q)},$$

says that a low KL guarantees a high $\alpha$, but it says nothing about which $q \in \mathcal{Q}$ has the highest $\alpha$. Intuitively the objectives disagree in how they handle modes they can't fit. Forward KL is mass-covering: it punishes $q(x) \approx 0$ wherever $p(x) > 0$, so a drafter with limited capacity spreads over all modes of the target. Reverse KL, $\operatorname{KL}(q \,\Vert\, p)$, is mode-seeking: it punishes $q$ for putting mass where $p$ has little, so the drafter collapses onto a single mode. The acceptance rate only rewards overlap. The observation that TV, and not KL, is the divergence tied to acceptance isn't new: DistillSpec ([Zhou et al., 2023](https://arxiv.org/pdf/2310.08461)) already compared it with KL variants for distilling drafters. Here's a toy example, where the target is a mixture of three Gaussians and the drafter is a single one:

<div id="kl_vs_alpha" class="svg-container" align="center"></div> 

<script>

function kl_vs_alpha() {
	const svg = d3.select("#kl_vs_alpha")
				  .append("svg")
				  .attr("viewBox", "0 0 700 400")
				  .style("width", "100%")
				  .style("max-width", "700px")
				  .style("height", "auto");
	
	const xs = d3.range(241).map(function(i) { return -10 + i / 12; });
	const mixture = [[0.4, -4.0, 0.8], [0.3, -0.5, 0.4], [0.3, 6.0, 0.4]];
	const log_sum_exp = function(values) {
		const top = d3.max(values);
		return top + Math.log(d3.sum(values, function(v) { return Math.exp(v - top); }));
	};
	const log_p_raw = xs.map(function(x) {
		return log_sum_exp(mixture.map(function(m) {
			return Math.log(m[0]) - (x - m[1]) * (x - m[1]) / (2 * m[2] * m[2]) - Math.log(m[2]);
		}));
	});
	const log_z_p = log_sum_exp(log_p_raw);
	const log_p = log_p_raw.map(function(v) { return v - log_z_p; });
	const p = log_p.map(Math.exp);
	
	function stats(mu, sigma) {
		const logits = xs.map(function(x) { return -(x - mu) * (x - mu) / (2 * sigma * sigma); });
		const log_z = log_sum_exp(logits);
		let kl = 0, reverse_kl = 0, alpha = 0;
		const q = logits.map(function(l, i) {
			const log_q = l - log_z, q_i = Math.exp(log_q);
			kl += p[i] * (log_p[i] - log_q);
			reverse_kl += q_i * (log_q - log_p[i]);
			alpha += Math.min(p[i], q_i);
			return q_i;
		});
		return {q: q, kl: kl, reverse_kl: reverse_kl, alpha: alpha};
	}
	
	const optima = [{label: "▸ forward KL optimum", key: "kl", sign: 1, value: Infinity},
	                {label: "▸ reverse KL optimum", key: "reverse_kl", sign: 1, value: Infinity},
	                {label: "▸ α optimum", key: "alpha", sign: -1, value: Infinity}];
	for (let i = 0; i <= 240; i += 1) {
		for (let j = 0; j <= 86; j += 1) {
			const mu = -6 + i * 0.05, sigma = 0.7 + j * 0.05, s = stats(mu, sigma);
			optima.forEach(function(opt) {
				if (opt.sign * s[opt.key] < opt.value) {
					opt.value = opt.sign * s[opt.key];
					opt.mu = mu;
					opt.sigma = sigma;
					opt.alpha = s.alpha;
				}
			});
		}
	}
	
	const x_start = 50, x_end = 450, y_start = 30, y_end = 250;
	const x = d3.scaleLinear().domain([-10, 10]).range([x_start, x_end]);
	const y = d3.scaleLinear().domain([0, 0.6]).range([y_end, y_start]);
	const fine = d3.range(1201).map(function(i) { return -10 + i / 60; });
	const fine_density = function(log_values) {
		const log_z = log_sum_exp(log_values);
		return log_values.map(function(l) { return Math.exp(l - log_z) * 60; });
	};
	const p_fine = fine_density(fine.map(function(x) {
		return log_sum_exp(mixture.map(function(m) {
			return Math.log(m[0]) - (x - m[1]) * (x - m[1]) / (2 * m[2] * m[2]) - Math.log(m[2]);
		}));
	}));
	const q_fine = function(mu, sigma) {
		return fine_density(fine.map(function(x) { return -(x - mu) * (x - mu) / (2 * sigma * sigma); }));
	};
	
	svg.append("clipPath").attr("id", "kl_clip")
	   .append("rect").attr("x", x_start).attr("y", y_start).attr("width", x_end - x_start).attr("height", y_end - y_start);
	arvo_axis(svg.append("g")
	   .attr("transform", "translate(0," + y_end + ")")
	   .call(d3.axisBottom(x).ticks(5)));
	
	const area = function(values) {
		return d3.area().curve(d3.curveBasis)
		   .x(function(d, i) { return x(fine[i]); }).y0(y_end).y1(function(d) { return y(d); })(values);
	};
	const curve = function(values) {
		return d3.line().curve(d3.curveBasis)
		   .x(function(d, i) { return x(fine[i]); }).y(function(d) { return y(d); })(values);
	};
	
	svg.append("path").attr("d", area(p_fine)).attr("fill", orange).attr("opacity", 0.45);
	const overlap = svg.append("path").attr("fill", green).attr("opacity", 0.8).attr("clip-path", "url(#kl_clip)");
	svg.append("path").attr("d", curve(p_fine)).attr("fill", "none").attr("stroke", "currentColor").attr("stroke-width", 1);
	const q_path = svg.append("path").attr("fill", "none").attr("stroke", drafter_color).attr("stroke-width", 2.5)
	   .attr("clip-path", "url(#kl_clip)");
	text_(svg, "target p", x(-9.5), y(d3.max(p_fine)) + 10, 12, orange);
	
	const kl_text = text_(svg, "", 480, 50, 14);
	const reverse_kl_text = text_(svg, "", 480, 72, 14);
	const alpha_text = text_(svg, "", 480, 94, 14);
	
	let mu = 3, sigma = 0.7;
	function update() {
		const s = stats(mu, sigma);
		const q_dense = q_fine(mu, sigma);
		q_path.attr("d", curve(q_dense));
		overlap.attr("d", area(q_dense.map(function(v, i) { return Math.min(v, p_fine[i]); })));
		kl_text.text("KL(p || q) = " + s.kl.toFixed(2));
		reverse_kl_text.text("KL(q || p) = " + s.reverse_kl.toFixed(2));
		alpha_text.text("α = " + s.alpha.toFixed(3));
	}
	update();
	
	const mu_x = d3.scaleLinear().domain([-6, 6]).range([0, 330]).clamp(true);
	const sigma_x = d3.scaleLinear().domain([0.7, 5]).range([0, 330]).clamp(true);
	const mu_handle = createSlider(svg, function(v) { mu = v; update(); }, mu_x, 90, 310, "μ", drafter_color, mu, round2, 6);
	const sigma_handle = createSlider(svg, function(v) { sigma = v; update(); }, sigma_x, 90, 360, "σ", drafter_color, sigma, round2, 5);
	
	optima.forEach(function(opt, i) {
		const y0 = 140 + i * 42;
		text_(svg, opt.label, 480, y0, 13).style("cursor", "pointer").on("click", function() {
			mu = opt.mu;
			sigma = opt.sigma;
			mu_handle.attr("cx", mu_x(mu));
			sigma_handle.attr("cx", sigma_x(sigma));
			update();
		});
		text_(svg, "μ = " + opt.mu.toFixed(2) + ", σ = " + opt.sigma.toFixed(2) + ", α = " + opt.alpha.toFixed(2), 
		      492, y0 + 18, 11);
	});
}

kl_vs_alpha();

</script>

![](.)
*A Gaussian drafter $q = \mathcal{N}(\mu, \sigma^2)$ (blue) fitted to a target (orange) that mixes $\mathcal{N}(-4, 0.8^2)$, $\mathcal{N}(-0.5, 0.4^2)$ and $\mathcal{N}(6, 0.4^2)$ with weights 0.4, 0.3 and 0.3, discretized into 241 tokens. The green area is the overlap $\alpha$. Move the sliders, or click the labels on the right to jump to the optimum of each objective.*

The forward KL optimum spreads over all three modes and accepts 42% of tokens. The reverse KL optimum sits on the widest mode and accepts 40%. The $\alpha$-optimal drafter covers the two left modes partially and accepts 54%. Same family, same capacity, and neither direction of KL finds it.

So why not maximize $\alpha$ directly? Look at the gradients with respect to the drafter's logits $\ell$:

$$\frac{\partial \alpha}{\partial \ell_y} = \frac{q(y)}{2}\Big(s(y) - \sum_x q(x)\, s(x)\Big), \qquad \frac{\partial \operatorname{KL}(p \,\Vert\, q)}{\partial \ell_y} = q(y) - p(y),$$

where $s(x) = \operatorname{sign}\big(p(x) - q(x)\big)$. The gradient of $\alpha$ is scaled by $q(y)$ itself, so the tokens where the drafter puts little mass, which are exactly the ones it needs to learn, barely get any push. KL's gradient is dense: wherever $p$ has mass and $q$ doesn't, it pulls. The figure starts in this regime: $q$ sits between two modes, $\alpha \approx 0.005$ and small moves of $\mu$ or $\sigma$ barely change it, while KL reacts to every move. A real drafter starts from $q$ spread over $10^5$ tokens or more, where the TV gradient is tiny, and in our experiments drafters trained on TV alone ended up well behind KL.

### LK losses

In our recent paper we introduced **LK losses** ([Samarin et al., 2026](https://arxiv.org/pdf/2602.23881)), which target the acceptance rate directly. The name comes from $D_{LK}$, Leviathan et al.'s notation for the TV distance in the first section. The first is a hybrid whose weight moves from KL to TV as the drafter improves:

$$\mathcal{L}_{\text{LK}}^{\lambda} = \lambda\, \operatorname{KL}(p \,\Vert\, q_\phi) + (1 - \lambda)\, \operatorname{TV}(p, q_\phi), \qquad \lambda = \exp\big(-\eta \cdot \operatorname{sg}[\alpha]\big),$$

where $\operatorname{sg}$ is a stop-gradient and $\alpha$ is the acceptance rate at the given draft position, averaged over the batch and the sequence. While the drafter is poor, $\lambda \approx 1$ and it gets KL's dense gradients; as $\alpha$ grows, the weight shifts to TV, i.e. to the acceptance rate itself. The second is the likelihood of acceptance:

$$\mathcal{L}_{\text{LK}}^{\alpha} = -\log \alpha = -\log \sum_x \min\big(p(x), q_\phi(x)\big), \qquad \nabla \mathcal{L}_{\text{LK}}^{\alpha} = \frac{1}{\alpha} \nabla \operatorname{TV}(p, q_\phi).$$

It has the same optimum as TV, but the factor $\frac{1}{\alpha}$ amplifies the gradient exactly when acceptance is low. Both losses are computed from the same two logit tensors as KL:

```python
@jit
def drafter_losses(target_logits, draft_logits, eta=3.0):
    """Logits: [N, γ, V], N = batch × sequence positions."""
    log_p = jax.nn.log_softmax(target_logits, axis=-1)
    log_q = jax.nn.log_softmax(draft_logits, axis=-1)
    p, q = jnp.exp(log_p), jnp.exp(log_q)
    kl = (p * (log_p - log_q)).sum(-1)
    alpha = jnp.minimum(p, q).sum(-1)
    lam = jnp.exp(-eta * jax.lax.stop_gradient(alpha.mean(0)))  # one λ per draft position
    lk_hybrid = lam * kl + (1.0 - lam) * (1.0 - alpha)
    lk_likelihood = -jnp.log(alpha)
    return kl.mean(), lk_hybrid.mean(), lk_likelihood.mean()
```

Across drafter architectures and target sizes, both losses improved the acceptance length over KL, most of all where acceptance is lowest: when sampling at temperature 1 and at late draft positions. The fixed blend of 0.1 cross-entropy and 0.9 TV used by DSpark and H-Spec is a hybrid with a constant weight; the adaptive weight lets the drafter lean on KL early in training and on acceptance later.

#### Kimi-K3

Kimi-K3's [technical report](https://github.com/MoonshotAI/Kimi-K3/blob/main/k3_tech_report.pdf) describes how its pre-trained MTP layer is fine-tuned into an EAGLE-3-style drafter: unrolled for seven steps following the training-time test, with the fusion matrix $\mathbf{W}\_{\text{fuse}}$ initialized as $[\mathbf{0}\ \mathbf{0}\ \mathbf{I}]$, so that training starts from the final hidden state the MTP layer was pre-trained on. As for the objective, since minimizing KL doesn't guarantee maximizing the acceptance rate of a capacity-limited drafter, they directly optimize the likelihood-based LK loss $\mathcal{L}\_{\text{LK}}^{\alpha} = -\log \alpha$, at temperature 1 and with no auxiliary cross-entropy term.

### Beyond per-token objectives

KL, TV and LK losses are all per-token: every draft position contributes a term of the same kind. But the acceptance length isn't a sum over positions, it's a sum of products:

$$\tau \approx 1 + \sum_{k=1}^{\gamma} \prod_{i=1}^{k} \alpha_i, \qquad \frac{\partial \tau}{\partial \alpha_k} = \frac{1}{\alpha_k} \sum_{j=k}^{\gamma} \prod_{i=1}^{j} \alpha_i.$$

Position $k$ only pays off if every position before it was accepted, so improving an early position is worth more than improving a late one. For a constant $\alpha$ the weight of position $k$ decays roughly as $\alpha^{k-1}$. Parallel drafters already use such weights: DFlash and Domino multiply the cross-entropy at position $k$ by $\exp\big(-\frac{k-1}{\kappa}\big)$, with $\kappa = 7$ for blocks of 16. That's the decay $\tau$ itself prescribes at $\alpha \approx e^{-1/7} \approx 0.87$, the per-position acceptance behind an acceptance length of 6–7 out of 16, about what DFlash reaches:

<div id="position_weights" class="svg-container" align="center"></div> 

<script>

function position_weights() {
	const svg = d3.select("#position_weights")
				  .append("svg")
				  .attr("viewBox", "0 0 700 300")
				  .style("width", "100%")
				  .style("max-width", "700px")
				  .style("height", "auto");
	
	const gamma = 16, kappa = 7;
	let alpha = 0.87;
	const x_start = 80, x_end = 520, y_start = 25, y_end = 210;
	const x = d3.scaleBand().domain(d3.range(1, gamma + 1)).range([x_start, x_end]).padding(0.25);
	const y = d3.scaleLinear().domain([0, 1.05]).range([y_end, y_start]);
	
	arvo_axis(svg.append("g")
	   .attr("transform", "translate(0," + y_end + ")")
	   .call(d3.axisBottom(x).tickValues([1, 4, 8, 12, 16])));
	arvo_axis(svg.append("g")
	   .attr("transform", "translate(" + x_start + ", 0)")
	   .call(d3.axisLeft(y).ticks(5)));
	text_(svg, "draft position k", x_start + 170, y_end + 35, 11);
	
	const decay = d3.range(1, gamma + 1).map(function(k) { return {k: k, w: Math.exp(-(k - 1) / kappa)}; });
	const bars = svg.append("g");
	svg.append("path")
	   .datum(decay)
	   .attr("fill", "none")
	   .attr("stroke", "currentColor")
	   .attr("stroke-width", 1.5)
	   .style("stroke-dasharray", "4,3")
	   .attr("d", d3.line().curve(d3.curveMonotoneX).x(function(d) { return x(d.k) + x.bandwidth() / 2; }).y(function(d) { return y(d.w); }));
	decay.forEach(function(d) {
		svg.append("circle").attr("cx", x(d.k) + x.bandwidth() / 2).attr("cy", y(d.w)).attr("r", 3)
		   .attr("fill", "currentColor");
	});
	
	rect(svg, 545, 40, 12, 12, orange, 0.85);
	text_(svg, "∂τ/∂αₖ, normalized", 563, 51, 12);
	line_(svg, 543, 72, 559, 72, "currentColor", 1.5, "4,3");
	text_(svg, "exp(−(k − 1)/7)", 563, 76, 12);
	const tau_text = text_(svg, "", 545, 110, 13);
	
	function update(a) {
		alpha = a;
		const raw = d3.range(1, gamma + 1).map(function(k) {
			return d3.sum(d3.range(k, gamma + 1), function(j) { return Math.pow(alpha, j - 1); });
		});
		bars.selectAll("*").remove();
		raw.forEach(function(w, i) {
			const h = w / raw[0];
			rect(bars, x(i + 1), y(h), x.bandwidth(), y(0) - y(h), orange, 0.85);
		});
		tau_text.text("τ = " + ((1 - Math.pow(alpha, gamma + 1)) / (1 - alpha)).toFixed(2));
	}
	update(alpha);
	
	const alpha_x = d3.scaleLinear().domain([0.5, 0.95]).range([0, 330]).clamp(true);
	createSlider(svg, update, alpha_x, x_start + 60, 275, "α", "currentColor", alpha, round2, 5);
}

position_weights();

</script>

![](.)
*Weights that the acceptance length assigns to the acceptance rate of each position in a block of 16 when every position has the same $\alpha$ (bars, normalized to the first position), against DFlash's fixed loss weights (dashed). Move the slider to change $\alpha$.*

A fixed schedule is only right for one acceptance rate. D-PACE ([Wu et al., 2026](https://arxiv.org/pdf/2605.18810)) makes the weights adaptive, computing them from a differentiable surrogate of the acceptance length built from the drafter's own confidences. The E2E-TV loss ([Li et al., 2026](https://arxiv.org/pdf/2606.12370)) drops the per-token form altogether and maximizes the expected number of accepted draft tokens directly, with gradients flowing through the products:

$$\mathcal{L}_{\text{E2E}} = -\sum_{k=1}^{\gamma} \prod_{i=1}^{k} \big(1 - \operatorname{TV}(p_i, q_i)\big).$$

E2E is built for token-wise verification, whose survival probability is a product of per-token terms. Block verification breaks this structure: a surplus at one token can pay for a deficit at the next, so two drafters with the same TV at every position can accept different numbers of tokens. **BV loss** ([Kim et al., 2026](https://arxiv.org/pdf/2609.34832)) derives the objective from block verification itself. For a fixed drafted prefix, averaged over the rest of the draft and the verifier's coins, block verification accepts at least its first $k$ tokens with probability exactly $w_k$, the running weight from the verifier's side. So the expected number of accepted draft tokens is $\sum_k \mathbb{E}\_{x \sim q}[w_k]$. Training text comes from the target, not from the drafter, and importance weighting moves the expectation there. Along a target-generated block the clipped running weight becomes a running minimum of likelihood ratios:

$$\mathbb{E}_{x \sim q}\big[w_k\big] = \mathbb{E}_{x \sim p}\big[\min(1, r_1, \dots, r_k)\big], \qquad r_k = \prod_{i=1}^{k} \frac{q_i(x_{t+i})}{p_i(x_{t+i})}.$$

As in our likelihood loss, a logarithm keeps the gradients alive when acceptance is low, and the loss on a target-generated block is

$$\mathcal{L}_{\text{BV}} = -\log \sum_{k=1}^{\gamma} \min\big(1, r_1, \dots, r_k\big).$$

In practice the last token of each prefix is integrated over the vocabulary, which brings in the target's full distribution and makes the single-token case exactly $\mathcal{L}^{\alpha}\_{\text{LK}}$. DFlash and DSpark drafters trained from scratch with BV loss for Qwen3-4B and Qwen3-8B accept 13–21% more tokens per round under block verification than with cross-entropy and 4–16% more than with LK losses, and the gains carry over to token-wise verification and greedy decoding.

Two more observations from the paper fit this section well. Drafters trained on TV alone reached only 1.0–2.2 tokens per round, the vanishing gradient we saw above. And by the end of training, the cross-entropy drafter matched the target's tokens more often than the BV drafter, 54% against 47%, yet the BV drafter kept 21% more draft tokens per round: like KL, token accuracy is only a proxy.

A second gap is the context the drafter conditions on. Losses are usually computed on target-generated text, while at inference the drafter continues its own guesses within the round, and the anchors where rounds start depend on earlier acceptances. EAGLE-3's training-time test closes the first part of this gap by unrolling the drafter during training. Draft-OPD ([Lei et al., 2026](https://arxiv.org/pdf/2605.29343)) goes further with on-policy distillation: it replays draft proposals from the anchors where rounds actually start at inference and weights the distillation by acceptance. DLoop's loop-aware training and Carryover's draft–verify–draft training apply the same idea to states that only exist across rounds: the drafter's own hidden states for unverified tokens and the target's states for rejected ones.

The third gap is between rounds. For a parallel drafter the distribution at position $t$ depends on where its round started, and where rounds start depends on how many tokens the drafter got accepted before. So the states on which the drafter is evaluated depend on the drafter itself. **EDR** ([Zhao and Cai, 2026](https://arxiv.org/pdf/2610.10411)) captures this by representing speculative decoding as a Markov reward process conditioned on a target rollout of length $L$. A state $(t_0, t)$ means that the current round started at position $t_0$ and has reached position $t$. With probability $A^{q}\_{t_0,t}$ the token is accepted and the process moves to $(t_0, t+1)$; otherwise a new round starts and it moves to $(t, t+1)$, and every rejection costs one round. The expected number of rounds is then exactly

$$\mathbb{E}[\#\text{rounds}] = 1 + \sum_{t=1}^{L} \sum_{t_0=0}^{t-1} \omega^{q}_{t_0,t}\, \epsilon^{q}_{t_0,t},$$

where $\omega^{q}\_{t_0,t}$ is the probability that decoding visits state $(t_0, t)$ (its occupancy), and $\epsilon^{q}\_{t_0,t}$ is the expected local rejection cost there, a TV distance between the drafter and the target.

Two observations fall out of this formula. For an autoregressive drafter, whose prediction at position $t$ doesn't depend on where the round started, the occupancies sum to one at every position and the formula collapses to $1 + \mathbb{E}\big[\sum_t \epsilon_t\big]$: the expected number of rounds is the total TV distance along target rollouts, so a per-token acceptance objective is exactly the right one. For parallel drafters it isn't, and even the product of $1 - \operatorname{TV}$ terms in E2E only approximates the survival probability inside a round, the same $\approx$ as in our formula for $\tau$. EDR weights the local costs by occupancies that depend on $q$, derives an exact temporal-difference gradient that can be estimated without bias from target rollouts, and has no hyperparameters. The same machinery gives an exact offline evaluator of the number of rounds, so two drafters can be compared on shared target rollouts without running speculative decoding. Fine-tuning DSpark and DFly[^DFLY] with EDR gives small but consistent gains over both the released checkpoints and E2E-TV across nine math, code and chat benchmarks.

Per-token, per-block and per-sequence objectives form a ladder: LK losses optimize the acceptance of each token, E2E and BV losses the acceptance length of each round, under token-wise and block verification respectively, and EDR the number of rounds for the whole sequence.

## Conclusion

Decoding is memory-bound: one forward pass reads all the weights, so verifying a handful of tokens costs about as much as generating one. Speculative sampling makes guessing lossless, shared Gumbel noise makes it reproducible, block verification and token trees squeeze more out of every draft, and as long as verification stays memory-bound the speedup reduces to a single formula, $\tau / (n_{\text{draft}} \cdot c + 1)$. Every drafter is a different answer to two questions: what should the guesser see, and how should it be trained?

Architecture decides what $q$ can express. We went from a separate small LLM to autoregressive drafting over target hidden states, to parallel drafters with target states injected into every layer, to parallel drafters that restore dependency with a light causal head or a second pass over their own block, and finally to drafters that carry rejected work into the next round and reuse the target's own cache instead of keeping one. The objective decides which $q$ we actually get, and since drafters are small, the KL optimum is not the acceptance optimum.

Plenty of questions remain open: drafters for long-context and agentic workloads, where prefill and memory dominate and the gains of tree drafting shrink; acceptance-aware training for parallel drafters, which have the most to gain from it; and drafters for diffusion language models, which already generate whole blocks on their own. Whatever comes next, it will still be the art of guessing well.

---
[^JAC]: Jacobi decoding ([Santilli et al., 2023](https://arxiv.org/pdf/2305.10427)) and Lookahead decoding ([Fu et al., 2024](https://arxiv.org/pdf/2402.02057)) let the target refine a whole window of guesses in parallel, and CLLMs ([Kou et al., 2024](https://arxiv.org/pdf/2403.00835)) fine-tune the model to converge in fewer iterations. Drafting without any model, by copying n-grams from the prompt (prompt lookup) or from earlier outputs (suffix decoding), is surprisingly strong for code editing and agent loops, where outputs repeat their inputs.

[^HYD]: Blockwise parallel decoding ([Stern et al., 2018](https://arxiv.org/pdf/1811.03115)) did this first. Hydra ([Ankner et al., 2024](https://arxiv.org/pdf/2402.05109)) made Medusa heads sequentially dependent, and Medusa-2's typical acceptance trades exactness for speed by accepting any plausible token instead of using speculative sampling.

[^SV2]: Serving engines matter as much as drafters. SGLang's Spec V2 overlaps host-side scheduling with GPU work and raises throughput by over 33%, from about 11.4k to 15.3k tokens/s, for Qwen3-8B on a single B200 at concurrency 32 ([LMSYS](https://www.lmsys.org/blog/2026-06-15-next-generation-speculative-decoding-dflash-v2/)).

[^DFLY]: DFly ([Liu et al., 2026](https://arxiv.org/pdf/2607.25852)) comes from the AngelSpec framework. It's another block-diffusion drafter with a predecessor-conditioned autoregressive head, i.e. the same "parallel + causal head" family as DSpark and Domino.
