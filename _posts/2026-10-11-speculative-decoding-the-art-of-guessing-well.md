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

This post is a sequel to [Transformers Inference Optimization Toolset](https://astralord.github.io/posts/transformer-inference-optimization-toolset/), whose notation, memory bandwidth arithmetic and KV cache calculations we'll reuse, so if "decoding is memory-bound" sounds unfamiliar, it's worth skimming its first sections. We'll focus on **drafters attached to the target model**: small networks that read the target's hidden states and run next to it on the same GPU. A separate small LLM appears only as a baseline, drafting by string matching (n-grams, prompt lookup, suffix trees) goes to a footnote[^JAC], and serving systems appear only where they shape how drafters are built and used.

## Why decoding is slow

Let's recall that [text generation](https://astralord.github.io/posts/transformer-inference-optimization-toolset/#kv-cache) runs in two stages. Prefill ingests the whole prompt in one compute-bound pass. Decoding produces one token per forward pass, and the [arithmetic intensity](https://astralord.github.io/posts/transformer-inference-optimization-toolset/#arithmetic-intensity-vs-opsbyte) of each pass is far below the `ops:byte` ratio of the GPU: at batch size 1, every weight is read from HBM, used for one multiply-add with the hidden state of a single token and left alone until the next step.

A real world example: take an 8B model in bf16 on H100 GPU. Just reading its weights once takes

$$\underset{\text{weights, bf16}}{16 \text{ GB}} \ / \ \underset{\text{H100 HBM bandwidth}}{3.35 \text{ TB/s}} \approx 4.8 \text{ ms} \ \Rightarrow \ \le 208 \ \tfrac{\text{tokens}}{\text{s}} \ \text{ for an 8B model at batch size 1},$$

no matter how fast the tensor cores are, while the math for one token, $2 \cdot 8 \cdot 10^9 = 16$ GFLOPs (ignoring attention over the KV cache), takes about $0.016$ ms at H100's $989$ TFLOPs of dense bf16 compute. And here's the punchline: 8 tokens instead of one cost 8 times more math and still nearly the same 4.8 ms, because the weights are read once either way. The pass stays memory-bound until the number of tokens in it reaches the `ops:byte` ratio, $989 / 3.35 \approx 295$.

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

Batching requests fills this gap, but it improves throughput, not latency: every user still gets one token per 4.8 ms. To make a single sequence faster, the extra tokens have to come from the same sequence, and we don't know them yet. Unless we guess them.

### Draft and verify

Suppose we have a cheap **drafter** $q$ that guesses the next tokens much faster than the **target** model $p$ we actually want to sample from. With greedy decoding, one round of **speculative decoding** looks like this:

1. The drafter proposes $\gamma$ tokens $x_{t+1}, \dots, x_{t+\gamma}$ that continue the verified context $x_{\le t}$.
2. The target runs *once* on the context extended with all $\gamma$ guesses. Thanks to causal attention, this pass gives its next-token prediction at each of the $\gamma + 1$ positions.
3. We keep the longest prefix of guesses on which the target's argmax agrees with the drafter and append the target's own token at the first disagreement. If all $\gamma$ guesses agree, the target's prediction after the last one comes for free.

The output is exactly what greedy decoding would have produced, it just arrives faster: a round costs one target pass and yields between 1 and $\gamma + 1$ tokens. The verification pass has already written the KV cache entries of accepted tokens, and those of rejected tokens are dropped.

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

With temperature sampling the target has a distribution $p$ rather than one correct answer, and the output must follow exactly that distribution. **Speculative sampling** ([Leviathan et al., 2023](https://arxiv.org/pdf/2211.17192); [Chen et al., 2023](https://arxiv.org/pdf/2302.01318)) generalizes the matching rule:

$$\text{accept } x \sim q \ \text{ w.p. } \min\Big(1, \frac{p(x)}{q(x)}\Big), \qquad \text{otherwise sample from } \ \frac{(p - q)_+}{\sum_x (p(x) - q(x))_+},$$

where $(\cdot)\_+ = \max(\cdot, 0)$. Tokens the drafter underestimates, $q(x) \le p(x)$, are always kept, tokens it overestimates are thinned out, and the residual tops up exactly the mass the drafter was missing. Averaging the acceptance probability over the drafter's samples gives the **acceptance rate**

$$\alpha = \mathbb{E}_{x \sim q}\Big[\min\Big(1, \frac{p(x)}{q(x)}\Big)\Big] = \sum_x \min\big(p(x), q(x)\big),$$

the overlap of the two distributions. The missing part is the **total variation distance**, the largest difference in probability the two distributions can assign to any set of tokens:

$$1 - \alpha = \operatorname{TV}(p, q) = \frac{1}{2}\sum_x \vert p(x) - q(x) \vert.$$

Leviathan et al. denote it $D_{LK}$. The closer the drafter is to the target in TV, the more of its guesses survive. The residual's normalizer is $\sum_x (p(x) - q(x))\_+ = 1 - \alpha$, the probability of a rejection, so the output follows $p$:

$$
\begin{aligned}
\Pr(\text{output} = x) &= q(x)\min\Big(1, \frac{p(x)}{q(x)}\Big) + (1 - \alpha)\,\frac{(p(x) - q(x))_+}{1 - \alpha} \\ &= \min\big(p(x), q(x)\big) + \big(p(x) - q(x)\big)_+ \\ &= p(x).
\end{aligned}
$$

Whatever the drafter does, the output follows the target exactly, and a bad drafter can only make us slower. Greedy matching is the special case of one-hot $p$ and $q$. For a draft of $\gamma$ tokens the rule is applied left to right and stops at the first rejection. When all $\gamma$ tokens are accepted, treating the missing drafter distribution as zero turns the residual into the target's own distribution for the bonus token:

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

Speculative sampling has one inconvenient property: with the same random seed, a different drafter makes different accept/reject decisions, so the text changes, even though its distribution doesn't. For an alternative, let's look at how sampling is implemented. Sampling from $\operatorname{softmax}(\ell)$ over a vocabulary of $V$ tokens, where $\ell$ are the logits, textbook-style means normalizing, computing a cumulative sum and inverting it with one uniform number. GPUs prefer the **Gumbel-max trick**, one elementwise pass and an argmax instead of a prefix sum over the vocabulary: add independent Gumbel noise to the logits and take the argmax,

$$x = \arg\max_i \big(\ell_i + g_i\big), \qquad g_i = -\log(-\log u_i), \quad u_i \sim U(0, 1).$$

Why does this work? $\xi_i = e^{-g_i} = -\log u_i$ is exponential with rate 1, and maximizing $\ell_i + g_i$ is the same as minimizing $\xi_i e^{-\ell_i}$, exponential with rate $e^{\ell_i}$. Among independent exponential clocks, clock $i$ rings first with probability equal to its rate over the sum of all rates:

$$\Pr(x = i) = \Pr\Big(\xi_i e^{-\ell_i} < \min_{j \neq i} \xi_j e^{-\ell_j}\Big) = \frac{e^{\ell_i}}{\sum_j e^{\ell_j}} = \operatorname{softmax}(\ell)_i.$$

vLLM's sampler, for example, runs exactly this race, taking the argmax of $p_i / \xi_i$, because `torch.multinomial` would force the GPU to synchronize with the CPU. For us the trick matters for another reason: once the noise $g$ is fixed, sampling is a deterministic argmax, just like greedy decoding.

So let the drafter and the target share the noise. At draft position $k$ the drafter proposes $x_{t+k} = \arg\max_i \big(\log q_k(i) + g_{k,i}\big)$, the target computes its own $\arg\max_i \big(\log p_k(i) + g_{k,i}\big)$ in the verification pass, and verification becomes exact matching, as in the greedy case. This is **Gumbel coupling** ([Daliri et al., 2024](https://arxiv.org/pdf/2408.07978)), and its output is exactly what the target would have sampled with that noise on its own. It's **drafter-invariant**: for a fixed seed every drafter produces the same text, only faster or slower, which makes speculative decoding reproducible and much easier to test and debug.

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
*Gumbel-max sampling for the token after "The cat sat on the". In each row, the orange dot is the target's score $\log p + g$ and the blue dot the drafter's; the rightmost dots are the target's sample and the draft, accepted when they coincide. With shared noise both dots of a row move together and the argmaxes mostly agree; with independent noise they rarely do. Click the titles to switch, and draw new noise to watch the match rate converge.*

The price is a lower acceptance rate. Speculative sampling reaches the best possible agreement, $1 - \operatorname{TV}(p, q)$, because the verifier sees both distributions. Two models that share only the noise agree with probability

$$\Pr\big[\text{match}\big] = \sum_j \frac{1}{\sum_i \max\big(\frac{p(i)}{p(j)}, \frac{q(i)}{q(j)}\big)} \ \geq \ \frac{1 - \operatorname{TV}(p, q)}{1 + \operatorname{TV}(p, q)},$$

and no scheme without communication can guarantee more than the right-hand side in the worst case. With independent noise the match probability would drop to $\sum_i p(i)\, q(i)$. In the example above that's 0.679 with shared noise and 0.181 with independent noise, against $\alpha = 0.70$ for speculative sampling, and on real LLM distributions the gap to $1 - \operatorname{TV}$ is often just as small. In JAX the whole draft-and-verify step is a few argmaxes over the same noise:

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

If each drafted token were accepted independently with probability $\alpha$, the number of accepted draft tokens would follow a geometric distribution truncated at $\gamma$, and with the bonus token a round would yield

$$\mathbb{E}[\#\text{tokens}] = \frac{1 - \alpha^{\gamma + 1}}{1 - \alpha}.$$

In practice the drafter gets worse the further it guesses. If $\alpha_k$ is the probability of accepting the $k$-th token given that all previous ones were accepted, the **acceptance length** $\tau$, the expected number of tokens per round, is

$$\tau \approx 1 + \sum_{k=1}^{\gamma} \prod_{i=1}^{k} \alpha_i.$$

This is the number papers report most often. For walltime, let's measure time in target passes, with $c$ the cost of one drafter pass. A drafter that predicts one token per pass runs $\gamma$ times per round, and verification costs about one decoding step (recall the first figure), so

$$\text{speedup} = \frac{\tau}{\gamma c + 1}.$$

A real world example: a decent drafter accepts $\alpha = 0.8$ of tokens, drafts $\gamma = 5$ of them, and costs $c = 0.05$ of a target pass. Then

$$\tau = \frac{1 - 0.8^6}{0.2} \approx 3.69, \qquad \text{speedup} = \frac{3.69}{5 \cdot 0.05 + 1} \approx 2.95\times.$$

A fifth of every round goes to drafting, and this share grows with $\gamma$. Both parts of the formula will keep coming back: better drafters raise $\tau$, and cheaper drafting shrinks the $\gamma c$ term.

## The verifier's side

Before we get to drafters, let's look at the verifier. It can accept more of the same chain, check several continuations at once, and it stops being free once the batch grows.

### Block verification

Speculative sampling stops at the first rejection, which is a myopic decision. If the drafter overestimates $x_{t+1}$ but underestimates $x_{t+2}$, the pair as a whole may be as likely under the target as under the drafter, yet token-wise verification often throws both away. **Block verification** ([Sun et al., 2024](https://arxiv.org/pdf/2403.10444)) decides on the whole draft jointly. With $p_k$ and $q_k$ the target's and the drafter's distributions at position $k$, the verifier keeps a running weight of how much of each prefix the target still covers,

$$w_k = \min\Big(1,\ w_{k-1}\frac{p_k(x_{t+k})}{q_k(x_{t+k})}\Big), \qquad w_0 = 1,$$

so a ratio below one on one token can be compensated by ratios above one later on (note that $\min(1, ab) \geq \min(1, a) \min(1, b)$). Each prefix survives with probability

$$\Pr\big(x_{t+1:t+k} \text{ survives}\big) = \frac{\sum_x \big(w_k\, p_{k+1}(x) - q_{k+1}(x)\big)_+}{\sum_x \big(w_k\, p_{k+1}(x) - q_{k+1}(x)\big)_+ + 1 - w_k},$$

the longest surviving prefix, of length $n$, is accepted, and the next token is sampled from the normalized residual $\big(w_n\, p_{n+1} - q_{n+1}\big)\_+$. With $q_{\gamma+1} \equiv 0$ the bonus token falls out for free again, since then the whole draft survives with probability $w_\gamma$. Compared to `verify`, only the prefix length and the residual change:

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

Sun et al. prove that for a given draft block verification accepts the most tokens in expectation among all lossless rules, so it never loses to the token-wise rule. In a toy example with $p(a) = \frac{1}{3}$, $p(b) = \frac{2}{3}$, $q(a) = \frac{2}{3}$, $q(b) = \frac{1}{3}$ at every position and $\gamma = 2$, token-wise verification accepts $\frac{10}{9}$ tokens per round, block verification $\frac{11}{9}$. With PaLM-2 models the gain was about 8% more tokens per round and 5–8% in walltime, for a few extra vector operations. It works with any drafter, and at the end of the post we'll see how to train a drafter for it.

### Token trees

A chain dies at the first wrong token, but the target can verify several continuations in the same pass. We can draft a **token tree** whose branches share prefixes, flatten it into one sequence and give the target a **tree attention mask**, in which every node attends only to the context and to its own ancestors ([Miao et al., 2023](https://arxiv.org/pdf/2305.09781)), with positional ids following the node depth. Verification walks down from the root, moving at every node to the child the target accepts, and returns the longest root-to-node path the target agrees with. The tree adds nodes to the verification pass, but no extra passes, and its expected acceptance length is

$$\tau(\mathcal{T}) = 1 + \sum_{v \in \mathcal{T}} \Pr\big(\text{path root} \to v \text{ accepted}\big),$$

the tree version of $\sum_k \prod_{i \le k} \alpha_i$. Given a budget of nodes, the best tree keeps the nodes with the highest path probabilities, and that's what the tree builders later in this post try to approximate.

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

function draw_token_tree(svg, budget, opts) {
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
	
	const accepted = opts.accepted || null;
	const on_path = function(path) {
		return accepted != null && path.length <= accepted.length && key(path) == key(accepted.slice(0, path.length));
	};
	
	function draw_edges(node) {
		const x0 = x_root + node.path.length * dx;
		node.children.forEach(function(child) {
			const walked = on_path(child.path);
			link_h(svg, x0 + w, y_top + node.row * dy + 11, x0 + dx, y_top + child.row * dy + 11, 
			       walked ? green : "currentColor");
			draw_edges(child);
		});
	}
	draw_edges(root);
	
	token_box(svg, x_root, y_top, w, "the", context_color);
	let total = 0;
	tree.forEach(function(item, i) {
		const node = nodes[key(item.path)], d = node.path.length;
		const token = tree_marginals[d - 1][node.path[d - 1]][0];
		const x = x_root + d * dx, y = y_top + node.row * dy;
		const color = on_path(node.path) ? green : draft_token_color;
		const opacity = accepted != null && !on_path(node.path) ? 0.45 : 1.0;
		rect(svg, x, y, w, 22, color, opacity);
		if (opts.order) {
			text_(svg, i + 1, x + 4, y + 9, 8, "black");
			text_(svg, token, x + 13, y + 15, 12, "black");
		} else {
			text_(svg, token, x + 5, y + 15, 12, "black").attr("opacity", opacity);
		}
		text_(svg, node.score.toFixed(2).substring(1), x + w - 4, y + 15, 9, "black")
		   .attr("text-anchor", "end").attr("opacity", opacity);
		total += node.score;
	});
	
	if (accepted != null) {
		const last = nodes[key(accepted)];
		const x = x_root + (accepted.length + 1) * dx, y = y_top + last.row * dy;
		link_h(svg, x - dx + w, y + 11, x, y + 11, orange, "2,2");
		token_box(svg, x, y, w, opts.bonus, target_color);
		text_(svg, "bonus", x + w / 2, y + 36, 10).attr("text-anchor", "middle");
	}
	
	const n = tree.length, cell = 14, m_x = 500, m_y = y_top + 20;
	if (opts.mask) {
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
	
	if (opts.marginals) {
		const t_x = 480, t_y = y_top + 22, cw = 52, ch = 26;
		text_(svg, "drafter marginals q_k", t_x, y_top + 2, 11);
		tree_marginals.forEach(function(column, d) {
			text_(svg, "k = " + (d + 1), t_x + d * cw + (cw - 4) / 2, t_y - 6, 10).attr("text-anchor", "middle");
			column.forEach(function(cand, r) {
				const used = tree.some(function(node) { return node.path.length > d && node.path[d] == r; });
				rect(svg, t_x + d * cw, t_y + r * ch, cw - 4, ch - 4, used ? draft_token_color : 'white');
				text_(svg, cand[0], t_x + d * cw + (cw - 4) / 2, t_y + r * ch + 10, 10, "black").attr("text-anchor", "middle");
				text_(svg, cand[1].toFixed(2).substring(1), t_x + d * cw + (cw - 4) / 2, t_y + r * ch + 19, 8, "black")
				   .attr("text-anchor", "middle");
			});
		});
		text_(svg, "τ ≈ 1 + Σ q(path) = " + (1 + total).toFixed(2), t_x, t_y + 3 * ch + 22, 13);
	}
}

function token_tree() {
	const svg = d3.select("#token_tree")
				  .append("svg")
				  .attr("viewBox", "0 0 700 200")
				  .style("width", "100%")
				  .style("max-width", "700px")
				  .style("height", "auto");
	draw_token_tree(svg.append("g"), 7, {mask: true, accepted: [0, 1], bonus: "then"});
}

token_tree();

</script>

![](.)
*A token tree of 7 drafted nodes continuing "...sat on the" (left) and its attention mask (right), where each row is a node and colored cells mark what it attends to. Numbers are the drafter's path probabilities; if calibrated, their sum plus one is the expected acceptance length. The target accepts "mat", then ",", the drafter's second choice at that depth, and appends its own "then": 3 tokens per pass, where the chain of top choices "mat and purred" would have stopped after "mat".*

#### Building trees without breaking exactness

Multi-draft versions of speculative sampling keep trees lossless: SpecInfer's multi-round rule tries the children one by one with the usual ratio, passing the target's residual on to the next sibling after each rejection, and SpecTr ([Sun et al., 2023](https://arxiv.org/pdf/2310.15141)) derives the optimal rule from optimal transport. Both assume that the children are *sampled* independently from $q$: the ratio $p(x) / q(x)$ corrects for how often the drafter proposes $x$, so it's exact only if $q(x)$ really is that frequency.

Practical trees are rarely built this way. EAGLE-2's dynamic tree, the best-first trees of DDTree and DominoTree below, and every tree pruned by a confidence threshold take the *top* candidates deterministically: the top candidate is proposed with probability one, a token outside the top-$k$ with probability zero, and plugging $q(x)$ into the rule biases the output. Other proposal schemes need their own rules, like sampling without replacement through the Gumbel-top-$k$ trick in Recursive Speculative Decoding ([Jeon et al., 2024](https://arxiv.org/pdf/2402.14160)). For a deterministic tree each child's honest proposal is a point mass, and the rule degenerates into **token matching**: sample the target's own token at the node and accept it if it's among the children. EAGLE's reference implementation does exactly this, setting $q(x) = 1$ for every child and renormalizing the target's distribution after each rejected sibling, and DDTree verifies by matching at temperature 1. A node is then accepted with probability $\sum_{x \in \text{children}} p(x)$: the drafter's probabilities shape the tree, never the acceptance. The same goes for anything that reshapes the drafter's distribution after the fact, such as top-$k$ truncation: the verifier must receive exactly the distribution the draft was sampled from. Traversal Verification ([Weng et al., 2025](https://arxiv.org/pdf/2505.12398)) brings block verification to trees with sampled children, checking root-to-leaf sequences from the leaves up instead of discarding a subtree as soon as its root is rejected.

#### Trees and hybrid targets

The tree mask works because softmax attention treats its keys as a set, so every node simply picks out its ancestors. But frontier targets are no longer pure softmax attention: Kimi-K3 interleaves 69 layers of Kimi Delta Attention, a linear attention with a recurrent state, with 24 MLA layers ([Kimi Team, 2026](https://github.com/MoonshotAI/Kimi-K3/blob/main/k3_tech_report.pdf)), Qwen3.5 uses Gated DeltaNet in most of its layers, and DeepSeek-V4 compresses the context into blocks. A recurrent layer updates one state token by token, so a tree needs a separate state per branch, which no mask can express, and with compressed attention branches diverging from a shared prefix compress into different blocks. STree ([Wu et al., 2025](https://arxiv.org/pdf/2505.14969)) verifies trees in state space and hybrid models by scanning them with accumulated state transitions. For DeepSeek-V4-Flash, [Liu and Li (2026)](https://arxiv.org/pdf/2609.24698) combine branch-aware causal verification, temporary per-branch state discarded after the pass, and a refresh of the compressed state along the accepted path. With 8 drafted nodes their trees accept 2.83–3.41 tokens per round against 2.39–2.84 for chains, but throughput grows by at most 18.5% and stops growing with larger budgets even as the acceptance length keeps rising.

Even with softmax attention only, trees are harder for the serving engine than chains:

- **Attention kernels.** The ancestor-only mask is neither causal nor a simple window, so it misses the fast paths of kernels such as FlashAttention-2, and tree verification often falls back to slower generic attention.
- **KV cache bookkeeping.** The verification pass writes keys and values for every node, but only the accepted path survives, so its entries have to be compacted or remapped in a [paged cache](https://astralord.github.io/posts/transformer-inference-optimization-toolset/#pagedattention--vllm). A rejected chain is simply truncated.
- **Static shapes.** Engines capture decoding steps as CUDA graphs and overlap CPU scheduling with GPU work, and both want the same shapes at every step. SGLang, for example, requires a chain (`--speculative-eagle-topk 1`) when its overlap scheduler is on, and that's the default.

### Verification bottlenecks

So far verification cost one decoding step, which holds only while the tokens scored across the batch stay below the `ops:byte` ratio. With $B$ requests and $N$ drafted nodes per request (a chain has $N = \gamma$), the pass scores $B(N + 1)$ tokens. If $T(n)$ is the time of a forward pass over $n$ tokens, the curve from the first figure, the speedup over plain decoding at the same batch size is

$$\text{speedup}(N, B) = \frac{\tau(N) \cdot T(B)}{T\big(B(N + 1)\big) + T_{\text{draft}}},$$

where $T_{\text{draft}}$ is the time spent drafting. The numerator grows slowly: even with an optimal tree, the acceptance length grows only roughly logarithmically with the tree size, as observed in Sequoia ([Chen et al., 2024](https://arxiv.org/pdf/2402.12374)). The denominator grows linearly once $B(N+1)$ passes the knee. So the best tree shrinks as the batch grows, roughly as $N^\ast \approx \frac{\text{ops:byte}}{B}$:

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

With $\alpha = 0.8$, a single request keeps gaining from bigger trees up to the knee, eight requests are best served by about 36 nodes, 32 requests by about 8, and at 128 requests the best "tree" is a chain of two tokens, while a 64-node tree makes decoding almost four times slower than no speculation. That's why Sequoia picks the tree size per hardware, 64 to 128 nodes on an A100 but 768 when the weights are offloaded and streamed to an L40, where the flat part of the curve becomes enormous. In serving, TurboSpec ([Liu et al., 2024](https://arxiv.org/pdf/2406.14066)) adapts speculation to the load by tracking goodput, the rate of tokens that actually get accepted, since naive speculation can make a busy server slower. Together with their engineering costs, this is why trees pay off most where latency matters and the batch is small, while high-throughput serving tends to use chains with a draft length that adapts to the load.

## Anatomy of a drafter

Drafters look very different on the surface, but every drafter in this post is one conditional distribution

$$q_\phi\big(x_{t+k} \mid x_{\le t},\; \mathbf{H},\; x_{t+1:t+k-1}\big), \qquad k = 1, \dots, \gamma,$$

where $x_{\le t}$ is the verified context, $\mathbf{H}$ is whatever the drafter reads from the target (from its last hidden state to several of its layers or even its KV cache), and $x_{t+1:t+k-1}$ are its own earlier guesses in this round. The most consequential decision is how a draft token depends on the guesses before it, since that decides how many sequential passes a round takes. There are three basic factorizations:

$$
\begin{aligned}
\text{independent:} \quad & q(x_{t+1:t+\gamma}) = \textstyle\prod_{k} q_k(x_{t+k} \mid x_{\le t}, \mathbf{H}) \\
\text{autoregressive:} \quad & q(x_{t+1:t+\gamma}) = \textstyle\prod_{k} q(x_{t+k} \mid x_{\le t}, \mathbf{H}, x_{t+1:t+k-1}) \\
\text{parallel + causal head:} \quad & q(x_{t+1:t+\gamma}) = \textstyle\prod_{k} \operatorname{head}\big(\mathbf{z}_k, x_{t+1:t+k-1}\big)(x_{t+k})
\end{aligned}
$$

The independent line costs one drafter pass, but no position knows what the others guessed. The autoregressive line is how an LLM generates text: full dependency for $\gamma$ sequential passes. The last line splits the work: a parallel network computes states $\mathbf{z}\_{1:\gamma} = f_\phi(x_{\le t}, \mathbf{H})$ for all positions in one pass, and only a tiny head runs token by token, looking at the previous guess or at all of them. The next sections follow these lines family by family.

## Autoregressive drafters

The original speculative decoding papers drafted with a separate small LLM from the target's family, for example a 4B model for a 70B Chinchilla in Chen et al. Any smaller model with the same tokenizer will do, but it shares nothing with the target: it recomputes representations the target has already built and keeps its own full KV cache. The drafters in this section stay autoregressive but read the target instead.

### EAGLE

**EAGLE** ([Li et al., 2024](https://arxiv.org/pdf/2401.15077)) drafts autoregressively at the level of hidden states, which it calls features. Each of the target's last hidden states $\mathbf{h}\_i$, right before the LM head, is concatenated with the embedding $e(x_{i+1})$ of the token actually sampled from it, an FC layer reduces the pair back to the hidden size, and a single transformer layer predicts the next hidden state:

$$\mathbf{a}_{t+1} = \operatorname{Layer}_\phi\Big(\operatorname{FC}\big[\mathbf{h}_i;\, e(x_{i+1})\big]_{i \le t}\Big).$$

The target's LM head turns $\mathbf{a}\_{t+1}$ into the drafter's distribution $q := q(\cdot \mid \mathbf{a}\_{t+1})$ over $x_{t+2}$, and the drafter learns both to match the target's next hidden state and to predict that token:

$$\mathcal{L} = \operatorname{SmoothL}_1\big(\mathbf{a}_{t+1}, \mathbf{h}_{t+1}\big) + w_{\text{cls}} \cdot \mathcal{L}_{\operatorname{CE}}\big(q,\, x_{t+2}\big),$$

with $w_{\text{cls}} = 0.1$. The shifted tokens matter, since $\mathbf{h}\_t$ alone doesn't tell which token was sampled from it. At later draft steps the target's hidden states don't exist yet, so the drafter feeds back its own predictions $\mathbf{a}$. **EAGLE-2** ([Li et al., 2024b](https://arxiv.org/pdf/2406.16858)) kept the drafter and replaced the static token tree with a dynamic one, grown wherever the drafter is confident, since its confidence turned out to be well calibrated with acceptance.

### EAGLE-3

**EAGLE-3** ([Li et al., 2025](https://arxiv.org/pdf/2503.01840)) made two changes. It fuses low-, mid- and high-level hidden states of the target, $\mathbf{H} = \mathbf{W}\_{\text{fuse}}\,\big[\mathbf{h}^{(l_1)};\, \mathbf{h}^{(l_2)};\, \mathbf{h}^{(l_3)}\big]$, and it drops the hidden-state regression, training on tokens only. Its hidden states $\mathbf{a}$ then no longer have to look like the target's, so later draft steps must be part of training. The **training-time test** unrolls the drafter for several steps: $\mathbf{H}$ is available only for the verified context, and at every later position the drafter consumes its own hidden states from the previous step, exactly as at inference, while the tokens come from the training text:

$$\mathcal{L}_{\text{TTT}} = \sum_{k=1}^{\gamma} \mathcal{L}_{\operatorname{CE}}\Big(q_\phi\big(\cdot \mid x_{< t+k},\ \mathbf{H}_{\le t},\ \mathbf{a}_{t+1:t+k-1}\big),\; x_{t+k}\Big).$$

With these changes the drafter finally scaled with training data, and EAGLE-3 reported speedups of up to 6.5×. But it's still sequential: a round costs $\gamma$ drafter passes, and when verification isn't free this overhead can eat the whole gain. In [vLLM's throughput tests on AMD GPUs](https://vllm.ai/blog/2026-08-23-speculative-decoding-amd-gpus), where plain decoding of Qwen3-8B already ran at 3,530 tokens/s, i.e. with a large batch, EAGLE-3 on MATH500 stayed at 0.44× to 0.88× of plain decoding for every draft length from 1 to 7. Both numbers are right, just from opposite ends of the batch size range.

### Multi-token prediction

Another way to get a drafter is to train it with the target. **Multi-token prediction (MTP)** modules were popularized by DeepSeek-V3 ([DeepSeek-AI, 2024](https://arxiv.org/pdf/2412.19437)) as an auxiliary pretraining objective that densifies the training signal: the model gets $D$ extra modules applied in sequence. The $k$-th module normalizes the hidden state $\mathbf{a}^{(k-1)}\_i$ from the previous depth, starting from $\mathbf{a}^{(0)}\_i = \mathbf{h}\_i$, concatenates it with the normalized embedding of the token $k$ steps ahead, and passes the pair through its own FC layer and transformer layer:

$$\mathbf{a}^{(k)}_t = \operatorname{Layer}_k\Big(\operatorname{FC}_k\big[\operatorname{RMSNorm}(\mathbf{a}^{(k-1)}_i);\, \operatorname{RMSNorm}(e(x_{i+k}))\big]_{i \le t}\Big).$$

The shared LM head turns $\mathbf{a}^{(k)}\_t$ into $q_k := q(\cdot \mid \mathbf{a}^{(k)}\_t)$ over token $x_{t+k+1}$, so the $k$-th module looks $k+1$ tokens ahead. The cross-entropies of all depths are averaged and added to the usual next-token loss with a small weight:

$$\mathcal{L} = \mathcal{L}_{\operatorname{CE}}\big(p,\, x_{t+1}\big) + \frac{w_{\text{MTP}}}{D} \sum_{k=1}^{D} \mathcal{L}_{\operatorname{CE}}\big(q_k,\, x_{t+k+1}\big).$$

At inference the modules become a sequential drafter that comes with the model for free. DeepSeek-V4 serves with MTP-1, a single module drafting one token per round, and Qwen3.5 has native MTP layers. Kimi-K3 pre-trains one MTP layer and fine-tunes it into an EAGLE-3-style drafter that fuses the outputs of its 1st, 4th and final AttnRes blocks ([Kimi Team, 2026](https://github.com/MoonshotAI/Kimi-K3/blob/main/k3_tech_report.pdf)), which is where the two families meet. We'll get back to its training objective at the end of the post.

## Parallel drafters

Every drafter so far runs once per drafted token, which in our example already cost a fifth of every round. A **parallel drafter** proposes all $\gamma$ tokens in a single pass. With $n_{\text{draft}}$ drafter passes per round the speedup becomes

$$\text{speedup} = \frac{\tau}{n_{\text{draft}} \cdot c + 1}, \qquad n_{\text{draft}} = \begin{cases} \gamma & \text{autoregressive drafter} \\ 1 & \text{parallel drafter} \end{cases}$$

and with the same numbers a parallel drafter reaches $\frac{3.69}{0.05 + 1} \approx 3.51\times$, almost 20% faster at the same acceptance rate and draft length. The gap grows with $\gamma$: an autoregressive drafter has an optimal draft length beyond which extra guesses cost more than they bring, while for a parallel drafter longer drafts are nearly free, at least while verification stays memory-bound. The idea is as old as speculative decoding itself[^JAC], but for a long time parallel guesses were too weak to compete.

A word of caution about the numbers from here on. Papers differ in benchmarks, block sizes, temperatures and checkpoints, so for DFlash on Qwen3-8B alone the papers below report anywhere from about 4 to 7.8 tokens per round. Acceptance lengths are only comparable within one paper, so we'll always quote a method next to baselines from the same setting.

### Medusa

**Medusa** ([Cai et al., 2024](https://arxiv.org/pdf/2401.10774)) puts $\gamma$ cheap heads on the target's last hidden state $\mathbf{h}\_t$, the $k$-th guessing the token $k$ steps ahead:

$$q_k(\cdot \mid \mathbf{h}_t) = \operatorname{softmax}\Big(\mathbf{W}_2^{(k)}\big(\operatorname{SiLU}(\mathbf{W}_1^{(k)} \mathbf{h}_t) + \mathbf{h}_t\big)\Big), \quad k = 1, \dots, \gamma,$$

where $\mathbf{W}\_2^{(k)}$ is initialized from the target's LM head. The heads run on top of the target's own forward pass, so drafting is almost free, and their candidates form a static token tree. The catch is the first line of our factorization: head $k$ never sees what heads $1, \dots, k-1$ guessed, and a single hidden state is little context to guess far ahead. If the first head hesitates between "New" and "Los", the second can only spread its mass between "York" and "Angeles", and $\alpha_k$ decays quickly with $k$[^HYD].

### P-EAGLE

The most direct way to a strong parallel drafter is to take a good autoregressive one and stop feeding it its own guesses. **P-EAGLE** ([Hui et al., 2026](https://arxiv.org/pdf/2602.01469)) does this to EAGLE-3, filling every position after the first with two learned vectors: a mask token embedding for the unknown token and a shared hidden state for the missing one. The harder part is training: each position of a sequence of length $L$ now carries $\gamma$ prediction depths, and attention over $L\gamma$ positions needs $\mathcal{O}\big((L\gamma)^2\big)$ memory, prohibitive for the long outputs of reasoning models. P-EAGLE precomputes the depth-aware attention mask once and slices it per example, and splits long sequences into segments with gradient accumulation inside a sequence. Its acceptance length stays on par with EAGLE-3, but with one drafter pass instead of five it is 1.10–1.36× faster than EAGLE-3 in vLLM on gpt-oss-120B, gpt-oss-20B and Qwen3-Coder-30B. PARD ([An et al., 2025](https://arxiv.org/pdf/2504.18583)) applies the same trick to a separate small LM that fills several placeholder positions per pass, so that one drafter can serve a whole family of targets.

### DFlash

Filling placeholder positions in one pass is what masked diffusion language models do, refining their guesses over several such steps at generation time, and early work drafted with whole discrete diffusion models ([Christopher et al., 2024](https://arxiv.org/pdf/2408.05636)). **DFlash** ([Chen et al., 2026](https://arxiv.org/pdf/2602.06036)) calls itself a **block diffusion** drafter, but the name needs care: it's trained like a masked diffusion model, yet at inference one forward pass over the last token the target produced and a block of mask tokens predicts all masked positions, with no refinement. In effect it's a masked block predictor; iterative refinement only comes back later, in DEdit.

The block is embedded with the target's embedding table, which DFlash shares and keeps frozen together with the LM head, and its positions attend to each other bidirectionally. To predict well from masks, the drafter needs a strong grip on the context, so DFlash takes hidden states from five target layers, spread uniformly between the second and the third-to-last, fuses them into $\mathbf{H}$ as EAGLE-3 does, with an extra RMSNorm, and injects them as extra keys and values into *every* draft layer:

$$\mathbf{z}^{(j+1)} = \operatorname{Attn}\Big(\mathbf{Q} = \mathbf{W}_Q^{(j)} \mathbf{z}^{(j)},\; \mathbf{K} = \big[\mathbf{W}_K^{(j)} \mathbf{H}_{\le t};\, \mathbf{W}_K^{(j)} \mathbf{z}^{(j)}\big],\; \mathbf{V} = \big[\mathbf{W}_V^{(j)} \mathbf{H}_{\le t};\, \mathbf{W}_V^{(j)} \mathbf{z}^{(j)}\big]\Big),$$

where $\mathbf{z}^{(j)}$ are the hidden states of the block at layer $j$, starting from the embeddings. EAGLE-3 sees the fused states only at its input, so their influence fades with depth; in DFlash every layer attends to the target's context directly, which lets the drafter be deeper (5 layers) and still cheap, because it runs once. Training samples random positions in target-generated text as **anchors**, the last verified token of an imagined round, masks the block after each anchor, and applies cross-entropy with weights decaying exponentially along the block.

The most telling numbers come from a controlled comparison by [LMSYS](https://www.lmsys.org/blog/2026-06-15-next-generation-speculative-decoding-dflash-v2/): 5-layer EAGLE-3 and DFlash drafters for Qwen3-4B trained on the same data both reach $\tau = 4.2$ on GSM8K, yet the speedups are 2.1× and 3.3×. Only $n_{\text{draft}}$ moved. These drafters are weaker than the released DFlash checkpoint, which reaches 5.4 tokens per round on the same task even at temperature 1 in the DSpark paper, but they were trained identically. Adoption followed quickly: the DFlash drafter for Qwen3.5-397B-A17B beats the model's native MTP in throughput in every setting LMSYS benchmarked, by 1.5× on HumanEval at concurrency 1.

#### DDTree

A parallel drafter gives us the full distribution at every position of the block at once. **DDTree** ([Ringel and Romano, 2026](https://arxiv.org/pdf/2604.12989)) turns these per-position marginals into a token tree without any training. A path $\rho$ scores its log-probability under the drafter, $\log q(\rho) = \sum_i \log q_i(\rho_i)$, which simply adds up over independent marginals, so the classic best-first enumeration finds the top paths for a node budget. Each popped path pushes two successors, its best child one level deeper and its next-best sibling:

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

Scores only decrease along both kinds of edges, so paths leave the heap in order and every parent before its children. The tree on the verifier's side was built exactly this way, and here is how it grows with the budget:

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
	const opts = {mask: false, marginals: true, order: true};
	draw_token_tree(g, 7, opts);
	
	const budget_x = d3.scaleLinear()
	    .domain([1, 12])
	    .range([0, 330])
	    .clamp(true);
	createSlider(svg, function(budget) { draw_token_tree(g, budget, opts); }, budget_x, 130, 300, "budget", "currentColor", 7, roundN, 11, 60);
}

token_tree_budget();

</script>

![](.)
*Best-first token tree built from the per-position marginals of a parallel drafter, as in DDTree. Right: the top candidates $q_k$ at each block position $k$ from one drafter pass, shaded if they made it into the tree. Left: the tree, with nodes numbered in the order they leave the heap. Move the slider to change the budget: the tree grows deeper along confident paths and wider where the drafter hesitates.*

The tree is verified in one target pass with an ancestor-only mask, which FlashAttention-2 doesn't support, so DDTree falls back to PyTorch's generic scaled dot-product attention, and since the tree is deterministic, sampling is verified by token matching. On Qwen3-8B with MATH-500 it lifts the acceptance length from DFlash's 7.79 to 10.73 at temperature 0 and the speedup from 5.56× to 7.52×; at temperature 1 the speedup goes from 4.56× to 6.59×.

## Restoring dependency in the block

DFlash has the same catch as Medusa, one level up. Every position of the block is a marginal, so nothing ties the token at position $t+k$ to the one before it, and when the target is unsure between two continuations, independent sampling happily mixes them:

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
*A drafter unsure between "New York" and "Los Angeles" after "I flew from Paris to". Left: the probability of each pair of first two tokens, green if coherent, red if mixed up. Right: four sampled drafts. With independent marginals half of them mix the cities, get rejected at the second token and lose the rest of the block (faded); a causal head conditions each token on the previous one and keeps every draft coherent. Click the titles to switch.*

Marginals fail in a more mundane way too: nothing tells position $k$ that position $k-1$ already took a token, so neighbours often predict the same token twice. [Chen et al. (2026)](https://arxiv.org/pdf/2610.06011) call it the **repetition trap**, and blocks that fall into it get shorter accepted prefixes.

How much does the missing dependency cost? [Qiang et al. (2026)](https://arxiv.org/pdf/2608.27339) call it **parallel blindness** and split rejection in two. The **information floor** is the rejection no drafter can avoid with the information it has: seeing the context and only the last $m$ realized tokens before position $k$, the best it can do is

$$F_k^{(m)} = \mathbb{E}\Big[\min_{q}\ \mathbb{E}\big[\operatorname{TV}(p_k, q) \mid x_{\le t},\, x_{t+k-m:t+k-1}\big]\Big],$$

where the inner expectation averages over the unseen earlier tokens of the block and the best $q$ is chosen separately for every information state. A fully parallel drafter has $m = 0$, and with the whole prefix the floor vanishes. Everything above the floor is the drafter's **model gap**. The floors are estimated by sampling hundreds of target continuations per anchor and solving the inner minimization exactly.

On Qwen3-4B the all-parallel floor grows from 0.078 at the second position of a 7-token block to 0.286 at the seventh, capping any parallel drafter, however large, at about 71% acceptance there. But a single realized predecessor removes 86–100% of the floor, and a mutual information estimate agrees: one previous token recovers 94% of the missing path information, two tokens 99%. So a cheap head that sees the previous token is almost enough, and that's what the drafters in this section add to the parallel backbone. Yet real drafters sit far above their floors: across four targets, 43–64% of what the released DFlash checkpoints reject at the last position is model gap, not parallel blindness.

### DSpark

**DSpark** ([Cheng et al., 2026](https://arxiv.org/pdf/2607.05147)) from DeepSeek puts a light causal head on a DFlash-style backbone. Its default **Markov head** adds a low-rank transition bias to the backbone's logits $\boldsymbol{\ell}\_k$, so that position $k$ depends on the token drafted just before it:

$$q_k(\cdot \mid x_{\le t+k-1}) = \operatorname{softmax}\big(\boldsymbol{\ell}_k + \mathbf{W}_1[x_{t+k-1}]\,\mathbf{W}_2\big),$$

where the lookup table $\mathbf{W}\_1 \in \mathbb{R}^{V \times 256}$ and $\mathbf{W}\_2 \in \mathbb{R}^{256 \times V}$ form a rank-256 transition matrix. The bias ignores the context: "York" gets the same boost after "New" whether the text is about a city or a newspaper, and everything else has to come from the backbone. The backbone runs once, tokens are then sampled left to right, each picking its row of the bias, and the block's logits for verification are one extra matmul:

```python
def markov_logits(z, prev_tok, W_o, W1, W2):
    """z: [γ, d] backbone states, prev_tok: [γ] previous draft tokens,
    W1: [V, 256], W2: [256, V] - low-rank token transition."""
    return z @ W_o + W1[prev_tok] @ W2
```

Here `W_o` is the target's LM head. The sequential part is so light that scaling the draft from 4 to 16 tokens adds only 0.2–1.3% of latency. The alternative **RNN head** carries a gated recurrent state over the drafted tokens (the last line of our factorization) but gives only marginal gains. Here are the parallel architectures so far next to EAGLE-3:

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
*Four drafters in one schema: the target on the left, what the drafter reads from it in the middle, the drafter on the right with inputs at the bottom and draft tokens at the top. Medusa puts independent heads on the last hidden state next to the target's LM head. EAGLE-3 feeds a low, a middle and a high layer into one decoder layer and drafts a token per pass. DFlash injects five fused layers as keys and values into every draft layer and drafts a masked block in one pass. DSpark adds a Markov head that samples the block left to right. Click the titles to switch.*

The second half of DSpark is about the verification budget. A **confidence head** $\pi_k = \sigma\big(\mathbf{w}^T[\mathbf{z}\_k; \mathbf{W}\_1[x_{t+k-1}]]\big)$ predicts the acceptance rate of position $k$ given that its prefix survived, trained towards $\pi_k^\ast = 1 - \operatorname{TV}(p_k, q_k)$. The product $\prod_{i \le k} \pi_{b,i}$ estimates the chance that the first $k$ tokens of request $b$ survive, and a scheduler picks the draft lengths $\gamma_b$ of the whole batch to maximize accepted tokens per second:

$$\lbrace \gamma_b \rbrace^{*} = \arg\max_{\lbrace \gamma_b \rbrace} \ \underbrace{\sum_b \Big(1 + \sum_{k \le \gamma_b} \prod_{i \le k} \pi_{b,i}\Big)}_{\text{expected tokens per round}} \cdot \ \underbrace{\operatorname{SPS}\Big(\sum_b (\gamma_b + 1)\Big)}_{\text{rounds per second}},$$

where $\operatorname{SPS}$ (steps per second) is the engine's profiled number of rounds per second for a given number of tokens in the batch. Under light load every request drafts long; under heavy load verification becomes compute-bound and only confident prefixes get budget. The problem is solved greedily by sorting the survival probabilities of all candidate positions.

DSpark lifts the acceptance length over DFlash by 16–18% on Qwen3-4B/8B/14B, averaged over nine math, code and chat benchmarks at temperature 1 with blocks of 7 (on Qwen3-8B with GSM8K, from 5.33 to 6.17). In Qiang et al.'s terms the Markov head removes almost the whole floor: given the true previous token, 85–92% of what DSpark still rejects at the last position is model gap. In live DeepSeek-V4 serving against MTP-1, per-user generation became 60–85% faster for V4-Flash and 57–78% faster for V4-Pro at matched throughput, and aggregate throughput grew by about 50% at a fixed per-user speed target. That makes it the first parallel drafter proven in frontier production, running with $\gamma \le 5$.

### Domino

**Domino** ([Huang et al., 2026](https://arxiv.org/pdf/2605.29707)) gives the backbone a head with full-prefix memory, the last line of our factorization: a small GRU summarizes the embeddings of all tokens drafted so far into $\mathbf{s}\_{k-1}$, and a rank-256 correction $\mathbf{W}\_2\sigma(\mathbf{W}\_1[\mathbf{z}\_k; \mathbf{s}\_{k-1}])$ is added to the logits $\boldsymbol{\ell}\_k$. A head that sees the prefix tempts the optimizer to neglect the backbone, so Domino's base-anchored curriculum trains the backbone's own distribution $q_k^{\text{base}} := \operatorname{softmax}(\boldsymbol{\ell}\_k)$ first and gradually shifts the loss towards the corrected $q_k$:

$$\mathcal{L} = (1 - \beta)\, \mathcal{L}_{\operatorname{CE}}\big(q_k,\, x_{t+k}\big) + \beta\, \mathcal{L}_{\operatorname{CE}}\big(q_k^{\text{base}},\, x_{t+k}\big),$$

with $\beta$ annealed linearly from 1 to 0. On Qwen3-8B, averaged over eight benchmarks with greedy decoding and blocks of 16, it raises the acceptance length over DFlash from 6.06 to 7.17 and the speedup from 4.66× to 5.49×.

#### DominoTree

DDTree assumes the distribution at each depth doesn't depend on the path, which breaks with a causal head. **DominoTree** ([Lin and Jang, 2026](https://arxiv.org/pdf/2607.08642)) is a training-free best-first tree that scores each node with Domino's correction recomputed along its own path, restricted to the top-$M$ candidates per node. Applying the correction gives +10.1% acceptance length, recomputing it per path another +4.7%, and overall DominoTree reaches up to 7.3× speedup over autoregressive decoding on Qwen3-8B, on GSM8K at temperature 0. Like DDTree, it builds the tree deterministically and verifies by matching.

### DFlash2

**DFlash2** ([Inco AI, 2026](https://inco.ai/blog/dflash2/)) selects among the candidates the backbone already proposes instead of rewriting each position's full distribution. Its diagnosis, on Inco's own five-layer DFlash drafter for Qwen3-4B on GSM8K at temperature 0: the correct token is among the top 16 candidates 99.5% of the time at the first position but only 87.8% at the seventh; the share of attention a position spends on the rest of the block falls from 30% in the first layer to 8% in the fifth; and an oracle picking the best path through the top-16 candidates would reach an acceptance length of 6.79 instead of 4.27. So DFlash2 changes two things, both easy to read in the [vLLM implementation](https://github.com/vllm-project/vllm/pull/52816): how positions exchange information inside the backbone, and how the draft is read off its output.

<div id="dflash2_architecture" class="svg-container" align="center"></div> 

<script>

function dflash2_architecture() {
	const svg = d3.select("#dflash2_architecture")
				  .append("svg")
				  .attr("viewBox", "0 0 700 390")
				  .style("width", "100%")
				  .style("max-width", "700px")
				  .style("height", "auto");
	
	const g = svg.append("g");
	const col_x = function(k) { return 190 + k * 86; };
	const w = 70, block_w = 4 * 86 + w, label_x = 612;
	const col_c = function(k) { return col_x(k) + w / 2; };
	const out_y = 44, head_y = 150, mlp_y = 194, attn_y = 248, in_y = 300;
	const cand_y = function(r) { return 82 + r * 20; };
	const conv_rows = [186, 222, 236, 280];
	const candidates = [
		[["Los", 0.46], ["New", 0.44], ["San", 0.05]],
		[["York", 0.41], ["Angeles", 0.38], ["City", 0.08]],
		[["last", 0.62], ["this", 0.20], ["on", 0.09]],
		[["week", 0.70], ["month", 0.12], ["year", 0.08]]
	];
	const target = ["Los", "Angeles", "last", "week"];
	const paths = [[0, 0, 0, 0], [0, 1, 0, 0]];
	const cw = 56, cand_x = function(k) { return col_x(k) + (w - cw) / 2; };
	
	function box(x, y, bw, bh, label, color, size=11) {
		rect(g, x, y, bw, bh, color);
		text_(g, label, x + bw / 2, y + bh / 2 + size / 3, size, "black").attr("text-anchor", "middle");
	}
	
	function curve(x0, y0, x1, y1, width, opacity) {
		const mx = (x0 + x1) / 2;
		g.append("path")
		   .attr("d", "M " + x0 + " " + y0 + " C " + mx + " " + y0 + ", " + mx + " " + y1 + ", " + x1 + " " + y1)
		   .attr("fill", "none")
		   .attr("stroke", "currentColor")
		   .attr("stroke-width", width)
		   .attr("opacity", opacity);
	}
	
	function draw(mode) {
		g.selectAll("*").remove();
		const path = paths[mode];
		const tokens = path.map(function(r, j) { return candidates[j][r][0]; });
		let accepted = 0;
		while (accepted < 4 && tokens[accepted] == target[accepted]) { accepted += 1; }
		
		line_(g, col_c(0), in_y, col_c(0), conv_rows[0], "currentColor", 1);
		for (let k = 1; k < 5; k += 1) {
			line_(g, col_c(k), in_y, col_c(k), cand_y(2) + 16, "currentColor", 1);
			line_(g, col_c(k), cand_y(path[k - 1]), col_c(k), out_y + 22, "currentColor", 1, "2,2");
		}
		
		["to", "[M]", "[M]", "[M]", "[M]"].forEach(function(token, k) {
			token_box(g, col_x(k), in_y, w, token, k == 0 ? context_color : 'white', 1.0, 11);
		});
		text_(g, "last verified token", 15, in_y + 10, 11);
		text_(g, "+ mask tokens", 15, in_y + 25, 11);
		
		box(col_x(0), attn_y, block_w, 24, "attention over the target's K, V and the whole block", drafter_color);
		rect(g, 15, attn_y, 135, 24, fuse_color);
		text_(g, "target states H", 82, attn_y + 16, 11, "black").attr("text-anchor", "middle");
		arrow_(g, 150, attn_y + 12, col_x(0) - 2, attn_y + 12);
		text_(g, "K, V", 156, attn_y + 4, 10);
		text_(g, "attention", label_x, attn_y + 16, 11);
		for (let k = 0; k < 5; k += 1) {
			box(col_x(k), mlp_y, w, 22, "MLP", drafter_color);
		}
		text_(g, "MLP", label_x, mlp_y + 15, 11);
		
		if (mode == 1) {
			conv_rows.forEach(function(y) {
				for (let k = 0; k < 5; k += 1) {
					rect(g, col_x(k), y, w, 6, head_color);
					if (k > 0) {
						arrow_(g, col_x(k - 1) + w + 1, y + 3, col_x(k) - 1, y + 3);
					}
				}
				text_(g, "conv", label_x, y + 7, 10);
			});
		}
		line_(g, 680, conv_rows[0], 680, conv_rows[3] + 6, "currentColor", 1);
		line_(g, 676, conv_rows[0], 680, conv_rows[0], "currentColor", 1);
		line_(g, 676, conv_rows[3] + 6, 680, conv_rows[3] + 6, "currentColor", 1);
		text_(g, "×5", 684, (conv_rows[0] + conv_rows[3]) / 2 + 8, 11);
		
		box(col_x(1), head_y, 3 * 86 + w, 20, "target's LM head", target_color);
		text_(g, "LM head", label_x, head_y + 14, 11);
		
		token_box(g, col_x(0), cand_y(1) - 3, w, "to", context_color, 1.0, 11);
		if (mode == 1) {
			candidates.forEach(function(column, j) {
				const k = j + 1;
				column.forEach(function(cand, r) {
					const sources = j == 0 ? [col_x(0) + w] : candidates[j - 1].map(function() { return cand_x(k - 1) + cw; });
					sources.forEach(function(x0, ra) {
						const y0 = j == 0 ? cand_y(1) + 8 : cand_y(ra) + 8;
						const chosen = r == path[j] && (j == 0 || ra == path[j - 1]);
						curve(x0, y0, cand_x(k), cand_y(r) + 8, chosen ? 2.5 : 1, chosen ? 1.0 : 0.15);
					});
				});
			});
		}
		candidates.forEach(function(column, j) {
			const k = j + 1;
			column.forEach(function(cand, r) {
				const picked = r == path[j];
				rect(g, cand_x(k), cand_y(r), cw, 16, draft_token_color, picked ? 1.0 : 0.5).attr("stroke-width", picked ? 2 : 1);
				text_(g, cand[0], cand_x(k) + 3, cand_y(r) + 12, 9, "black");
				text_(g, cand[1].toFixed(2).substring(1), cand_x(k) + cw - 3, cand_y(r) + 12, 8, "black").attr("text-anchor", "end");
			});
			const color = j < accepted ? green : (j == accepted ? red : draft_token_color);
			token_box(g, col_x(k), out_y, w, tokens[j], color, j > accepted ? 0.45 : 1.0, 11);
		});
		text_(g, mode == 0 ? "argmax" : "selector", label_x, cand_y(1) + 4, 11);
		text_(g, mode == 0 ? "per slot" : "walk", label_x, cand_y(1) + 18, 11);
		text_(g, "draft", label_x, out_y + 15, 11);
		
		const lines = mode == 0 ?
			["positions exchange information only through attention and are decoded on their own:",
			 "“Los York” mixes two cities, and the target accepts 1 of 4 tokens"] :
			["two-tap convolutions around every sublayer read the left neighbour, and the selector",
			 "walks from “to” through the top candidates: the target accepts all 4 tokens"];
		lines.forEach(function(line, i) {
			text_(g, line, 350, 350 + i * 17, 12).attr("text-anchor", "middle");
		});
	}
	
	tabs_(svg, ["DFlash", "DFlash2"], 290, 22, draw);
}

dflash2_architecture();

</script>

![](.)
*One draft pass of DFlash and DFlash2 after "I flew from Paris to", when the target continues with "Los Angeles last week" (norms and residual connections left out). Both read the fused target states as keys and values in every layer. DFlash decodes every position on its own and mixes the two cities. DFlash2 wraps every attention and MLP sublayer in two-tap convolutions (pink) and walks through the top candidates from the last verified token. Click the titles to switch.*

**Convolutions inside the backbone.** Since the block barely uses attention to look at itself, DFlash2 adds a hard-wired channel to the left neighbour, a two-tap depthwise **dynamic convolution**

$$\operatorname{Conv}(\mathbf{z})_k = \big(\mathbf{b}_0 + \boldsymbol{\delta}_{k,0}\big) \odot \mathbf{z}_k + \big(\mathbf{b}_1 + \boldsymbol{\delta}_{k,1}\big) \odot \mathbf{z}_{k-1},$$

where $\mathbf{b}\_0, \mathbf{b}\_1$ are learned per-channel kernels and $\boldsymbol{\delta}\_k$ is a linear projection of $\mathbf{z}\_k$, one number per group of 16 channels. Each attention and MLP sublayer is wrapped in two of them, on its normalized input and on its output before the residual connection, with the coefficients of both predicted from the sublayer's input. Twenty convolutions in a row carry a signal across the whole block. They run over the block only, with the tap zeroed at its boundary, so the first mask position reads the last verified token, while the target's context still arrives as keys and values. They add 16.5M parameters, 3% of the drafter, recover much of the gain of a 15-layer drafter at late positions, and take the local exchange over from attention: layers 4–5 now spend 0.5% of their attention inside the block instead of 9.4%.

**A selector instead of an argmax.** DFlash2 keeps the top $K = 16$ candidates $C_k$ at each position with their logits $\boldsymbol{\ell}\_k[b]$ and scores every adjacent pair $a \in C_{k-1}$, $b \in C_k$:

$$S_k(a, b) = \boldsymbol{\ell}_k[b] + \big\langle \mathbf{A}[a] \odot \mathbf{P}\mathbf{z}_k,\ \mathbf{B}[b] \big\rangle,$$

where $\mathbf{A}$ and $\mathbf{B}$ are 256-dimensional token codebooks for the predecessor and the successor and $\mathbf{P}$ projects $\mathbf{z}\_k$ to 256 dimensions. One batched product yields a $16 \times 16$ matrix per position, with no extra backbone or LM head pass. The draft is a walk through this lattice: starting from the last verified token, each position takes the row $S_k(a, \cdot)$ of the token chosen before and picks its argmax, or samples from its softmax at the sampling temperature. In vLLM the walk is one small Triton program per request, and lattice plus walk take 0.02 ms. The verifier receives exactly this softmax over the 16 candidates, zero elsewhere, as $q_k$, so the truncation is part of the proposal and the output stays exact.

Unlike DSpark's head, the transition is gated by the context through $\mathbf{z}\_k$, so "York" after "New" can be boosted for a city and not for a newspaper, and it's evaluated on $16 \times 16$ pairs instead of the full vocabulary. In Inco's Qwen3-4B ablation the selector costs 2.0M parameters and 0.6% of round latency against 77.8M and 9.6% for DSpark's correction, and lifts the acceptance length from 4.27 to 4.61 at temperature 0 and from 3.78 to 4.25 at temperature 1, against 4.49 and 4.08 with DSpark's correction. The price is that no path can leave the top 16, one more reason for the convolutions to keep recall high.

Altogether it's a parallel backbone with a head that sees only the previous token, order one in Qiang et al.'s terms, for 18.5M parameters and about 1.3% of round latency. On Qwen3.5-4B with thinking on and temperature 1, the mean acceptance length over five benchmarks grows from 4.92 with DFlash and 5.49 with DSpark to 5.97. For Qwen3.8-27B with blocks of 8 it reaches 4.80 against 4.28 for the native MTP and 2.7–3.4× the throughput of autoregressive decoding at batch size 1. In vLLM, against a community DSpark drafter with 7 draft tokens on GSM8K, it serves 3.51× the autoregressive throughput against 2.79× at concurrency 1, and 2.20× against 1.77× at concurrency 32.

Qiang et al. push the selector further. At order one there is almost no floor left, so what remains is model gap, and they argue that a context-blind transition added to the logits, as in DSpark, limits what such a head can express. Their **prefix-attention head** builds a query from the predecessor's embedding and the drafter state, attends over the committed prefix and maps the result through a low-rank projection to predecessor-dependent logits, on top of DFlash2's lattice and convolutions. Its floor is DSpark's, but its model gap at the last position is 25.5% lower, and in SGLang their Qwen3-4B drafter accepts 4.94 tokens per round at temperature 1 against 4.75 for the released DSpark checkpoint, averaged over nine tasks.

### Looping and editing the block

A parallel drafter is cheap enough to run twice: with $c = 0.05$, two passes still cost a tenth of a target pass. So instead of adding a head, we can loop the drafter over its own block, the Jacobi-style refinement[^JAC] done by a drafter rather than by the target. Such drafters don't fit a single line of our factorization, since every extra pass conditions the whole block on the previous guess.

**D-Loop** ([Chen et al., 2026](https://arxiv.org/pdf/2610.06011)), from the authors who named the repetition trap, runs the DFlash drafter twice: the first pass proposes a block, the second conditions on a selected prefix of it and regenerates the suffix in parallel. The block becomes two semi-autoregressive halves without a single new parameter, and one drafter learns both roles with a prefix–suffix objective. On Qwen3-4B and Qwen3-8B it beats DFlash and DSpark trained in the same setting, most of all when sampling: averaged over eight benchmarks at temperature 1, it accepts 5.77 tokens per round on Qwen3-8B against 4.20 for DSpark, and the speedup grows from 3.19× to 4.03×.

**DEdit** ([Yu et al., 2026](https://arxiv.org/pdf/2609.38510)) lets the DFlash drafter edit its whole draft: after the first unmasking pass, each editing pass takes the current draft instead of masks and re-predicts every position at once, with bidirectional attention inside the block, so later guesses can repair earlier errors. The hard part is repairing without breaking correct tokens. ProposalMix training builds editing inputs by mixing the drafter's first-pass predictions with ground-truth tokens according to its confidence, which halves the edits that shorten the accepted prefix. With two editing passes over a block of 32, DEdit reaches 7.58 tokens per round on Qwen3-4B with greedy decoding against 6.91 for DSpark trained on the same data with a block of 16, and average greedy speedups of 5.72× and 5.97× on Qwen3-4B and Qwen3-8B. Every editing pass costs a drafter pass, so the speedup grows less than the acceptance length, and a causal editor accepts less: the future context is what makes editing work.

## Beyond a single round

So far every round has been self-contained: draft, verify once, throw away the rejected part and start from scratch. The newest drafters break these habits.

### Drafting past the verifier

With strong drafters the target often accepts every drafted token, so verifying after each drafting stage is often unnecessary. Adaptive draft lengths decide how much to draft before verifying: SpecDec++ ([Huang et al., 2024](https://arxiv.org/pdf/2405.19715)) trains an acceptance prediction head for an autoregressive drafter and stops once the predicted chance of a rejection crosses a threshold, and DSpark's scheduler chooses lengths for a whole batch, but only within one block. Going beyond a block is harder for a parallel drafter, whose next block needs target hidden states for tokens that aren't verified yet. **DLoop** ([Gu et al., 2026](https://arxiv.org/pdf/2610.07659)), not to be confused with D-Loop above, keeps drafting while the drafter stays confident, feeding it its own hidden states in place of the missing target ones, and verifies all accumulated tokens in one pass. Loop-aware training exposes the drafter to these self-generated states so that it stays reliable in later stages. DLoop trades a few cheap drafter passes for fewer expensive verifications and improves walltime by 5 to 41% across EAGLE-3, DFlash, Domino, DSpark and MTP drafters. PEARL ([Liu et al., 2024](https://arxiv.org/pdf/2408.11850)) attacks the same idle time from the other side, drafting the next tokens while the target is still verifying the current ones, so that neither model waits for the other.

### Recycling rejected work

Verification also computes target hidden states for rejected tokens, and the rejected part of a draft often still contains the right tokens a little later. Token Recycling ([Luo et al., 2024](https://arxiv.org/pdf/2408.08696)) was a training-free version of this idea, storing candidate tokens from earlier steps in an adjacency matrix and growing the next draft tree from them. Two recent methods recycle much richer state into parallel drafters.

**ReTrace** ([Lin et al., 2026](https://arxiv.org/pdf/2608.29748)) shows that rejected positions of a DFlash draft often still align with the target's continuation, and conditions the next block on the rejected suffix instead of fresh mask tokens. It keeps the suffix's hidden representations, aligns them with the new block, corrects them with signals from the same verification pass and fuses them into the drafter's input embeddings through a gated residual connection, without an extra forward pass.

**Carryover drafting** ([Koo et al., 2026](https://arxiv.org/pdf/2609.14717)) recycles the target's side: hidden states the verifier computed for rejected tokens become temporary KV context for the drafter, marked with a single learned embedding and replaced every round, so their size stays bounded by one block. Standard parallel training never produces rejected states like the ones at inference, so Carryover's draft–verify–draft training produces them while keeping all positions in parallel. With DFlash and a DSpark-style drafter it raises the acceptance length by 6.5–14.7% and the vLLM speedup by 7.9–14.4%, and by up to 28.8% on translation.

Both stay lossless, because rejected tokens are never committed, they only inform the next draft. And both change what a drafter keeps between rounds, which brings us to memory.

## Drafters and the KV cache

So far we've counted only the weights a drafter reads. With long contexts and large batches the KV cache takes over: MagicDec ([Sadhukhan et al., 2024](https://arxiv.org/pdf/2408.11049)) showed that decoding then becomes bound by reading the KV cache, which grows with both the sequence length and the batch size, and that speculative decoding keeps paying off even at batch sizes from 32 to 256, as long as the drafter's own KV cache stays small.

DFlash and the drafters built on it keep their own KV cache: keys and values projected from $\mathbf{H}$ in every draft layer, for the whole context. A real world example: a DFlash-style drafter for Qwen3-8B has 5 layers with the target's layout of 8 KV heads of dimension 128, so one request with 32k tokens of context in bf16 needs

$$\underset{\mathbf{K/V}}{2} \cdot \underset{\text{bf16}}{2} \cdot \underset{\text{sequence length}}{32{,}768} \cdot \underset{\text{draft layers}}{5} \cdot \underset{\text{KV heads}}{8} \cdot \underset{\text{head dim}}{128} = \underset{\text{bytes per request}}{671{,}088{,}640}.$$

That's 0.67 GB, only 14% on top of the target's own 36-layer cache, but 128 such requests need 86 GB for the drafter alone, more than the whole HBM of an H100. Hybrid targets fare worse: Qwen3.5-397B-A17B uses linear attention in most layers, so its own cache is small, and a drafter cache inflates the total by 1.8×.

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
	text_(svg, "full context", x(128) + 8, y(128 * per_request) + 4, 12, drafter_color);
	const per_request_swa = 2 * 2 * 2048 * 5 * 8 * 128 / 1e9;
	line_(svg, x(1), y(per_request_swa), x(128), y(128 * per_request_swa), drafter_color, 2, "5,3");
	text_(svg, "2,048-token window", x(128) + 8, y(128 * per_request_swa) + 4, 12, drafter_color);
	
	[16 / per_request, 80 / per_request].forEach(function(n) {
		svg.append("circle").attr("cx", x(n)).attr("cy", y(n * per_request)).attr("r", 4)
		   .attr("fill", drafter_color).attr("stroke", "black");
		text_(svg, Math.round(n) + " requests", x(n) - 75, y(n * per_request) - 6, 11);
	});
}

drafter_memory();

</script>

![](.)
*Memory taken by the KV cache of a DFlash-style drafter (5 layers, 8 KV heads of dimension 128, bf16) for Qwen3-8B at 32k context versus the number of concurrent requests. Attending to the full context, it overtakes the target's weights at 24 requests and the whole HBM of an H100 at 119. With a 2,048-token sliding window (dashed) it stays at 5.4 GB even with 128 requests.*

### Sliding windows

The most straightforward fix is the one long-context LLMs use themselves: a sliding window. If each draft layer attends only to the last $W$ positions, the cache stops growing with the context, and with $W = 2{,}048$ our drafter needs

$$\underset{\mathbf{K/V}}{2} \cdot \underset{\text{bf16}}{2} \cdot \underset{\text{window}}{2{,}048} \cdot \underset{\text{draft layers}}{5} \cdot \underset{\text{KV heads}}{8} \cdot \underset{\text{head dim}}{128} \approx \underset{\text{bytes per request}}{42 \text{ MB}},$$

16 times less at 32k tokens and 500 times less at 1M. MagicDec itself drafted this way, running the target over a StreamingLLM-style window of recent tokens and a few attention sinks, and parallel drafters have converged on it too: z-lab trains its recent DFlash drafters, such as the [one for Qwen3.5-27B](https://huggingface.co/z-lab/Qwen3.5-27B-DFlash), with sliding-window attention "for improved long-context performance", Red Hat's [DFlash drafter for DeepSeek-V4-Flash](https://huggingface.co/RedHatAI/DeepSeek-V4-Flash-speculator.dflash) uses 2,048-token windows in all five layers, and H-Spec below runs its attention and all its baselines with the same window.

The window costs less acceptance than one might expect. The keys and values in it are projected from target hidden states that have already attended over the whole context, and drafters mostly look at recent tokens anyway: [Eldenk et al. (2026)](https://arxiv.org/pdf/2605.09992) observe that the attention of EAGLE-3 and MTP drafters drifts from the prompt towards the tokens they've just drafted. Windowed-MTP ([Valliappan, 2026](https://arxiv.org/pdf/2607.21535)) puts 4,032 recent tokens and 64 sinks in the window of the native MTP drafters of hybrid Qwen models at 1M tokens of context. On Qwen3.6-35B the acceptance length even grew over full-context drafting, from 4.49 to 4.74, and the time per output token fell by 1.57×; only a 1K window started to hurt, with 3.79.

So for memory alone a window is mostly enough. It doesn't remove the drafter's cache as a separate structure: every new token still has to be projected into keys and values in every draft layer and written into a second cache pool with its own layout, which the engine has to allocate, page and roll back (Windowed-MTP needs a ring buffer and a remapping step to actually reclaim the memory). On Qwen3-8B this injection step gets 4.7× slower going from 4 to 128 concurrent requests. And it leaves open another question: could the drafter get *more* out of the target's own cache than out of the few hidden states it projects into its own?

### Other ways to save memory

Before windows became the default for parallel drafters, three other ways to keep the drafter's memory in check appeared.

**Shrink the drafter's cache.** TriForce ([Sun et al., 2024](https://arxiv.org/pdf/2404.11912)) drafts with the target itself over a small, retrieval-based selection of its KV cache, SpecExtend ([Cha et al., 2025](https://arxiv.org/pdf/2505.20776)) uses the target's attention scores to decide which part of the context a small drafter keeps, without retraining, and QuantSpec ([Tiwari et al., 2025](https://arxiv.org/pdf/2502.10424)) drafts with the target over a hierarchical 4-bit quantized KV cache and 4-bit weights, accepting over 90% of drafted tokens. [Yuan et al. (2026)](https://arxiv.org/pdf/2608.30252) give a strong drafter a compressed memory that a small adaptor builds and updates incrementally, cutting drafter-side memory by over 70% at 32k contexts.

**Reuse the target's cache.** Self-speculative decoding drafts with a part of the target: Draft & Verify ([Zhang et al., 2023](https://arxiv.org/pdf/2309.08168)) skips some intermediate layers, and LayerSkip ([Elhoushi et al., 2024](https://arxiv.org/pdf/2404.16710)) trains the target with early exits and drafts with its first layers, whose cache and activations verification then reuses. LongSpec ([Yang et al., 2025](https://arxiv.org/pdf/2502.17421)) gives a small drafter self-attention over a 512-token sliding window and cross-attention to the target's KV cache, so its own cache stays constant however long the context. Gemma 4's MTP assistants go all the way: their decoder layers have only query projections and read keys and values straight from the target's cache.

**Keep a recurrent state.** ReDrafter ([Cheng et al., 2024](https://arxiv.org/pdf/2403.09919)) drafts with a small RNN conditioned on the target's last hidden state, and Mamba drafters ([Choi et al., 2025](https://arxiv.org/pdf/2506.01206)) replace the drafter's transformer with a state space model. OWL ([Lee et al., 2025](https://arxiv.org/pdf/2510.07535)) drafts with an LSTM conditioned only on the last token's state, and on long inputs, where EAGLE-3 slows decoding down to 0.81×, reaches about 5 times EAGLE-3's acceptance length.

### The target's cache as a source of information

The target's KV cache holds keys and values from every layer for every position, far more than the three or five hidden states a drafter usually reads, and some drafters read it purely for acceptance. GliDe ([Du et al., 2024](https://arxiv.org/pdf/2402.02082)) keeps a small drafter with its own cache and adds a cross-attention layer that "catches a glimpse" of the target's cache so the drafter behaves more like the target, raising the acceptance rate by up to 23.5%. Mixture of Attentions ([Zimmer et al., 2024](https://arxiv.org/pdf/2410.03804)) blames EAGLE's limits on partial observability, since the last hidden states are a lossy function of the target's full state. Its Layer Self-Attention attends across the keys and values of all target layers for each verified token and averages them into one vector per token for the drafter to cross-attend to; the full model accepts 25% more tokens per round than EAGLE-2, though the layer attention itself contributes only a modest part. Both are autoregressive and use the target's cache on top of what they already had. A parallel drafter that drops its own cache must rely on the target's cache instead, a stronger bet.

### H-Spec

**H-Spec** ([Jiang et al., 2026](https://arxiv.org/pdf/2609.24197)) makes that bet, combining two of the older answers: it reads the target's keys and values in place and keeps a recurrent state rebuilt every round, and it gains acceptance rather than paying for the memory with it. It starts from the obvious fix: let each DFlash layer attend to one target layer's keys and values directly, like GliDe. The first two positions of the block stayed as good as before, but conditional acceptance at later ones fell by up to 16.5 percentage points. Keys and values are made for retrieval, each describing one position for future queries, and none summarizes the context as a whole, as the projected hidden states did.

A constant-size summary is already there: under causal attention the target's last hidden state $\mathbf{h}\_t$ has seen the whole context, and every verification pass computes it. To consume one vector per round, H-Spec turns to state space models. The recurrent form of [linear attention](https://astralord.github.io/posts/transformer-inference-optimization-toolset/#linear-attention) from the toolset post folds the whole prefix into one matrix, $\mathbf{U}\_i = \mathbf{U}\_{i-1} + \phi(\mathbf{k}\_i)\mathbf{v}\_i^T$, and **Mamba-2** ([Dao and Gu, 2024](https://arxiv.org/pdf/2405.21060)) adds a learned, input-dependent forgetting factor $a_i \in (0, 1)$:

$$\mathbf{U}_i = a_i\, \mathbf{U}_{i-1} + \mathbf{k}_i \mathbf{v}_i^T, \qquad \mathbf{o}_i = \mathbf{U}_i^T \mathbf{q}_i,$$

where $a_i$, $\mathbf{q}\_i$, $\mathbf{k}\_i$ and $\mathbf{v}\_i$ are computed from the current input (Mamba calls them $\bar{A}$, $C$, $B$ and $x$). Where attention keeps and reads a key and a value for every past token, a Mamba layer keeps only $\mathbf{U}$, whose size doesn't depend on the context, and decides what to forget. That's how hybrid targets like Qwen3.5 get their small caches, and why they still keep some full attention layers: the summary is lossy and can't retrieve an arbitrary token exactly.

H-Spec gives these two jobs to two modules, both fed by the target for free. Each of its four layers starts with a Mamba-2 module, and three follow it with attention. The Mamba modules start from a state $\mathbf{U}\_0$ projected from the fused last-position hidden states of five target layers and scan only the block, in one parallel pass since the recurrence is linear; identical mask tokens still get different states at different positions. The attention modules match the target's KV layout and read three target layers' keys and values in place, through a 2,048-token window, for the token-level detail a summary can't hold. DSpark's Markov head sits on top. For Qwen3-8B the Mamba state holds 48 heads × 64 × 4 = 12,288 numbers per layer and is rebuilt every round, so nothing grows with the context and nothing has to be injected.

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
			text_(g, "drafter keeps a cache of its own:", x0, 52, 12);
			text_(g, "20 KiB per token (Qwen3-8B), growing with the context", x0, 70, 12);
			text_(g, "unless the drafter attends through a window", x0, 88, 12);
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
*How a parallel drafter conditions on a context of $L$ tokens. DFlash writes the target's projected hidden states into its own cache for every new token, a cache that grows with the context unless windowed. H-Spec's attention reads the target's KV cache in place, and its Mamba layers start from a fixed-size state built from $\mathbf{h}\_t$ and scan the block in one parallel pass. Move the slider to grow the context, and click the titles to switch.*

Both halves matter: with the architecture fixed, dropping the target's keys and values cuts the acceptance length by 24–31%, and dropping the summary by 7–12%. At a matched parameter count the hybrid backbone beats a Mamba-only drafter by 29–45% and an attention-only one by 13–15%. Against P-EAGLE, DFlash and DSpark retrained with the same recipe, H-Spec improves the acceptance length on Llama-3.1-8B, Qwen3-4B and Qwen3-8B by 13.3%, 5.0% and 8.7%, keeping its margin from 1k to 32k tokens of context. All of them use the same 2,048-token window, so these gains are what the target's cache and summary add as information. In vLLM H-Spec reaches 5–17% higher peak throughput at the lowest KV cache utilization, with an advantage growing with concurrency. The costs are a deeper computational graph, which drafts slower than DFlash below about 32 concurrent requests or 512 tokens of input, and attention layers that must copy the target's KV layout.

Serving engines have converged on a handful of these families: vLLM's benchmark on AMD GPUs compares exactly four `method` values of `--speculative-config`, namely `mtp`, `eagle3`, `dflash` and `dspark`. Across all of them the architecture has been moving the same way, from drafting step by step towards a parallel backbone, a small causal head and deep conditioning on the target, built with the serving engine in mind[^SV2]. What's left is the objective.

## Training the drafter

Recall that the speedup is governed by the acceptance length, and the acceptance length by $\alpha = 1 - \operatorname{TV}(p, q)$. Yet EAGLE-3, MTP, DFlash and Domino are all trained with cross-entropy, i.e. towards a KL objective, and newer drafters are moving away from it: DSpark and H-Spec put 90% of their loss on a total variation term, and Kimi-K3 drops KL altogether.

### KL is only a proxy

The usual recipe is cross-entropy on target-generated tokens, or forward KL to the target's full distribution, the same objective with a denser signal. KL reaches zero exactly when $q = p$, where $\alpha = 1$, but this only helps if the drafter family $\mathcal{Q}$ contains $p$. A drafter typically has 1–5% of the target's parameters and sees it only through a few hidden states, so in general it can't represent $p$ exactly, and then the two objectives can pick different solutions:

$$q_{\text{KL}} = \arg\min_{q \in \mathcal{Q}} \operatorname{KL}(p \,\Vert\, q) \ \ne \ q_{\alpha} = \arg\max_{q \in \mathcal{Q}} \sum_x \min\big(p(x), q(x)\big).$$

Pinsker's inequality,

$$1 - \alpha \le \sqrt{\tfrac{1}{2} \operatorname{KL}(p \,\Vert\, q)},$$

guarantees a high $\alpha$ for a low KL, but not which $q \in \mathcal{Q}$ has the highest $\alpha$. The objectives disagree on modes they can't fit. Forward KL is mass-covering: it punishes $q(x) \approx 0$ wherever $p(x) > 0$, so a drafter with limited capacity spreads over all modes of the target. Reverse KL, $\operatorname{KL}(q \,\Vert\, p)$, is mode-seeking: it punishes mass where $p$ has little, so the drafter collapses onto a single mode. The acceptance rate only rewards overlap. That TV, not KL, is the divergence tied to acceptance isn't new: DistillSpec ([Zhou et al., 2023](https://arxiv.org/pdf/2310.08461)) already compared it with KL variants for distilling drafters. Here's a toy example with a mixture of three Gaussians as the target and a single Gaussian as the drafter:

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

The forward KL optimum spreads over all three modes and accepts 42% of tokens, the reverse KL optimum sits on the widest mode and accepts 40%, and the $\alpha$-optimal drafter partially covers the two left modes and accepts 54%. Same family, same capacity, and neither KL finds it.

So why not maximize $\alpha$ directly? Look at the gradients with respect to the drafter's logits $\ell$:

$$\frac{\partial \alpha}{\partial \ell_y} = \frac{q(y)}{2}\Big(s(y) - \sum_x q(x)\, s(x)\Big), \qquad \frac{\partial \operatorname{KL}(p \,\Vert\, q)}{\partial \ell_y} = q(y) - p(y),$$

where $s(x) = \operatorname{sign}\big(p(x) - q(x)\big)$. The gradient of $\alpha$ is scaled by $q(y)$, so the tokens where the drafter puts little mass, exactly the ones it needs to learn, barely get any push, while KL pulls wherever $p$ has mass and $q$ doesn't. The figure starts in this regime: $q$ sits between two modes, $\alpha \approx 0.005$ and small moves of $\mu$ or $\sigma$ barely change it, while KL reacts to every move. A real drafter starts spread over $10^5$ tokens or more, where the TV gradient is tiny, and in our experiments drafters trained on TV alone ended up well behind KL.

### LK losses

In our recent paper we introduced **LK losses** ([Samarin et al., 2026](https://arxiv.org/pdf/2602.23881)), which target the acceptance rate directly; the name comes from Leviathan et al.'s $D_{LK}$. The first is a hybrid whose weight moves from KL to TV as the drafter improves:

$$\mathcal{L}_{\text{LK}}^{\lambda} = \lambda\, \operatorname{KL}(p \,\Vert\, q_\phi) + (1 - \lambda)\, \operatorname{TV}(p, q_\phi), \qquad \lambda = \exp\big(-\eta \cdot \operatorname{sg}[\alpha]\big),$$

where $\operatorname{sg}$ is a stop-gradient and $\alpha$ is the acceptance rate at the given draft position, averaged over the batch and the sequence. While the drafter is poor, $\lambda \approx 1$ and it gets KL's dense gradients; as $\alpha$ grows, the weight shifts to the acceptance rate itself. The second is the likelihood of acceptance:

$$\mathcal{L}_{\text{LK}}^{\alpha} = -\log \alpha = -\log \sum_x \min\big(p(x), q_\phi(x)\big), \qquad \nabla \mathcal{L}_{\text{LK}}^{\alpha} = \frac{1}{\alpha} \nabla \operatorname{TV}(p, q_\phi).$$

It has the same optimum as TV, but the factor $\frac{1}{\alpha}$ amplifies the gradient exactly when acceptance is low. Both losses come from the same two logit tensors as KL:

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

Across drafter architectures and target sizes, both losses improved the acceptance length over KL, most of all where acceptance is lowest: at temperature 1 and at late draft positions. DSpark's and H-Spec's fixed blend of 0.1 cross-entropy and 0.9 TV is a hybrid with a constant weight; the adaptive weight lets the drafter lean on KL early in training and on acceptance later.

#### Kimi-K3

Kimi-K3's [technical report](https://github.com/MoonshotAI/Kimi-K3/blob/main/k3_tech_report.pdf) fine-tunes its pre-trained MTP layer into an EAGLE-3-style drafter, unrolled for seven steps following the training-time test, with $\mathbf{W}\_{\text{fuse}}$ initialized as $[\mathbf{0}\ \mathbf{0}\ \mathbf{I}]$ so that training starts from the final hidden state the MTP layer was pre-trained on. Since minimizing KL doesn't guarantee maximizing the acceptance rate of a capacity-limited drafter, they directly optimize $\mathcal{L}\_{\text{LK}}^{\alpha} = -\log \alpha$, at temperature 1 and with no auxiliary cross-entropy term.

### Beyond per-token objectives

KL, TV and LK losses are all per-token, but the acceptance length is a sum of products:

$$\tau \approx 1 + \sum_{k=1}^{\gamma} \prod_{i=1}^{k} \alpha_i, \qquad \frac{\partial \tau}{\partial \alpha_k} = \frac{1}{\alpha_k} \sum_{j=k}^{\gamma} \prod_{i=1}^{j} \alpha_i.$$

Position $k$ only pays off if every position before it was accepted, so early positions are worth more; for a constant $\alpha$ the weight of position $k$ decays roughly as $\alpha^{k-1}$. Parallel drafters already use such weights: DFlash multiplies the cross-entropy at position $k$ by $\exp\big(-\frac{k-1}{\kappa}\big)$, with $\kappa = 7$ for blocks of 16, and Domino uses the same exponential decay. That's the decay $\tau$ prescribes at $\alpha \approx e^{-1/7} \approx 0.87$, the per-position acceptance behind an acceptance length of 6–7 out of 16, about what DFlash reaches:

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

A fixed schedule is only right for one acceptance rate. D-PACE ([Wu et al., 2026](https://arxiv.org/pdf/2605.18810)) computes adaptive weights from a differentiable surrogate of the acceptance length built from the drafter's own confidences. The E2E-TV loss ([Li et al., 2026](https://arxiv.org/pdf/2606.12370)) drops the per-token form and maximizes the expected number of accepted draft tokens directly, with gradients flowing through the products:

$$\mathcal{L}_{\text{E2E}} = -\sum_{k=1}^{\gamma} \prod_{i=1}^{k} \big(1 - \operatorname{TV}(p_i, q_i)\big).$$

This product is the survival probability of token-wise verification. Under block verification a surplus at one token can pay for a deficit at the next, so drafters with the same TV at every position can accept different numbers of tokens. **BV loss** ([Kim et al., 2026](https://arxiv.org/pdf/2609.34832)) derives the objective from block verification itself: for a fixed drafted prefix, averaged over the rest of the draft and the verifier's coins, it accepts at least the first $k$ tokens with probability exactly $w_k$, the running weight from the verifier's side, so the expected number of accepted draft tokens is $\sum_k \mathbb{E}\_{x \sim q}[w_k]$. Training text comes from the target, and importance weighting turns the clipped running weight into a running minimum of likelihood ratios:

$$\mathbb{E}_{x \sim q}\big[w_k\big] = \mathbb{E}_{x \sim p}\big[\min(1, r_1, \dots, r_k)\big], \qquad r_k = \prod_{i=1}^{k} \frac{q_i(x_{t+i})}{p_i(x_{t+i})}.$$

As in our likelihood loss, a logarithm keeps the gradients alive when acceptance is low:

$$\mathcal{L}_{\text{BV}} = -\log \sum_{k=1}^{\gamma} \min\big(1, r_1, \dots, r_k\big).$$

In practice the last token of each prefix is integrated over the vocabulary, which brings in the target's full distribution and makes the single-token case exactly $\mathcal{L}^{\alpha}\_{\text{LK}}$. DFlash and DSpark drafters trained from scratch with BV loss for Qwen3-4B and Qwen3-8B accept 13–21% more tokens per round under block verification than with cross-entropy and 4–16% more than with LK losses, and the gains carry over to token-wise verification and greedy decoding. Drafters trained on TV alone reached only 1.0–2.2 tokens per round, the vanishing gradient from above. And the cross-entropy drafter ended up matching the target's tokens more often than the BV drafter, 54% against 47%, yet the BV drafter kept 21% more draft tokens per round: like KL, token accuracy is only a proxy.

A second gap is the context. Losses are computed on target-generated text, while at inference the drafter continues its own guesses, from anchors that depend on earlier acceptances. EAGLE-3's training-time test closes the first part by unrolling the drafter. Draft-OPD ([Lei et al., 2026](https://arxiv.org/pdf/2605.29343)) goes further with on-policy distillation, replaying draft proposals from the anchors where rounds actually start at inference and weighting the distillation by acceptance. DLoop's loop-aware training and Carryover's draft–verify–draft training apply the same idea to states that exist only across rounds: the drafter's own hidden states for unverified tokens and the target's states for rejected ones.

The third gap is between rounds. For a parallel drafter the distribution at position $t$ depends on where its round started, which depends on how many tokens the drafter got accepted before, so the states it is evaluated on depend on the drafter itself. **EDR** ([Zhao and Cai, 2026](https://arxiv.org/pdf/2610.10411)) captures this with a Markov reward process conditioned on a target rollout of length $L$. A state $(t_0, t)$ means that the current round started at position $t_0$ and has reached $t$. With probability $A^{q}\_{t_0,t}$ the token is accepted and the process moves to $(t_0, t+1)$; otherwise a new round starts at $(t, t+1)$, and every rejection costs one round. The expected number of rounds is then exactly

$$\mathbb{E}[\#\text{rounds}] = 1 + \sum_{t=1}^{L} \sum_{t_0=0}^{t-1} \omega^{q}_{t_0,t}\, \epsilon^{q}_{t_0,t},$$

where $\omega^{q}\_{t_0,t}$ is the probability that decoding visits state $(t_0, t)$ (its occupancy) and $\epsilon^{q}\_{t_0,t}$ is the expected local rejection cost there, a TV distance between the drafter and the target.

For an autoregressive drafter, whose prediction at position $t$ doesn't depend on where the round started, the occupancies sum to one at every position and the formula collapses to $1 + \mathbb{E}\big[\sum_t \epsilon_t\big]$, the total TV distance along target rollouts, so a per-token acceptance objective is exactly right. For parallel drafters it isn't, and even E2E's product of $1 - \operatorname{TV}$ terms only approximates the survival probability inside a round, the same $\approx$ as in our formula for $\tau$; Qiang et al. found that for DFlash and DSpark in serving acceptances along a block are positively correlated, so the true survival is higher than the product. EDR weights the local costs by occupancies that depend on $q$, derives an exact temporal-difference gradient that can be estimated without bias from target rollouts, and has no hyperparameters. The same machinery gives an exact offline evaluator of the number of rounds, so two drafters can be compared on shared target rollouts without running speculative decoding. Fine-tuning DSpark and DFly[^DFLY] with EDR gives small but consistent gains over both the released checkpoints and E2E-TV across nine math, code and chat benchmarks.

Per-token, per-block and per-sequence objectives form a ladder: LK losses optimize the acceptance of each token, E2E and BV losses the acceptance length of each round, under token-wise and block verification respectively, and EDR the number of rounds for the whole sequence.

## Conclusion

Decoding is memory-bound: one forward pass reads all the weights, so verifying a handful of tokens costs about as much as generating one. Speculative sampling makes guessing lossless, shared Gumbel noise makes it reproducible, block verification and token trees squeeze more out of every draft, and as long as verification stays memory-bound the speedup reduces to a single formula, $\tau / (n_{\text{draft}} \cdot c + 1)$. Every drafter is a different answer to two questions: what should the guesser see, and how should it be trained?

Architecture decides what $q$ can express. We went from a separate small LLM to autoregressive drafting over target hidden states, to parallel drafters with target states injected into every layer, to parallel drafters that restore dependency with a light causal head or a second pass over their own block, and finally to drafters that carry rejected work into the next round and read the target's own cache instead of keeping one. Information floors show that a head needs to see only a token or two back to undo most of parallel blindness, so what remains is the drafter's own model gap, and since a sliding window already bounds the drafter's memory, reading the target's cache pays off as information more than as saved memory. The objective decides which $q$ we actually get, and since drafters are small, the KL optimum is not the acceptance optimum.

Plenty of questions remain open: drafters for long-context and agentic workloads, where prefill and memory dominate and tree drafting gains shrink; acceptance-aware training for parallel drafters, which have the most to gain from it; and drafters for diffusion language models, which already generate whole blocks. Whatever comes next, it will still be the art of guessing well.

---
[^JAC]: Jacobi decoding ([Santilli et al., 2023](https://arxiv.org/pdf/2305.10427)) and Lookahead decoding ([Fu et al., 2024](https://arxiv.org/pdf/2402.02057)) let the target refine a whole window of guesses in parallel, and CLLMs ([Kou et al., 2024](https://arxiv.org/pdf/2403.00835)) fine-tune the model to converge in fewer iterations. Drafting without any model, by copying n-grams from the prompt (prompt lookup) or from earlier outputs (suffix decoding), is surprisingly strong for code editing and agent loops, where outputs repeat their inputs.

[^HYD]: Blockwise parallel decoding ([Stern et al., 2018](https://arxiv.org/pdf/1811.03115)) did this first. Hydra ([Ankner et al., 2024](https://arxiv.org/pdf/2402.05109)) made Medusa heads sequentially dependent, and Medusa-2's typical acceptance trades exactness for speed by accepting any plausible token instead of using speculative sampling.

[^SV2]: Serving engines matter as much as drafters. SGLang's Spec V2 overlaps host-side scheduling with GPU work and raises throughput by over 33%, from about 11.4k to 15.3k tokens/s, for Qwen3-8B on a single B200 at concurrency 32 ([LMSYS](https://www.lmsys.org/blog/2026-06-15-next-generation-speculative-decoding-dflash-v2/)).

[^DFLY]: DFly ([Liu et al., 2026](https://arxiv.org/pdf/2607.25852)) comes from the AngelSpec framework. It's another block-diffusion drafter with a predecessor-conditioned autoregressive head, i.e. the same "parallel + causal head" family as DSpark and Domino.
