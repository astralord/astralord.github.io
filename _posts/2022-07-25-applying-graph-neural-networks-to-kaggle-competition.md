---
layout: post
title: 'Applying Graph Neural Networks to Kaggle Competition'
date: 2022-07-25 00:00 +0800
categories: [Graph Neural Networks]
tags: [graph-neural-networks, kaggle, reinforcement-learning]
math: true
enable_d3: true
---

Few months ago, Kaggle launched featured simulation competition [Kore-2022](https://www.kaggle.com/competitions/kore-2022). In this kind of competitions participants bots are competing against each other in an game environment, supported by Kaggle. Often there are 2 or 4 players in a game, at the end each winner/loser moves up/down according to skill rating system. Team reaching top-rating wins.

### Kore rules

That's how entry to Kore competition looks like:

> In this turn-based simulation game you control a small armada of spaceships. As you mine the rare mineral “kore” from the depths of space, you teleport it back to your homeworld. But it turns out you aren’t the only civilization with this goal. In each game two players will compete to collect the most kore from the board. Whoever has the largest kore cache by the end of 400 turns—or eliminates all of their opponents from the board before that—will be the winner!


![Kore Gameplay]({{'/assets/img/kore-gameplay.gif'|relative_url}})

*Fig. 1. Gameplay visualization by [Tong Hui Kang](https://www.kaggle.com/competitions/kore-2022/discussion/320987)*

Game setup looks extremely complex at first glance (at least compared to the other Kaggle simulation competitions like [Hungry Geese](https://www.kaggle.com/competitions/hungry-geese), which was basically a variation of Tetris Snake game). 

Here I'll try to list the main rules:

- The game board is 21x21 tiles large and wraps around on both the north/south and east/west borders
- You start the game with 1 shipyard and 500 kore.
- Each turn you can either spawn new ships (each costs 10 kore), or launch fleet with a flight plan, or do nothing.
- Flight plan is a sequence of letters and integers:
	- 'N', 'E', 'S' or 'W' defines the direction of the fleet.
	- The integer following the letter determinetiles how many steps (+1) fleet will take in that direction.
	- For example, 'NE2SW' forms a circle trajectory, where fleet makes one step up, three steps right, one step down and then it moves left until it reaches starting point.
	- The maximum length of the flight plan depends on fleet size by formula:
	  $$\max \text{length} = \lfloor 2 \log(\text{fleet size}) \rfloor + 1$$ 
	  E.g. the minimum number of ships for a fleet that can complete a circle like 'NE2SW' is 8.
	- If plan ends with 'C', at the end tile of the flight fleet builds a shipyard, consuming 50 of its ships.
- Each fleet does damage to all enemy fleets orthogonally adjacent to it equal to its current ship count. Destroyed fleets drop 50% of their kore onto their currently occupied position with 50% being awarded to their attackers. If the attacking fleet does not survive, the kore is dropped back onto the tile instead.
- Fleets that move onto the same square collide. Allied fleets are absorbed into the largest fleet. The largest fleet survives and all other fleets will be destroyed. The remaining fleet loses ships equal to the ship count of the second largest fleet. The largest fleet steals all of the kore from other fleets it collides with.

In a nutshell: you collect kore, spawn ships and try to destroy all of enemy objects before he destroys yours. The main difficulty for player is to construct an optimal path planning solution for fleet flights. Players must calculate many steps ahead as the effects of their actions in most cases do not appear immediately.
  
### Graph construction

Now how one can approach this task? Prior to this competition, Kaggle launched its trial version, Kore Beta, with the only difference being that it was 4-players game. First places were taken by rule-based agents, based mostly on simple heuristics. 

Another way is to think of this task as text generation. One can train a network which takes board image as input and returns flight plan as text output for each shipyard. [A solution with this kind of approach](https://www.kaggle.com/competitions/kore-2022/discussion/337476) took one of the top places eventually.

One can also represent board as graph, where each tile is a node and edges connect adjacent tiles. Then each turn, when a fleet is sent from shipyard, searching for optimal flight plan is equivalent to searching for optimal path on a graph.

![Torus Graph]({{'/assets/img/torus.png'|relative_url}})

*Fig. 2. This is how periodic 21x21 grid looks like: a doughnut.*

One of the main advantages of an algorithm on a graph is rotation equivariance. Board is symmetric, therefore algorithm behaviour must not change if we swap players initial positions. With standard convolutional networks we can try to overcome this issue by using augmentations, but there is no need for them if we use graph neural networks.

We also don't want to miss the information which is not included in this kind of board representation: future positions of fleets. We know all of them by fact, because players plans are not hidden from each other. Therefore, it seems useful to add board representations at the next step, at the step after next and so on until we reach desirable amount of steps. Also, after we make our moves, these future graphs may change, so it doesn't always seem reasonable to look far away from current step. At my implementation I was looking 12 steps ahead.

<script src="https://d3js.org/d3.v4.min.js"></script>
<link href="https://fonts.googleapis.com/css?family=Arvo" rel="stylesheet">

<div id="grphzmd" class="svg-container" align="center"></div> 

<script>
function graph_zoomed() {

var svg = d3.select("#grphzmd")
			  .append("svg")
			  .attr("width", 600)
			  .attr("height", 300);

function draw_edge(x, y, type, opacity=0.95) {
   var x1 = 0, x2 = 0, y1 = 0, y2 = 0;
   var dash = 0;
   if (type == 'hrz') {
       x1 = x - 40;
       y1 = y;
       x2 = x + 40;
       y2 = y;
   }
   else if (type == 'vrt') {
       x1 = x;
       y1 = y - 40;
       x2 = x;
       y2 = y + 40;
   }
   else if (type == 'tmp') {
       x1 = x;
       y1 = y;
       x2 = x + 35;
       y2 = y - 50;
       dash = 1;
   }

	svg.append('line')
	  .attr('x1', x1)
	  .attr('y1', y1)
	  .attr('x2', x2)
	  .attr('y2', y2)
	  .style("stroke-width", 4)
	  .style("stroke-dasharray", ("10, " + dash))
	  .attr("opacity", opacity)
	  .attr('stroke', '#E86456');

}

function draw_node(x, y, opacity=0.95) {
	svg.append('circle')
	  .attr('cx', x - 50)
	  .attr('cy', y)
	  .attr('r', 10)
	  .attr('stroke', 'black')
	  .attr("opacity", opacity)
	  .attr('fill', '#348ABD');
}

function draw_cross(x, y, opacity) {
	draw_edge(x - 50, y, 'hrz', opacity);
	draw_edge(x + 50, y, 'hrz', opacity);
	draw_edge(x, y - 50, 'vrt', opacity);
	draw_edge(x, y + 50, 'vrt', opacity);
	draw_edge(x + 10, y - 15, 'tmp', opacity);
	
	draw_node(x + 50, y, opacity);
	draw_node(x - 50, y, opacity);
	draw_node(x + 50, y + 100, opacity);
	draw_node(x + 50, y - 100, opacity);
	draw_node(x + 150, y, opacity);
	draw_node(x + 98, y - 69, opacity);

	var triangleSize = 70;
	var triangle = d3.symbol()
	            .type(d3.symbolTriangle)
	            .size(triangleSize);
		
	svg.append("path")
	   .attr("d", triangle)
	   .attr("stroke", "white")
	   .attr("fill", "#E86456")
	   .attr("opacity", opacity)
	   .attr("transform",
	   		function(d) { return "translate(" + (x + 10) + "," + (y - 15) + ") rotate(" + -25  + ")"; });
}

draw_cross(300, 150, 0.95);

var opacity = 0.04;

draw_cross(300, 50, opacity);
draw_cross(200, 150, opacity);
draw_cross(400, 150, opacity);
draw_cross(300, 250, opacity);
draw_cross(348, 81, opacity);

draw_cross(400, 250, opacity);
draw_cross(200, 250, opacity);
draw_cross(400, 50, opacity);
draw_cross(200, 50, opacity);
draw_cross(348, 181, opacity);
draw_cross(448, 81, opacity);
draw_cross(248, 81, opacity);

svg.append('text')
  .attr('x', 50)
  .attr('y', 130)
  .text("Node features")
  .style("font-size", "14px")
  .attr("font-family", "Arvo");
  
svg.append('line')
  .attr('x1', 110)
  .attr('y1', 105)
  .attr('x2', 287)
  .attr('y2', 53)
  .style("stroke-width", 1)
  .attr("opacity", 0.95)
  .attr('stroke', 'black');
  
for (var i = 0; i < 6; i += 1) {
	svg.append('rect')
	  .attr('x', 100 - i * 6)
	  .attr('y', 85 + i * 3)
	  .attr('width', 12)
	  .attr('height', 12)
	  .attr('stroke', 'black')
	  .attr("rx", 3)
	  .attr("opacity", 0.9)
	  .attr('fill', '#348ABD');
}
            
svg.append('text')
  .attr('x', 50)
  .attr('y', 260)
  .text("Edge features")
  .style("font-size", "14px")
  .attr("font-family", "Arvo");
  
svg.append('text')
  .attr('x', 295)
  .attr('y', 155)
  .text("C")
  .style("font-size", "14px")
  .attr("font-family", "Arvo");
  
svg.append('text')
  .attr('x', 191)
  .attr('y', 155)
  .text("W")
  .style("font-size", "14px")
  .attr("font-family", "Arvo");
  
svg.append('text')
  .attr('x', 395)
  .attr('y', 155)
  .text("E")
  .style("font-size", "14px")
  .attr("font-family", "Arvo");
  
svg.append('text')
  .attr('x', 294)
  .attr('y', 55)
  .text("N")
  .style("font-size", "14px")
  .attr("font-family", "Arvo");
  
svg.append('text')
  .attr('x', 296)
  .attr('y', 255)
  .text("S")
  .style("font-size", "14px")
  .attr("font-family", "Arvo");
  
svg.append('text')
  .attr('x', 343)
  .attr('y', 86)
  .text("T")
  .style("font-size", "14px")
  .attr("font-family", "Arvo");
  
svg.append('line')
  .attr('x1', 120)
  .attr('y1', 230)
  .attr('x2', 255)
  .attr('y2', 155)
  .style("stroke-width", 1)
  .attr("opacity", 0.95)
  .attr('stroke', 'black');
  
for (var i = 0; i < 3; i += 1) {
	svg.append('rect')
	  .attr('x', 100 - i * 6)
	  .attr('y', 225 + i * 3)
	  .attr('width', 12)
	  .attr('height', 12)
	  .attr('stroke', 'black')
	  .attr("rx", 3)
	  .attr("opacity", (i == 2) ? 1.0 : 0.6)
	  .attr('fill', '#E86456');
}
	   		

}

graph_zoomed();
  
</script>

![](.)
*Fig. 3. Graph zoomed in. Each node corresponds to a tile on the board, each edge represents that either nodes are spatial neighbors, or they are the same tiles but at adjacent timestamps. Graph is oriented, nodes from future graphs are pointing at nodes at previous steps. This way all the information flows to current board representation which later is going to be used for actions generation.*

Node input features $\mathbf{v}_i$ contain kore amount on the tile, timestamp, total amount of kore collected by player and its opponent. Also, if any fleet is located on the tile - its cargo, collection rate and number of ships. If any shipyard is on the tile - its number of ships, maximum plan length for this number and maximum number of ships, which can be spawned.

The edge input features $\mathbf{e}_{ji}$ are $(1, 0, 0)^T$, $(0, 1, 0)^T$ and $(0, 0, 1)^T$ for temporal, lateral and longitudinal edges respectively. Basically, they are one-hot representations of these edge classes. Why treat spatial edges differently? When you create a flight plan, its length increases when fleet makes a turn: while for both "N1S" and "NEWS" fleet moves 2 tiles from shipyard and returns back, the first one is smaller and requires fewer ships.

### Graph Encoder Architecture

If we had to work with standard representations of board, we would most likely use convolutional layers. There exists an analogue of convolutions in a graph world, called, big surprise, graph convolutions. Similarly to ResNet architecture we can build a ResGCN encoder:

<div id="archtctr" class="svg-container" align="center"></div> 

<script>
			  
function draw_triangle(svg, x, y, rotate=0) {
	var triangleSize = 25;
	var triangle = d3.symbol()
	            .type(d3.symbolTriangle)
	            .size(triangleSize);
	
	svg.append("path")
	   .attr("d", triangle)
	   .attr("stroke", "black")
	   .attr("fill", "gray")
	   .attr("transform",
	   		function(d) { return "translate(" + x + "," + y + ") rotate(" + rotate  + ")"; });
}

function architecture() {

var svg = d3.select("#archtctr")
			  .append("svg")
			  .attr("width", 600)
			  .attr("height", 95);

function draw_graph(x, y, node_color, edge_color) {

	svg.append('circle')
	  .attr('cx', x)
	  .attr('cy', y)
	  .attr('r', 5)
	  .attr('stroke', 'black')
	  .attr('stroke-width', 1.25)
	  .attr("opacity", 0.95)
	  .attr('fill', node_color);
	  
	svg.append('circle')
	  .attr('cx', x + 20)
	  .attr('cy', y - 20)
	  .attr('r', 5)
	  .attr('stroke', 'black')
	  .attr('stroke-width', 1.25)
	  .attr("opacity", 0.95)
	  .attr('fill', node_color);
	  
	svg.append('circle')
	  .attr('cx', x + 40)
	  .attr('cy', y)
	  .attr('r', 5)
	  .attr('stroke', 'black')
	  .attr('stroke-width', 1.25)
	  .attr("opacity", 0.95)
	  .attr('fill', node_color);
	  
	svg.append('circle')
	  .attr('cx', x + 20)
	  .attr('cy', y + 20)
	  .attr('r', 5)
	  .attr('stroke', 'black')
	  .attr('stroke-width', 1.25)
	  .attr("opacity", 0.95)
	  .attr('fill', node_color);

	svg.append('line')
	  .attr('x1', x + 4)
	  .attr('y1', y - 4)
	  .attr('x2', x + 16)
	  .attr('y2', y - 16)
	  .style("stroke-width", 4)
	  .attr("opacity", 0.95)
	  .attr('stroke', edge_color);

	svg.append('line')
	  .attr('x1', x + 24)
	  .attr('y1', y - 16)
	  .attr('x2', x + 36)
	  .attr('y2', y - 4)
	  .style("stroke-width", 4)
	  .attr("opacity", 0.95)
	  .attr('stroke', edge_color);

	svg.append('line')
	  .attr('x1', x + 36)
	  .attr('y1', y + 4)
	  .attr('x2', x + 24)
	  .attr('y2', y + 16)
	  .style("stroke-width", 4)
	  .attr("opacity", 0.95)
	  .attr('stroke', edge_color);
	  
	svg.append('line')
	  .attr('x1', x + 20)
	  .attr('y1', y - 15)
	  .attr('x2', x + 20)
	  .attr('y2', y + 15)
	  .style("stroke-width", 4)
	  .attr("opacity", 0.95)
	  .attr('stroke', edge_color);
}

draw_graph(100, 50, '#348ABD', '#E86456');
draw_graph(450, 50, '#A4D8D8', '#B19CD9');

svg.append('rect')
  .attr('x', 220)
  .attr('y', 30)
  .attr('width', 120)
  .attr('height', 35)
  .attr('stroke', 'black')
  .attr("rx", 3)
  .attr("opacity", 0.6)
  .attr('fill', '#65AD69');
  
svg.append('text')
  .attr('x', 252)
  .attr('y', 52)
  .text("ResGCN")
  .style("font-size", "14px")
  .attr("font-family", "Arvo");
  
svg.append('line')
  .attr('x1', 150)
  .attr('y1', 50)
  .attr('x2', 220)
  .attr('y2', 50)
  .style("stroke-width", 1)
  .attr("opacity", 0.95)
  .attr('stroke', 'black');
  
svg.append('line')
  .attr('x1', 375)
  .attr('y1', 50)
  .attr('x2', 440)
  .attr('y2', 50)
  .style("stroke-width", 1)
  .attr("opacity", 0.95)
  .attr('stroke', 'black');
  
svg.append('line')
  .attr('x1', 340)
  .attr('y1', 50)
  .attr('x2', 355)
  .attr('y2', 50)
  .style("stroke-width", 1)
  .attr("opacity", 0.95)
  .attr('stroke', 'black');
  
svg.append('line')
  .attr('x1', 195)
  .attr('y1', 50)
  .attr('x2', 195)
  .attr('y2', 10)
  .style("stroke-width", 1)
  .attr("opacity", 0.95)
  .attr('stroke', 'black');
  
svg.append('line')
  .attr('x1', 195)
  .attr('y1', 10)
  .attr('x2', 365)
  .attr('y2', 10)
  .style("stroke-width", 1)
  .attr("opacity", 0.95)
  .attr('stroke', 'black');
  
svg.append('line')
  .attr('x1', 365)
  .attr('y1', 10)
  .attr('x2', 365)
  .attr('y2', 40)
  .style("stroke-width", 1)
  .attr("opacity", 0.95)
  .attr('stroke', 'black');
  
draw_triangle(svg, 215, 50, 90);
draw_triangle(svg, 435, 50, 90);

svg.append('rect')
  .attr('x', 175)
  .attr('y', 1)
  .attr('width', 215)
  .attr('height', 75)
  .attr('stroke', 'black')
  .attr("rx", 3)
  .style('stroke-dasharray', ('2,3'))
  .attr('fill', 'none');
  
svg.append('text')
  .attr('x', 275)
  .attr('y', 92)
  .text("x4")
  .style("font-size", "14px")
  .attr("font-family", "Arvo");
  
svg.append("path")
    .attr("d", d3.symbol().type(d3.symbolCross).size(75))
    .attr("transform", "translate (365, 50)")
    .style("fill", "gray")
    .style('stroke', 'black')
    .style('stroke-width', '0.7')
    .style('opacity', 1.0);
    
svg.append('circle')
  .attr('cx', 365)
  .attr('cy', 50)
  .attr('r', 10)
  .attr('stroke', 'black')
  .attr('stroke-width', 1.25)
  .attr("opacity", 0.95)
  .attr('fill', "none");
  
}

architecture();

</script>

![](.)
*Fig. 4. Bird's eye view of graph encoder architecture. The summation of ResGCN input and output graphs is performed both over node and over edge features.*

It is also extremely important to make our graph neural network **anisotropic**, which means that each neighbor should have a different effect on the node depending on the weight of the edge between them. The idea is that the neural network transforms the nodes in such a way that the encoded features of the nodes lying on the agent's path are more similar to each other than to nodes that are not.

<div id="resgcn" class="svg-container" align="center"></div> 

<script>

function resgcn() {

var svg = d3.select("#resgcn")
			  .append("svg")
			  .attr("width", 700)
			  .attr("height", 200);
			  
for (var i = 0; i < 6; i += 1) {
	svg.append('rect')
	  .attr('x', 60 - i * 6)
	  .attr('y', 15 + i * 3)
	  .attr('width', 12)
	  .attr('height', 12)
	  .attr('stroke', 'black')
	  .attr("rx", 3)
	  .attr("opacity", 0.9)
	  .attr('fill', '#348ABD');
}

for (var i = 0; i < 3; i += 1) {
	svg.append('rect')
	  .attr('x', 50 - i * 6)
	  .attr('y', 150 + i * 3)
	  .attr('width', 12)
	  .attr('height', 12)
	  .attr('stroke', 'black')
	  .attr("rx", 3)
	  .attr("opacity", (i == 2) ? 1.0 : 0.6)
	  .attr('fill', '#E86456');
}

svg.append('line')
  .attr('x1', 50)
  .attr('y1', 40)
  .attr('x2', 50)
  .attr('y2', 145)
  .style("stroke-width", 1)
  .attr("opacity", 0.95)
  .attr('stroke', 'black');
 
svg.append("path")
   .attr("stroke", "black")
   .datum([{x: 50, y: 110}, {x: 100, y: 110}, {x: 100, y: 25}, {x: 150, y: 25}])
   .attr("fill", "none")
   .attr("d",  d3.line()
       .curve(d3.curveBasis)
       .x(function(d) { return d.x; })
       .y(function(d) { return d.y; }));
 
svg.append("path")
   .attr("stroke", "black")
   .datum([{x: 50, y: 110}, {x: 100, y: 110}, {x: 100, y: 60}, {x: 150, y: 60}])
   .attr("fill", "none")
   .attr("d",  d3.line()
       .curve(d3.curveBasis)
       .x(function(d) { return d.x; })
       .y(function(d) { return d.y; }));
 
svg.append("path")
   .attr("stroke", "black")
   .datum([{x: 50, y: 110}, {x: 100, y: 110}, {x: 100, y: 175}, {x: 150, y: 175}])
   .attr("fill", "none")
   .attr("d",  d3.line()
       .curve(d3.curveBasis)
       .x(function(d) { return d.x; })
       .y(function(d) { return d.y; }));
	       
svg.append('rect')
  .attr('x', 150)
  .attr('y', 10)
  .attr('width', 120)
  .attr('height', 30)
  .attr('stroke', 'black')
  .attr("rx", 3)
  .attr("opacity", 0.6)
  .attr('fill', '#65AD69');
  
svg.append('text')
  .attr('x', 165)
  .attr('y', 30)
  .text("ResGCN block")
  .style("font-size", "14px")
  .attr("font-family", "Arvo");
	       
svg.append('rect')
  .attr('x', 150)
  .attr('y', 45)
  .attr('width', 120)
  .attr('height', 30)
  .attr('stroke', 'black')
  .attr("rx", 3)
  .attr("opacity", 0.6)
  .attr('fill', '#65AD69');
  
svg.append('text')
  .attr('x', 165)
  .attr('y', 65)
  .text("ResGCN block")
  .style("font-size", "14px")
  .attr("font-family", "Arvo");
	       
svg.append('rect')
  .attr('x', 150)
  .attr('y', 160)
  .attr('width', 120)
  .attr('height', 30)
  .attr('stroke', 'black')
  .attr("rx", 3)
  .attr("opacity", 0.6)
  .attr('fill', '#65AD69');
  
svg.append('text')
  .attr('x', 165)
  .attr('y', 180)
  .text("ResGCN block")
  .style("font-size", "14px")
  .attr("font-family", "Arvo");
  
svg.append('line')
  .attr('x1', 210)
  .attr('y1', 85)
  .attr('x2', 210)
  .attr('y2', 155)
  .style("stroke-width", 2)
  .style('stroke-dasharray', ('2,10'))
  .attr("opacity", 0.95)
  .attr('stroke', 'black');
  
draw_triangle(svg, 145, 25, 90);
draw_triangle(svg, 145, 60, 90);
draw_triangle(svg, 145, 175, 90);

svg.append("path")
   .attr("stroke", "black")
   .datum([{x: 270, y: 25}, {x: 320, y: 25}, {x: 320, y: 110}, {x: 345, y: 110}])
   .attr("fill", "none")
   .attr("d",  d3.line()
       .curve(d3.curveBasis)
       .x(function(d) { return d.x; })
       .y(function(d) { return d.y; }));
 
svg.append("path")
   .attr("stroke", "black")
   .datum([{x: 270, y: 60}, {x: 320, y: 60}, {x: 320, y: 110}, {x: 345, y: 110}])
   .attr("fill", "none")
   .attr("d",  d3.line()
       .curve(d3.curveBasis)
       .x(function(d) { return d.x; })
       .y(function(d) { return d.y; }));
 
svg.append("path")
   .attr("stroke", "black")
   .datum([{x: 270, y: 175}, {x: 320, y: 175}, {x: 320, y: 110}, {x: 345, y: 110}])
   .attr("fill", "none")
   .attr("d",  d3.line()
       .curve(d3.curveBasis)
       .x(function(d) { return d.x; })
       .y(function(d) { return d.y; }));
       
svg.append('circle')
  .attr('cx', 355)
  .attr('cy', 110)
  .attr('r', 10)
  .attr('stroke', 'black')
  .attr('stroke-width', 1.25)
  .attr("opacity", 0.95)
  .attr('fill', "none");
  
svg.append('line')
  .attr('x1', 365)
  .attr('y1', 110)
  .attr('x2', 380)
  .attr('y2', 110)
  .style("stroke-width", 1)
  .attr("opacity", 0.95)
  .attr('stroke', 'black');
  
svg.append('text')
  .attr('x', 352)
  .attr('y', 115)
  .text("||")
  .style("font-size", "14px")
  .attr("font-family", "Arvo");
			  
for (var i = 0; i < 6; i += 1) {
	svg.append('rect')
	  .attr('x', 610 - i * 6)
	  .attr('y', 15 + i * 3)
	  .attr('width', 12)
	  .attr('height', 12)
	  .attr('stroke', 'black')
	  .attr("rx", 3)
	  .attr("opacity", 0.9)
	  .attr('fill', '#A4D8D8');
}
  
svg.append('line')
  .attr('x1', 400)
  .attr('y1', 125)
  .attr('x2', 420)
  .attr('y2', 125)
  .style("stroke-width", 1)
  .attr("opacity", 0.95)
  .attr('stroke', 'black');
  
svg.append('line')
  .attr('x1', 380)
  .attr('y1', 110)
  .attr('x2', 380)
  .attr('y2', 25)
  .style("stroke-width", 1)
  .attr("opacity", 0.95)
  .attr('stroke', 'black');
  
svg.append('line')
  .attr('x1', 380)
  .attr('y1', 110)
  .attr('x2', 380)
  .attr('y2', 155)
  .style("stroke-width", 1)
  .attr("opacity", 0.95)
  .attr('stroke', 'black');
  
svg.append('line')
  .attr('x1', 380)
  .attr('y1', 25)
  .attr('x2', 570)
  .attr('y2', 25)
  .style("stroke-width", 1)
  .attr("opacity", 0.95)
  .attr('stroke', 'black');
  
svg.append('line')
  .attr('x1', 400)
  .attr('y1', 125)
  .attr('x2', 400)
  .attr('y2', 180)
  .style("stroke-width", 1)
  .attr("opacity", 0.95)
  .attr('stroke', 'black');
  
svg.append('line')
  .attr('x1', 400)
  .attr('y1', 180)
  .attr('x2', 420)
  .attr('y2', 180)
  .style("stroke-width", 1)
  .attr("opacity", 0.95)
  .attr('stroke', 'black');
  
svg.append('line')
  .attr('x1', 380)
  .attr('y1', 155)
  .attr('x2', 400)
  .attr('y2', 155)
  .style("stroke-width", 1)
  .attr("opacity", 0.95)
  .attr('stroke', 'black');
  
draw_triangle(svg, 570, 25, 90);
draw_triangle(svg, 420, 125, 90);
draw_triangle(svg, 420, 180, 90);

svg.append('rect')
  .attr('x', 425)
  .attr('y', 110)
  .attr('width', 80)
  .attr('height', 30)
  .attr('stroke', 'black')
  .attr("rx", 3)
  .attr("opacity", 0.6)
  .attr('fill', '#EDA137');
  
svg.append('text')
  .attr('x', 445)
  .attr('y', 130)
  .text("Query")
  .style("font-size", "14px")
  .attr("font-family", "Arvo");

svg.append('rect')
  .attr('x', 425)
  .attr('y', 165)
  .attr('width', 80)
  .attr('height', 30)
  .attr('stroke', 'black')
  .attr("rx", 3)
  .attr("opacity", 0.6)
  .attr('fill', '#EDA137');
  
svg.append('text')
  .attr('x', 452)
  .attr('y', 185)
  .text("Key")
  .style("font-size", "14px")
  .attr("font-family", "Arvo");
  
svg.append('line')
  .attr('x1', 505)
  .attr('y1', 125)
  .attr('x2', 520)
  .attr('y2', 125)
  .style("stroke-width", 1)
  .attr("opacity", 0.95)
  .attr('stroke', 'black');
  
svg.append('line')
  .attr('x1', 520)
  .attr('y1', 125)
  .attr('x2', 520)
  .attr('y2', 145)
  .style("stroke-width", 1)
  .attr("opacity", 0.95)
  .attr('stroke', 'black');
  
svg.append('line')
  .attr('x1', 520)
  .attr('y1', 180)
  .attr('x2', 520)
  .attr('y2', 165)
  .style("stroke-width", 1)
  .attr("opacity", 0.95)
  .attr('stroke', 'black');
  
svg.append('line')
  .attr('x1', 505)
  .attr('y1', 180)
  .attr('x2', 520)
  .attr('y2', 180)
  .style("stroke-width", 1)
  .attr("opacity", 0.95)
  .attr('stroke', 'black');

svg.append('circle')
  .attr('cx', 520)
  .attr('cy', 155)
  .attr('r', 10)
  .attr('stroke', 'black')
  .attr('stroke-width', 1.25)
  .attr("opacity", 0.95)
  .attr('fill', "none");
  
svg.append('circle')
  .attr('cx', 520)
  .attr('cy', 155)
  .attr('r', 1)
  .attr('stroke', 'black')
  .attr('stroke-width', 1.25)
  .attr("opacity", 0.95)
  .attr('fill', "black");
  
svg.append('line')
  .attr('x1', 530)
  .attr('y1', 155)
  .attr('x2', 570)
  .attr('y2', 155)
  .style("stroke-width", 1)
  .attr("opacity", 0.95)
  .attr('stroke', 'black');
  
svg.append('text')
  .attr('x', 535)
  .attr('y', 150)
  .text("tanh")
  .style("font-size", "12px")
  .attr("font-family", "Arvo");
  
for (var i = 0; i < 3; i += 1) {
	svg.append('rect')
	  .attr('x', 600 - i * 6)
	  .attr('y', 150 + i * 3)
	  .attr('width', 12)
	  .attr('height', 12)
	  .attr('stroke', 'black')
	  .attr("rx", 3)
	  .attr("opacity", (i == 2) ? 1.0 : 0.6)
	  .attr('fill', '#B19CD9');
}

svg.append('line')
  .attr('x1', 600)
  .attr('y1', 40)
  .attr('x2', 600)
  .attr('y2', 145)
  .style("stroke-width", 1)
  .attr("opacity", 0.95)
  .attr('stroke', 'black');

draw_triangle(svg, 570, 155, 90);

}


resgcn();

</script>

![](.)
*Fig. 5. ResGCN architecture. First, node features are encoded through several  independent ResGCN blocks. Each block is a mapping $\mathbb{R}^d \rightarrow \mathbb{R}^{d/n}$, where $n$ is a number of blocks. Outputs are concatenated together, constructing vector of the same size as the input. Then to get new edge features we pass outputs through feed-forward layers: $\operatorname{Query}, \operatorname{Key} \in \mathbb{R}^{d \times 3}$, followed by element-wise multiplication and $\operatorname{tanh}$ activation.*

ResGCN block consists of sequential graph convolutional layers:

$$\operatorname{GCN}(v_i, e) = \Theta^T \sum_{j\in\mathcal{N}(i) \cup \lbrace i \rbrace} \frac{e_{ji}}{\sqrt{\hat{d}_i \hat{d}_j}}  v_j$$

with 
$$\hat{d}_i = 1 + \sum_{j \in \mathcal{N}(i)}e_{ji}$$ and $\mathcal{N}(i)$ - set of all neighbors for node $v_i$.

<div id="resgcn_head" class="svg-container" align="center"></div> 

<script>

function resgcn_head() {

var svg = d3.select("#resgcn_head")
			  .append("svg")
			  .attr("width", 500)
			  .attr("height", 342);

for (var i = 0; i < 6; i += 1) {
	svg.append('rect')
	  .attr('x', 200 - i * 6)
	  .attr('y', 5 + i * 3)
	  .attr('width', 12)
	  .attr('height', 12)
	  .attr('stroke', 'black')
	  .attr("rx", 3)
	  .attr("opacity", 0.9)
	  .attr('fill', '#348ABD');
}

svg.append('line')
  .attr('x1', 195)
  .attr('y1', 30)
  .attr('x2', 195)
  .attr('y2', 80)
  .style("stroke-width", 1)
  .attr("opacity", 0.95)
  .attr('stroke', 'black');
  
svg.append('rect')
  .attr('x', 135)
  .attr('y', 80)
  .attr('width', 120)
  .attr('height', 30)
  .attr('stroke', 'black')
  .attr("rx", 3)
  .attr("opacity", 0.6)
  .attr('fill', '#65AD69');
  
svg.append('text')
  .attr('x', 160)
  .attr('y', 100)
  .text("GCN (48, 8)")
  .style("font-size", "14px")
  .attr("font-family", "Arvo");
  
svg.append('text')
  .attr('x', 120)
  .attr('y', 137)
  .text("GraphNorm")
  .style("font-size", "12px")
  .attr("font-family", "Arvo");

svg.append('line')
  .attr('x1', 195)
  .attr('y1', 110)
  .attr('x2', 195)
  .attr('y2', 160)
  .style("stroke-width", 1)
  .attr("opacity", 0.95)
  .attr('stroke', 'black');
  
svg.append('rect')
  .attr('x', 135)
  .attr('y', 160)
  .attr('width', 120)
  .attr('height', 30)
  .attr('stroke', 'black')
  .attr("rx", 3)
  .attr("opacity", 0.6)
  .attr('fill', '#65AD69');
  
svg.append('text')
  .attr('x', 160)
  .attr('y', 180)
  .text("GCN (8, 32)")
  .style("font-size", "14px")
  .attr("font-family", "Arvo");
  
svg.append('text')
  .attr('x', 156)
  .attr('y', 220)
  .text("GELU")
  .style("font-size", "12px")
  .attr("font-family", "Arvo");

svg.append('line')
  .attr('x1', 195)
  .attr('y1', 190)
  .attr('x2', 195)
  .attr('y2', 240)
  .style("stroke-width", 1)
  .attr("opacity", 0.95)
  .attr('stroke', 'black');
  
svg.append('rect')
  .attr('x', 135)
  .attr('y', 240)
  .attr('width', 120)
  .attr('height', 30)
  .attr('stroke', 'black')
  .attr("rx", 3)
  .attr("opacity", 0.6)
  .attr('fill', '#65AD69');
  
svg.append('text')
  .attr('x', 160)
  .attr('y', 260)
  .text("GCN (32, 8)")
  .style("font-size", "14px")
  .attr("font-family", "Arvo");

svg.append('line')
  .attr('x1', 195)
  .attr('y1', 270)
  .attr('x2', 195)
  .attr('y2', 320)
  .style("stroke-width", 1)
  .attr("opacity", 0.95)
  .attr('stroke', 'black');
  

for (var i = 0; i < 3; i += 1) {
	svg.append('rect')
	  .attr('x', 400 - i * 6)
	  .attr('y', 5 + i * 3)
	  .attr('width', 12)
	  .attr('height', 12)
	  .attr('stroke', 'black')
	  .attr("rx", 3)
	  .attr("opacity", (i == 2) ? 1.0 : 0.6)
	  .attr('fill', '#E86456');
}

svg.append('line')
  .attr('x1', 400)
  .attr('y1', 30)
  .attr('x2', 400)
  .attr('y2', 45)
  .style("stroke-width", 1)
  .attr("opacity", 0.95)
  .attr('stroke', 'black');
  
svg.append('rect')
  .attr('x', 340)
  .attr('y', 45)
  .attr('width', 120)
  .attr('height', 30)
  .attr('stroke', 'black')
  .attr("rx", 3)
  .attr("opacity", 0.6)
  .attr('fill', '#EDA137');
  
svg.append('text')
  .attr('x', 363)
  .attr('y', 65)
  .text("Linear (3, 1)")
  .style("font-size", "14px")
  .attr("font-family", "Arvo");
  
svg.append('text')
  .attr('x', 387)
  .attr('y', 88)
  .text("σ")
  .style("font-size", "12px")
  .attr("font-family", "Arvo");

svg.append('line')
  .attr('x1', 400)
  .attr('y1', 75)
  .attr('x2', 400)
  .attr('y2', 255)
  .style("stroke-width", 1)
  .attr("opacity", 0.95)
  .attr('stroke', 'black');
  
svg.append('line')
  .attr('x1', 400)
  .attr('y1', 95)
  .attr('x2', 255)
  .attr('y2', 95)
  .style("stroke-width", 1)
  .attr("opacity", 0.95)
  .attr('stroke', 'black');
  
svg.append('line')
  .attr('x1', 400)
  .attr('y1', 175)
  .attr('x2', 255)
  .attr('y2', 175)
  .style("stroke-width", 1)
  .attr("opacity", 0.95)
  .attr('stroke', 'black');
  
svg.append('line')
  .attr('x1', 400)
  .attr('y1', 255)
  .attr('x2', 255)
  .attr('y2', 255)
  .style("stroke-width", 1)
  .attr("opacity", 0.95)
  .attr('stroke', 'black');

draw_triangle(svg, 195, 75, 180);
draw_triangle(svg, 195, 155, 180);
draw_triangle(svg, 195, 235, 180);
draw_triangle(svg, 195, 315, 180);
draw_triangle(svg, 260, 95, -90);
draw_triangle(svg, 260, 175, -90);
draw_triangle(svg, 260, 255, -90);

for (var i = 0; i < 2; i += 1) {
	svg.append('rect')
	  .attr('x', 192 - i * 6)
	  .attr('y', 325 + i * 3)
	  .attr('width', 12)
	  .attr('height', 12)
	  .attr('stroke', 'black')
	  .attr("rx", 3)
	  .attr("opacity", 0.9)
	  .attr('fill', '#A4D8D8');
}

}

resgcn_head();

</script>

![](.)
*Fig. 6. ResGCN block schema. [GraphNorm](https://arxiv.org/pdf/2009.03294.pdf) layer normalizes node features over each graph in a batch.*

### Imitation learning

Now, we can train our network to imitate actions of best agents on a leaderboard. Each turn for each node with player shipyard on it, we have to decide for two things:

- Should we spawn, launch or do nothing?
- What is a number of ships to be spawned/launched?

One can train a network with cross-entropy loss for the action and mean-squared loss for the ships number. However, due to the fact that flight plan length depends on discretized $2 \log$ of fleet size, we can end up with errors like predicting number of ships to be $20$ with true number $21$ and thus having maximum plan length of $6.99$ and not being able to build a path with desired length of $7$. To avoid this we must split our policy into multiple classes, each representing maximum flight plan length:

- Do nothing
- Spawn
- Launch 1 ship (maximum length 1)
- Launch 2 ships (maximum length 2)
- Launch 3 or 4 ships (maximum length 3)
- ...
- Launch 666, 667, ..., 1096 ships (maximum length 14)

Here in total we have 16 classes, but this amount can be reduced or increased, depending on what engineer thinks is a reasonable number. To choose a fleet size we can set our target to be a ratio of ships to total amount of ships in a shipyard (or to maximum spawn number in case of 'spawn' action). 

<div id="gnn_losses" class="svg-container" align="center"></div> 

<script>

function losses() {

var svg = d3.select("#gnn_losses")
			  .append("svg")
			  .attr("width", 800)
			  .attr("height", 280);
			  
for (var i = 0; i < 6; i += 1) {
	svg.append('rect')
	  .attr('x', 50 - i * 6)
	  .attr('y', 85 + i * 3)
	  .attr('width', 12)
	  .attr('height', 12)
	  .attr('stroke', 'black')
	  .attr("rx", 3)
	  .attr("opacity", 0.9)
	  .attr('fill', '#A4D8D8');
}
  
svg.append('line')
  .attr('x1', 75)
  .attr('y1', 95)
  .attr('x2', 105)
  .attr('y2', 95)
  .style("stroke-width", 1)
  .attr("opacity", 0.95)
  .attr('stroke', 'black');
  
svg.append('rect')
  .attr('x', 105)
  .attr('y', 80)
  .attr('width', 120)
  .attr('height', 30)
  .attr('stroke', 'black')
  .attr("rx", 3)
  .attr("opacity", 0.6)
  .attr('fill', '#EDA137');
  
svg.append('text')
  .attr('x', 120)
  .attr('y', 100)
  .text("Linear (48, 16)")
  .style("font-size", "14px")
  .attr("font-family", "Arvo");

svg.append('line')
  .attr('x1', 225)
  .attr('y1', 95)
  .attr('x2', 250)
  .attr('y2', 95)
  .style("stroke-width", 1)
  .attr("opacity", 0.95)
  .attr('stroke', 'black');
  
draw_triangle(svg, 250, 95, 90);
  
for (var i = 0; i < 6; i += 1) {
	svg.append('rect')
	  .attr('x', 320)
	  .attr('y', 60 + i * 12)
	  .attr('width', 12)
	  .attr('height', 12)
	  .attr('stroke', 'black')
	  .attr("rx", 2)
	  .attr("opacity", 0.6)
	  .attr('fill', i == 1 ? '#65AD69' : 'white');
}

for (var i = 0; i < 6; i += 1) {
	svg.append('rect')
	  .attr('x', 270)
	  .attr('y', 60 + i * 12)
	  .attr('width', 12)
	  .attr('height', 12)
	  .attr('stroke', 'black')
	  .attr("rx", 2)
	  .attr("opacity", 0.6)
	  .attr('fill', '#EDA137');
}
  
svg.append('line')
  .attr('x1', 275)
  .attr('y1', 55)
  .attr('x2', 275)
  .attr('y2', 40)
  .style("stroke-width", 1)
  .attr("opacity", 0.95)
  .attr('stroke', 'black');
  
svg.append('line')
  .attr('x1', 325)
  .attr('y1', 55)
  .attr('x2', 325)
  .attr('y2', 40)
  .style("stroke-width", 1)
  .attr("opacity", 0.95)
  .attr('stroke', 'black');
  
svg.append('line')
  .attr('x1', 325)
  .attr('y1', 40)
  .attr('x2', 275)
  .attr('y2', 40)
  .style("stroke-width", 1)
  .attr("opacity", 0.95)
  .attr('stroke', 'black');
  
svg.append('line')
  .attr('x1', 300)
  .attr('y1', 40)
  .attr('x2', 300)
  .attr('y2', 30)
  .style("stroke-width", 1)
  .attr("opacity", 0.95)
  .attr('stroke', 'black');

svg.append('text')
  .attr('x', 275)
  .attr('y', 20)
  .text("CE Loss")
  .style("font-size", "14px")
  .attr("font-family", "Arvo");
  
svg.append('line')
  .attr('x1', 50)
  .attr('y1', 110)
  .attr('x2', 50)
  .attr('y2', 180)
  .style("stroke-width", 1)
  .attr("opacity", 0.95)
  .attr('stroke', 'black');
  
svg.append('line')
  .attr('x1', 50)
  .attr('y1', 180)
  .attr('x2', 315)
  .attr('y2', 180)
  .style("stroke-width", 1)
  .attr("opacity", 0.95)
  .attr('stroke', 'black');
  
svg.append('circle')
  .attr('cx', 325)
  .attr('cy', 180)
  .attr('r', 10)
  .attr('stroke', 'black')
  .attr('stroke-width', 1.25)
  .attr("opacity", 0.95)
  .attr('fill', "none");
  
svg.append('text')
  .attr('x', 322)
  .attr('y', 185)
  .text("||")
  .style("font-size", "14px")
  .attr("font-family", "Arvo");
  
svg.append('line')
  .attr('x1', 325)
  .attr('y1', 140)
  .attr('x2', 325)
  .attr('y2', 170)
  .style("stroke-width", 1)
  .attr("opacity", 0.95)
  .attr('stroke', 'black');
  
svg.append('line')
  .attr('x1', 335)
  .attr('y1', 180)
  .attr('x2', 365)
  .attr('y2', 180)
  .style("stroke-width", 1)
  .attr("opacity", 0.95)
  .attr('stroke', 'black');
  
svg.append('rect')
  .attr('x', 365)
  .attr('y', 165)
  .attr('width', 120)
  .attr('height', 30)
  .attr('stroke', 'black')
  .attr("rx", 3)
  .attr("opacity", 0.6)
  .attr('fill', '#EDA137');
  
svg.append('text')
  .attr('x', 380)
  .attr('y', 185)
  .text("Linear (64, 1)")
  .style("font-size", "14px")
  .attr("font-family", "Arvo");
  
svg.append('line')
  .attr('x1', 525)
  .attr('y1', 180)
  .attr('x2', 485)
  .attr('y2', 180)
  .style("stroke-width", 1)
  .attr("opacity", 0.95)
  .attr('stroke', 'black');
  
svg.append('line')
  .attr('x1', 325)
  .attr('y1', 190)
  .attr('x2', 325)
  .attr('y2', 220)
  .style("stroke-width", 1)
  .attr("opacity", 0.95)
  .attr('stroke', 'black');
  
svg.append('line')
  .attr('x1', 325)
  .attr('y1', 220)
  .attr('x2', 365)
  .attr('y2', 220)
  .style("stroke-width", 1)
  .attr("opacity", 0.95)
  .attr('stroke', 'black');
  
svg.append('rect')
  .attr('x', 365)
  .attr('y', 205)
  .attr('width', 120)
  .attr('height', 30)
  .attr('stroke', 'black')
  .attr("rx", 3)
  .attr("opacity", 0.6)
  .attr('fill', '#EDA137');
  
svg.append('text')
  .attr('x', 380)
  .attr('y', 225)
  .text("Linear (64, 1)")
  .style("font-size", "14px")
  .attr("font-family", "Arvo");
  
svg.append('line')
  .attr('x1', 485)
  .attr('y1', 220)
  .attr('x2', 525)
  .attr('y2', 220)
  .style("stroke-width", 1)
  .attr("opacity", 0.95)
  .attr('stroke', 'black');
  
draw_triangle(svg, 525, 180, 90);
draw_triangle(svg, 525, 220, 90);

svg.append('rect')
  .attr('x', 540)
  .attr('y', 142)
  .attr('width', 12)
  .attr('height', 12)
  .attr('stroke', 'black')
  .attr("rx", 2)
  .attr("opacity", 0.6)
  .attr('fill', '#65AD69');

svg.append('rect')
  .attr('x', 540)
  .attr('y', 173)
  .attr('width', 12)
  .attr('height', 12)
  .attr('stroke', 'black')
  .attr("rx", 2)
  .attr("opacity", 0.6)
  .attr('fill', '#EDA137');

svg.append('rect')
  .attr('x', 540)
  .attr('y', 213)
  .attr('width', 12)
  .attr('height', 12)
  .attr('stroke', 'black')
  .attr("rx", 2)
  .attr("opacity", 0.6)
  .attr('fill', '#EDA137');

svg.append('rect')
  .attr('x', 540)
  .attr('y', 246)
  .attr('width', 12)
  .attr('height', 12)
  .attr('stroke', 'black')
  .attr("rx", 2)
  .attr("opacity", 0.6)
  .attr('fill', 'white');
  
svg.append('line')
  .attr('x1', 575)
  .attr('y1', 180)
  .attr('x2', 555)
  .attr('y2', 180)
  .style("stroke-width", 1)
  .attr("opacity", 0.95)
  .attr('stroke', 'black');
  
svg.append('line')
  .attr('x1', 575)
  .attr('y1', 148)
  .attr('x2', 555)
  .attr('y2', 148)
  .style("stroke-width", 1)
  .attr("opacity", 0.95)
  .attr('stroke', 'black');
  
svg.append('line')
  .attr('x1', 575)
  .attr('y1', 180)
  .attr('x2', 575)
  .attr('y2', 148)
  .style("stroke-width", 1)
  .attr("opacity", 0.95)
  .attr('stroke', 'black');
  
svg.append('line')
  .attr('x1', 575)
  .attr('y1', 164)
  .attr('x2', 590)
  .attr('y2', 164)
  .style("stroke-width", 1)
  .attr("opacity", 0.95)
  .attr('stroke', 'black');
svg.append('line')
  .attr('x1', 575)
  .attr('y1', 220)
  .attr('x2', 555)
  .attr('y2', 220)
  .style("stroke-width", 1)
  .attr("opacity", 0.95)
  .attr('stroke', 'black');
  
svg.append('line')
  .attr('x1', 575)
  .attr('y1', 252)
  .attr('x2', 555)
  .attr('y2', 252)
  .style("stroke-width", 1)
  .attr("opacity", 0.95)
  .attr('stroke', 'black');
  
svg.append('line')
  .attr('x1', 575)
  .attr('y1', 220)
  .attr('x2', 575)
  .attr('y2', 252)
  .style("stroke-width", 1)
  .attr("opacity", 0.95)
  .attr('stroke', 'black');
  
svg.append('line')
  .attr('x1', 575)
  .attr('y1', 236)
  .attr('x2', 590)
  .attr('y2', 236)
  .style("stroke-width", 1)
  .attr("opacity", 0.95)
  .attr('stroke', 'black');
  
svg.append('text')
  .attr('x', 600)
  .attr('y', 170)
  .text("BCE Loss")
  .style("font-size", "14px")
  .attr("font-family", "Arvo");
  
svg.append('text')
  .attr('x', 600)
  .attr('y', 240)
  .text("BCE Loss")
  .style("font-size", "14px")
  .attr("font-family", "Arvo");
  
svg.append('text')
  .attr('x', 480)
  .attr('y', 130)
  .text("Expert spawn ships ratio")
  .style("font-size", "12px")
  .attr("font-family", "Arvo");
  
svg.append('text')
  .attr('x', 480)
  .attr('y', 275)
  .text("Expert launch ships ratio")
  .style("font-size", "12px")
  .attr("font-family", "Arvo");
  
svg.append('text')
  .attr('x', 340)
  .attr('y', 100)
  .text("Expert action")
  .style("font-size", "12px")
  .attr("font-family", "Arvo");
}

losses();

</script>

![](.)
*Fig. 7. Main policy heads. Encoded embedding is taken from node of current board graph representation, where shipyard is located on. This embedding is passed through linear layer to predict an expert action. In parallel, it is concatenated with one-hot representation of expert action and then goes through another feed-forward layers to predict expert spawn/launch ships ratio. On inference, $\arg\max$ of agent predictions can be used instead of expert action.*

Finally, we face path generation task. The idea is that for each node, starting from shipyard, we take all of its neighbors and itself, and predict next node in a flight plan. If it chooses node itself, we convert ships to a new shipyard (we can mask such action if fleet size is less than 50 or if fleet is already on a shipyard). To make the prediction dependent not only on the current node, but also on the entire sequence of previously selected nodes, we can use recurrent layers. It is also important to consider amount of space we have left for path generation.

<div id="pthsrch" class="svg-container" align="center"></div> 

<script>
function path_search() {

var svg = d3.select("#pthsrch")
			  .append("svg")
			  .attr("width", 800)
			  .attr("height", 350);

function draw_node(x, y, opacity=0.95) {
	svg.append('circle')
	  .attr('cx', x)
	  .attr('cy', y)
	  .attr('r', 10)
	  .attr('stroke', 'black')
	  .attr("opacity", opacity)
	  .attr('fill', '#A4D8D8');
}

var opacity = 0.04;
  
for (var i = 0; i < 5; i += 1) {
	
	draw_node(50, 45 + i * 30);
	
	for (var j = 0; j < 6; j += 1) {
		svg.append('rect')
		  .attr('x', 100 - j * 6)
		  .attr('y', 32 + j * 3 + i * 30)
		  .attr('width', 12)
		  .attr('height', 12)
		  .attr('stroke', 'black')
		  .attr("rx", 3)
		  .attr("opacity", 0.9)
		  .attr('fill', '#A4D8D8');
		  
		svg.append('rect')
		  .attr('x', 480 - j * 6)
		  .attr('y', 32 + j * 3 + i * 30)
		  .attr('width', 12)
		  .attr('height', 12)
		  .attr('stroke', 'black')
		  .attr("rx", 3)
		  .attr("opacity", 0.9)
		  .attr('fill', '#AFB2D8');
		  
		svg.append('rect')
		  .attr('x', 315 - j * 6)
		  .attr('y', 32 + j * 3 + i * 30)
		  .attr('width', 12)
		  .attr('height', 12)
		  .attr('stroke', 'black')
		  .attr("rx", 3)
		  .attr("opacity", 0.9)
		  .attr('fill', '#AFB2D8');
	}
	 
	svg.append("path")
	   .attr("stroke", "black")
	   .datum([{x: 120, y: 50 + i * 30}, {x: 170, y: 50 + i * 30}])
	   .attr("fill", "none")
	   .attr("d",  d3.line()
	       .curve(d3.curveBasis)
	       .x(function(d) { return d.x; })
	       .y(function(d) { return d.y; }));
	 
	svg.append("path")
	   .attr("stroke", "black")
	   .datum([{x: 500, y: 50 + i * 30}, {x: 550, y: 50 + i * 30}])
	   .attr("fill", "none")
	   .attr("d",  d3.line()
	       .curve(d3.curveBasis)
	       .x(function(d) { return d.x; })
	       .y(function(d) { return d.y; }));
	
	svg.append("path")
	   .attr("stroke", "black")
	   .datum([{x: 220, y: 50 + i * 30}, {x: 270, y: 50 + i * 30}])
	   .attr("fill", "none")
	   .attr("d",  d3.line()
	       .curve(d3.curveBasis)
	       .x(function(d) { return d.x; })
	       .y(function(d) { return d.y; }));
	       
	draw_triangle(svg, 270, 50 + i * 30, 90);

	draw_triangle(svg, 550, 50 + i * 30, 90);
	
	svg.append('rect')
	  .attr('x', 400)
	  .attr('y', 35 + i * 30)
	  .attr('width', 40)
	  .attr('height', 20)
	  .attr('stroke', 'black')
	  .attr("rx", 10)
	  .attr("opacity", 0.9)
	  .attr('fill', '#AFB2D8');

	svg.append('line')
	  .attr('x1', 570)
	  .attr('y1', 50 + i * 30)
	  .attr('x2', 600)
	  .attr('y2', 50 + i * 30)
	  .style("stroke-width", 2)
	  .style('stroke-dasharray', ('2,10'))
	  .attr("opacity", 0.95)
	  .attr('stroke', 'black');
	}

svg.append('text')
  .attr('x', 41)
  .attr('y', 50)
  .text("W")
  .style("font-size", "14px")
  .attr("font-family", "Arvo");
  
svg.append('text')
  .attr('x', 407)
  .attr('y', 50)
  .text("NW")
  .style("font-size", "14px")
  .attr("font-family", "Arvo");
  
svg.append('text')
  .attr('x', 44)
  .attr('y', 80)
  .text("N")
  .style("font-size", "14px")
  .attr("font-family", "Arvo");
  
svg.append('text')
  .attr('x', 412)
  .attr('y', 80)
  .text("N1")
  .style("font-size", "14px")
  .attr("font-family", "Arvo");
  
svg.append('text')
  .attr('x', 46)
  .attr('y', 110)
  .text("S")
  .style("font-size", "14px")
  .attr("font-family", "Arvo");
  
svg.append('text')
  .attr('x', 410)
  .attr('y', 110)
  .text("NS")
  .style("font-size", "14px")
  .attr("font-family", "Arvo");
  
svg.append('text')
  .attr('x', 45)
  .attr('y', 140)
  .text("E")
  .style("font-size", "14px")
  .attr("font-family", "Arvo");
  
svg.append('text')
  .attr('x', 410)
  .attr('y', 140)
  .text("NE")
  .style("font-size", "14px")
  .attr("font-family", "Arvo");
  
svg.append('text')
  .attr('x', 45)
  .attr('y', 170)
  .text("C")
  .style("font-size", "14px")
  .attr("font-family", "Arvo");
  
svg.append('text')
  .attr('x', 410)
  .attr('y', 170)
  .text("NC")
  .style("font-size", "14px")
  .attr("font-family", "Arvo");
  
svg.append('rect')
  .attr('x', 170)
  .attr('y', 35)
  .attr('width', 50)
  .attr('height', 150)
  .attr('stroke', 'black')
  .attr("rx", 3)
  .attr("opacity", 0.6)
  .attr('fill', '#BA8CD7');
 
svg.append('text')
  .attr('x', 175)
  .attr('y', 115)
  .text("LSTM")
  .style("font-size", "14px")
  .attr("font-family", "Arvo");
       
svg.append('rect')
  .attr('x', 65)
  .attr('y', 151)
  .attr('width', 50)
  .attr('height', 30)
  .attr('stroke', 'black')
  .attr("rx", 3)
  .style('stroke-dasharray', ('2,3'))
  .attr('fill', 'none');
       
svg.append('rect')
  .attr('x', 280)
  .attr('y', 25)
  .attr('width', 50)
  .attr('height', 160)
  .attr('stroke', 'black')
  .attr("rx", 3)
  .style('stroke-dasharray', ('2,3'))
  .attr('fill', 'none');
	 
svg.append('rect')
  .attr('x', 65)
  .attr('y', 205)
  .attr('width', 50)
  .attr('height', 30)
  .attr('stroke', 'black')
  .attr("rx", 3)
  .attr("opacity", 0.6)
  .attr('fill', '#EDA137');
 
svg.append('text')
  .attr('x', 78)
  .attr('y', 224)
  .text("Key")
  .style("font-size", "14px")
  .attr("font-family", "Arvo");
		  
svg.append("path")
   .attr("stroke", "black")
   .datum([{x: 90, y: 180}, {x: 90, y: 205}])
   .attr("fill", "none")
   .attr("d",  d3.line()
       .x(function(d) { return d.x; })
       .y(function(d) { return d.y; }));
       
svg.append("path")
   .attr("stroke", "black")
   .datum([{x: 305, y: 185}, {x: 305, y: 205}])
   .attr("fill", "none")
   .attr("d",  d3.line()
       .x(function(d) { return d.x; })
       .y(function(d) { return d.y; }));
	 
svg.append('rect')
  .attr('x', 280)
  .attr('y', 205)
  .attr('width', 50)
  .attr('height', 30)
  .attr('stroke', 'black')
  .attr("rx", 3)
  .attr("opacity", 0.6)
  .attr('fill', '#EDA137');
 
svg.append('text')
  .attr('x', 285)
  .attr('y', 224)
  .text("Query")
  .style("font-size", "14px")
  .attr("font-family", "Arvo");
 
svg.append('rect')
  .attr('x', 165)
  .attr('y', 205)
  .attr('width', 60)
  .attr('height', 30)
  .attr('stroke', 'black')
  .attr("rx", 3)
  .attr("opacity", 0.6)
  .attr('fill', 'none');
  
svg.append('text')
  .attr('x', 167)
  .attr('y', 224)
  .text("MatMul")
  .style("font-size", "14px")
  .attr("font-family", "Arvo");
  
svg.append("path")
   .attr("stroke", "black")
   .datum([{x: 280, y: 220}, {x: 225, y: 220}])
   .attr("fill", "none")
   .attr("d",  d3.line()
       .x(function(d) { return d.x; })
       .y(function(d) { return d.y; }));
  
svg.append("path")
   .attr("stroke", "black")
   .datum([{x: 165, y: 220}, {x: 115, y: 220}])
   .attr("fill", "none")
   .attr("d",  d3.line()
       .x(function(d) { return d.x; })
       .y(function(d) { return d.y; }));
  
svg.append("path")
   .attr("stroke", "black")
   .datum([{x: 195, y: 235}, {x: 195, y: 270}])
   .attr("fill", "none")
   .attr("d",  d3.line()
       .x(function(d) { return d.x; })
       .y(function(d) { return d.y; }));
  
svg.append('text')
  .attr('x', 145)
  .attr('y', 255)
  .text("Softmax")
  .style("font-size", "12px")
  .attr("font-family", "Arvo");
  
draw_triangle(svg, 195, 270, 180);

svg.append('rect')
  .attr('x', 170)
  .attr('y', 300)
  .attr('width', 10)
  .attr('height', 30)
  .attr('stroke', 'black')
  .attr("rx", 1)
  .attr("opacity", 0.6)
  .attr('fill', '#EDA137');
  
svg.append('rect')
  .attr('x', 180)
  .attr('y', 280)
  .attr('width', 10)
  .attr('height', 50)
  .attr('stroke', 'black')
  .attr("rx", 1)
  .attr("opacity", 0.6)
  .attr('fill', '#EDA137');
  
svg.append('rect')
  .attr('x', 190)
  .attr('y', 310)
  .attr('width', 10)
  .attr('height', 20)
  .attr('stroke', 'black')
  .attr("rx", 1)
  .attr("opacity", 0.6)
  .attr('fill', '#EDA137');
  
svg.append('rect')
  .attr('x', 200)
  .attr('y', 325)
  .attr('width', 10)
  .attr('height', 5)
  .attr('stroke', 'black')
  .attr("rx", 1)
  .attr("opacity", 0.6)
  .attr('fill', '#EDA137');
  
svg.append('rect')
  .attr('x', 210)
  .attr('y', 305)
  .attr('width', 10)
  .attr('height', 25)
  .attr('stroke', 'black')
  .attr("rx", 1)
  .attr("opacity", 0.6)
  .attr('fill', '#EDA137');
  
svg.append('text')
  .attr('x', 140)
  .attr('y', 345)
  .text("Move probabilities")
  .style("font-size", "12px")
  .attr("font-family", "Arvo");
  
svg.append("path")
   .attr("stroke", "black")
   .datum([{x: 330, y: 80}, {x: 380, y: 80}])
   .attr("fill", "none")
   .attr("d",  d3.line()
       .x(function(d) { return d.x; })
       .y(function(d) { return d.y; }));
       
draw_triangle(svg, 380, 80, 90);

svg.append("path")
   .attr("stroke", "black")
   .datum([{x: 230, y: 310}, {x: 410, y: 310}, {x: 410, y: 265}, {x: 395, y: 265}])
   .attr("fill", "none")
   .attr("d",  d3.line()
       .x(function(d) { return d.x; })
       .y(function(d) { return d.y; }));

svg.append("path")
   .attr("stroke", "black")
   .datum([{x: 360, y: 260}, {x: 360, y: 100}])
   .attr("fill", "none")
   .attr("d",  d3.line()
       .x(function(d) { return d.x; })
       .y(function(d) { return d.y; }));
       
draw_triangle(svg, 360, 100, 0);
  
svg.append('text')
  .attr('x', 325)
  .attr('y', 285)
  .text("Expert move")
  .style("font-size", "12px")
  .attr("font-family", "Arvo");
  
for (var i = 0; i < 5; i += 1) {
	svg.append('rect')
	  .attr('x', 330 + i * 12)
	  .attr('y', 260)
	  .attr('width', 12)
	  .attr('height', 12)
	  .attr('stroke', 'black')
	  .attr("rx", 2)
	  .attr("opacity", 0.6)
	  .attr('fill', i == 1 ? '#65AD69' : 'white');
}

svg.append("path")
   .attr("stroke", "black")
   .datum([{x: 410, y: 290}, {x: 430, y: 290}])
   .attr("fill", "none")
   .attr("d",  d3.line()
       .x(function(d) { return d.x; })
       .y(function(d) { return d.y; }));
  
svg.append('text')
  .attr('x', 435)
  .attr('y', 295)
  .text("CE Loss")
  .style("font-size", "14px")
  .attr("font-family", "Arvo");
  
  
for (var i = 0; i < 5; i += 1) {
	svg.append('rect')
	  .attr('x', 110 + i * 12)
	  .attr('y', 2)
	  .attr('width', 12)
	  .attr('height', 12)
	  .attr('stroke', 'black')
	  .attr("rx", 2)
	  .attr("opacity", 0.6)
	  .attr('fill', i == 4 ? '#65AD69' : 'white');
	  
}

svg.append('text')
  .attr('x', 180)
  .attr('y', 12)
  .text("Space left")
  .style("font-size", "12px")
  .attr("font-family", "Arvo");
  
for (var i = 0; i < 5; i += 1) {
	svg.append('rect')
	  .attr('x', 490 + i * 12)
	  .attr('y', 2)
	  .attr('width', 12)
	  .attr('height', 12)
	  .attr('stroke', 'black')
	  .attr("rx", 2)
	  .attr("opacity", 0.6)
	  .attr('fill', i == 3 ? '#65AD69' : 'white');
	  
}

svg.append('text')
  .attr('x', 560)
  .attr('y', 12)
  .text("Space left")
  .style("font-size", "12px")
  .attr("font-family", "Arvo");

}

path_search();
  
</script>

![](.)
*Fig. 8. Path policy head. We start at shipyard node. At each step all neighbor candidates, including node itself, are concatenated with one-hot representation of a 'space left' for plan generation. Then they are passed through recurrent layer. Received neighbor embeddings are passed through linear layer $\operatorname{Query}$ and multiplied with node features passed through $\operatorname{Key}$ layer. Resulting vector represents probabilities of each neighbor to be the next node. Then we move along path defined by an expert agent and repeat the same procedure.*

Now having probabilities for each move, we can generate path by greedy or beam search, cutting path when it reaches shipyard or when there is no more space left. There is also a case, when path stuck in a loop like 'NW999..', and it can take forever until we won't have enough space. For such thing it is useful to make a stop by some maximum-moves threshold.

#### Bonus: reinforcement learning

In my experiments, imitation learning was good enough for an agent to start making good moves, however, there was constant instability on inference. At some point an agent could make a ridiculously stupid move (with small probability, but nevertheless), effecting balance of the game dramatically. This move could never be made by an expert agent, therefore we end up with board state which neural network didn't see in training data. So the chance of making wrong moves increases at each step and finally imitation agent loses. 

To tackle this issue I've tried off-policy reinforcement learning. The architecture of policy neural network was the same, critic neural network had similar but smaller encoder. Reward was +1 for winning, -1 for losing. I used TD($\lambda$) target $v_s$ for value prediction and policy loss with clipped importance sampling:

$$
\mathcal{L}_{value} = (v_s - V_\omega(s))^2, \quad \mathcal{L}_{policy} = -\log \pi_\theta(a|s) \min \big(1, \frac{\pi_\theta(a|s)}{\beta(a|s)} \big) A_\omega(a, s),
$$

where $A_\omega(a, s)$ is an advantage, obtained by [UPGO](https://paperswithcode.com/method/alphastar) method.

RL helped to stabilize inference, but sadly it didn't go far beyond that. New agent was able to beat rule-based agents, which were on top of Kore Beta competition) ~80% of the time.


### Results and conclusions

Unfortunately, best of my agents were able to reach only up to 30-ish places, but still, it was a fun ride. What could've been done better? Here is my list:

- Better feature engineering. The input state which I was using contained a lot of data, however not all of the important information could have been distilled by neural network. Looking ahead for 12 steps made input graph enormously huge: $21 \times 21 \times 12 \times n$, where $n$ is a feature dimension for every node. And still there were a lot of opponent flight plans with bigger time-length than 12 steps.
- Larger neural network. It is related to previous issue: construction of huge input state at every turn was taking a lot of time on inference. In order to fit Kaggle time requirements I had to reduce network size. I'm convinced that bigger dimensions could lead to better results.
- Reinforcement learning experiments. It is well known that it takes a lot of time to make RL work. Training is slow and unstable most of the time. It is crucial to meticulously watch training procedure and analyze the results.

Hopefully I'll take this experience into account in the next Kaggle simulation competition.