---
layout: post
title: 'Unfolding AlphaFold'
date: 2023-01-01 11:00 +0800
categories: [Deep Learning in Biology, Protein Folding]
tags: [alphafold, protein-folding, protein-structure-prediction, deepmind]
math: true
enable_d3: true
published: false
---

> It was a breakthrough moment for science when AlphaFold 2, a deep-learning-based artificial intelligence system, was able to solve a problem that had perplexed researchers for more than 50 years. This remarkable achievement was the result of a decade-long effort from an international team of scientists and engineers from DeepMind, and it has the potential to revolutionize the field of protein folding and open the doors to a new era of medical discoveries.

- Sources:
	- [Highly accurate protein structure prediction with AlphaFold](https://www.nature.com/articles/s41586-021-03819-2) 
	- [AlphaFold: Using AI for scientific discovery](https://www.deepmind.com/blog/alphafold-using-ai-for-scientific-discovery-2020)
	- [Protein Structure](https://www.nature.com/scitable/topicpage/protein-structure-14122136/)
	- [A few words about AlphaFold 2](https://yakovlev.me/para-slov-za-alphafold2/) (in Russian)

### Protein structure prediction

**Proteins** are large, complex molecules essential to all of life. Nearly every function that our body performs - contracting muscles, sensing light, or turning food into energy - relies on proteins, and how they move and change. Proteins are built from small organic molecules, called **amino acids** or **residues**, which are linked together by **peptide bonds**, thereby forming a long chain. There are 20 different types of amino acids, so a protein can be represented as a string from an alphabet of 20 letters. The linear sequence of amino acids within a protein is considered the **primary structure** of the protein.

![Amino Acids]({{'/assets/img/amino-acids.png'|relative_url}})
*Building blocks of our life. Source: DeepMind website*

The primary structure of a protein drives the folding and intramolecular bonding of the linear amino acid chain, which ultimately determines the protein's unique three-dimensional shape. Amino acids in neighboring regions of the protein chain interact with each other and cause certain patterns of folding to occur. Known as alpha helices and beta sheets, these stable folding patterns make up the **secondary structure** of a protein. Most proteins contain multiple helices and sheets, in addition to other less common patterns. The ensemble of formations and folds in a single linear chain of amino acids — sometimes called a **polypeptide** — constitutes the **tertiary structure** of a protein. Finally, the **quaternary structure** of a protein refers to those macromolecules with multiple polypeptide chains or subunits. 

![Protein Folding]({{'/assets/img/protein-folding.svg'|relative_url}})
*Complex 3D shapes emerge from a string of amino acids. Source: DeepMind website*

Protein structures help us understand how proteins work and what function they fulfil. For example, antibody proteins utilised by our immune systems are ‘Y-shaped’, and form unique hooks. By latching on to viruses and bacteria, these antibody proteins are able to detect and tag disease - causing microorganisms for elimination. Collagen proteins are shaped like cords, which transmit tension between cartilage, ligaments, bones, and skin. Other types of proteins include Cas9, which, using CRISPR sequences as a guide, act like scissors to cut and paste sections of DNA; antifreeze proteins, whose 3D structure allows them to bind to ice crystals and prevent organisms from freezing; and ribosomes, which act like a programmed assembly line, helping to build proteins themselves.

The recipes for those proteins - called **genes** - are directly encoded in our DNA. If you are able to read a DNA sequence, you know which protein will be synthesized from it. An error in the genetic recipe may result in a malformed protein, which could result in disease or death for an organism. Many diseases, therefore, are fundamentally linked to proteins.

But just because you know the genetic recipe for a protein doesn’t mean you automatically know its shape. DNA only contains information about the sequence of amino acids - not how they fold into shape. The bigger the protein, the more difficult it is to model, because there are more interactions between amino acids to take into account. As demonstrated by [Levinthal’s paradox](https://en.wikipedia.org/wiki/Levinthal%27s_paradox), it would take longer than the age of the known universe to randomly enumerate all possible configurations of a typical protein before reaching the true 3D structure - yet proteins themselves fold spontaneously, within milliseconds. Predicting how these chains will fold into the intricate 3D structure of a protein is what’s known as the **protein-folding problem** - a challenge that scientists have worked on for decades.

#### CASP competition

In 1958, Sir John Kendrew, an English biochemist, determined the world’s first protein structure by decoding myoglobin – a protein found in the heart and skeletal muscle tissue of mammals. Since then through an enormous experimental effort the structures of around 100,000 unique protein have been determined. But this is a tiny fraction, compared to the over 200 millions of known protein sequences. The reason is that experimental protein structure determination is an extremely hard task.
 
Proteins are too small to visualize, even with a microscope. So, scientists must use indirect methods to figure out what they look like and how they are folded. The most common methods used to study protein structures are cryo-electron microscopy, nuclear magnetic resonance and X-ray crystallography, but each method depends on a lot of trial and error, which can take years of work, and cost tens or hundreds of thousands of dollars per protein structure. 

In 1994, computational biologist John Moult set up an international competition, called the [Critical Assessment of protein Structure Prediction (CASP)](https://predictioncenter.org/index.cgi). During this competition, participants are offered protein sequences whose structures have already been obtained, but not yet published. Competitors, given only these sequences, must predict the structures of these proteins. Prediction tasks are divided into categories according to complexity (depending on how different the protein is from all existing ones) and solution methods (fully automated or with varying degrees of human participation).

The most difficult category is *ab initio* modeling, i.e. structure prediction from scratch, without additional information about similar proteins. For a long time, there were practically no breakthroughs in this area, until CASP13 in 2018. The first place in this competition was take by team called 'A7D', the results of which were much better than the results of other competitors. It turned out that under this name was hidden the new AlphaFold algorithm from the company DeepMind, which had already made a splash by that time with victories in the fields of board game Go.

### AlphaFold

#### Network architecture

![AlphaFold Detailed]({{'/assets/img/alphafold-network.png'|relative_url}})
*AlphaFold 2 architecture. Source: AlphaFold 2 paper*