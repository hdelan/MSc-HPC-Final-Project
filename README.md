Networks are ubiquitous structures, appearing in virtually all scientific or non-scientific fields of study. Networks may represent the interactions between animals in an ecosystem, links between roads in a network, academics citing other academics, users interacting in a social network, as well as countless other representations.  

In the study and analysis of networks it is of great importance to identify central and non-central nodes. There are countless ways of measuring the centrality of nodes in a network, all with slightly different interpretations of what it means to be central. Finding highly central and poorly central nodes is essential in maintaining and managing real life graphs and networks. 

The aim of this project is to compute the centrality of nodes in an undirected graph ($A^T=A$). The action of the exponential matrix function $f(A)x = e^Ax$ will be used as a measure of node centrality. The desired quantity $e^Ax$ will be approximated without forming  $e^A$ explicitly, instead using Krylov subspace methods, most notably the Lanczos method. This makes computing $e^Ax$ scalable to the limits of the memory hardware, which will be tested in this project. 

This project computes the action of the matrix exponential on a vector (usually of ones) both in serial and in parallel.

The accompanying report can be found at ```writeup.pdf```.

In order to run code, data needs to be downloaded from 

```https://drive.google.com/drive/folders/1HdyMdnjphMtafk8TBbWc0L-2V4OeWlmj?usp=sharing```

and untarred. Please replace the empty ```data/``` directory with the untarred ```data.tar```.

Find serial implementation in ```serial/``` 
- For details of how to run code in ```serial/``` see ```serial/README.md```
