I want to see how I can use CUDA in conjunction with parallel RL algorithms.
The key issue is that the simulation typically can't be ran on the GPU, but perhaps some training steps can be ran in parallel over histories.

Partly this is just a refresher in CUDA interface doing something fun.

I'll start by trying out Mancala

2 parts to the code

CPU bound playable simulator - allowing us to watch games/play against the resulting AI

GPU bound trainer - runs many games in parallel, playing the selected agents against each other, then runs a training step on the aggregate results


agents are built and tested in CPU bound first then converted to a version for the GPU type

AI data (where relevant) is saved to a Protobuf.