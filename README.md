# Robustness-of-Graph-Neural-Diffusion
On the Robustness of Graph Neural Diffusion to Topology Perturbations ---Accepted by NIPS 2022   
Code will release soon.

## Running the experiments
The codes are written based on 
- https://github.com/twitter-research/graph-neural-pde
- https://github.com/THUDM/grb

### Experiments
For example to run 
```
cd src
python run_GNN2.py 
```
Saved models are available in ./saved_models2.

### Requirements

* scipy==1.5.2
* numpy==1.19.1
* torch==1.8.0
* networkx==2.5
* pandas~=1.2.3
* cogdl~=0.3.0.post1
* torch-cluster==1.5.9
* torch-geometric==1.7.0
* torch-scatter==2.0.6
* torch-sparse==0.6.9
* torch-spline-conv==1.2.1
* torchdiffeq==0.2.1
