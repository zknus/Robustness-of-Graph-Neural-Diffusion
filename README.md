
This repository contains the implementation to reproduce the numerical experiments 
of the **NEURIPS 2022** paper [On the Robustness of Graph Neural Diffusion to Topology Perturbations](https://arxiv.org/abs/2209.07754)

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

## Citation
If you found our work useful in your research, please cite our paper at:
```bibtex
@INPROCEEDINGS{SonKanWan:C22,
						author      = {Yang Song and Qiyu Kang and Sijie Wang and Kai Zhao and Wee Peng Tay},
						title       = {On the Robustness of Graph Neural Diffusion to Topology Perturbations},
						booktitle   = {Advances in Neural Information Processing Systems (NeurIPS)},
						month       = {Nov.},
						year        = {2022},
						address     = {New Orleans, USA},
						}
```
(Also consider starring the project on GitHub.)
