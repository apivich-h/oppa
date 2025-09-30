# Configuring Parallel Training of Neural Networks using Bayesian Optimization

## Abstract

> Training of modern large neural networks (NNs) is often done in parallel across multiple GPUs. While there are existing parallel training frameworks which easily allow NN training using multi-dimensional parallelism, the challenge remains in optimizing the balance between size of each parallelism dimensions, and in tuning the hyperparameters within these parallelism dimensions. Due to a large number of possible parallelism configurations (PCs) for a given training scenario, it is infeasible to perform exhaustive search over all candidates. Existing PC optimization methods typically either require conducting training trials on a large number of PCs, each of which can be expensive to perform, or rely on an approximate cost model which may be inaccurate and hardware-specific. To overcome these issues, we present OPPA, which combines constrained Bayesian optimization methods with prior knowledge in the form of a parallelism-informed prior belief, to obtain an optimal PC using a minimal number of NN training trials. We also propose a framework for early termination of trails involving suboptimal PCs, whose efficiency gains can be theoretically justified. We show that OPPA finds an optimal PC more efficiently for training transformers on various multi-GPU systems compared to the methods used in existing parallel training frameworks.


## Instructions

- Setup scripts can be found in `reqs/`. There are different setup methods for different parallelized training frameworks.

- The scripts for running the experiments can be found in `examples/`. See the `run.py` files in each respective folders.
