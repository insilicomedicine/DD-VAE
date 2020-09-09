# Deterministic Decoding for Discrete Data in Variational Autoencoders

Variational autoencoders are prominent generative models for modeling discrete data. However, with flexible decoders, they tend to ignore the latent codes.  In this paper, we study a VAE model with a deterministic decoder (DD-VAE) for sequential data that selects the highest-scoring tokens instead of sampling. Deterministic decoding solely relies on latent codes as the only way to produce diverse objects, which improves the structure of the learned manifold. To implement DD-VAE, we propose a new class of bounded support proposal distributions and derive Kullback-Leibler divergence for Gaussian and uniform priors. We also study a continuous relaxation of deterministic decoding objective function and analyze the relation of reconstruction accuracy and relaxation parameters. We demonstrate the performance of DD-VAE on multiple datasets, including molecular generation and optimization problems.

For more details, please refer to the [full paper](https://arxiv.org/abs/2003.02174).

### Repository
In this repository, we provide all code and data that is necessary to reproduce all the results from the paper. To reproduce the experiments, we recommend using Docker image built using a provided `Dockerfile`:
```{bash}
nvidia-docker build -t dd_vae .
nvidia-docker run -it --shm-size 10G --network="host" --name dd_vae -w=/code/dd_vae dd_vae
```
All the code will be available inside `/code/dd_vae` folder. For more details on using Docker, please refer to [Docker manual](https://docs.docker.com/)

You can also install `dd_vae` locally by running `python setup.py install` command.

### Reproducing the experiments
You can train any model using `train.py` script. This scripts takes only two arguments: `--config` (path to .ini file that sets up the experiment) and `--device` (PyTorch-style device naiming such as `cuda:0`). We provide all configuration files in `configs/` folder. For each experiment we provide a separate Jupyter Notebook, where you will find further instructions to reproduce the experiments:
* [Synthetic](./synthetic.ipynb)
* [MNIST](./mnist.ipynb)
* [MOSES (metrics)](./moses_prepare_metrics.ipynb), [MOSES (plots)](./moses_plots.ipynb)
* [ZINC](./bo.ipynb)

### How to cite
```
@InProceedings{pmlr-v108-polykovskiy20a,
  title = {Deterministic Decoding for Discrete Data in Variational Autoencoders},
  author = {Polykovskiy, Daniil and Vetrov, Dmitry},
  booktitle = {Proceedings of the Twenty Third International Conference on Artificial Intelligence and Statistics},
  pages = {3046--3056},
  year = {2020},
  editor = {Silvia Chiappa and Roberto Calandra},
  volume = {108},
  series = {Proceedings of Machine Learning Research}, address = {Online},
  month = {26--28 Aug},
  publisher = {PMLR}
}
```
