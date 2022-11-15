# Self-Supervised Learning Notebook

# Overview

This repo can be used for running self-supervised learning experiments. There are three main architectures that can be
run from the notebook:

1) Autoencoder
2) "Denoising" Autoencoder
3) SimCLR Encoder

Both of the autoencoder structures can be trained with a "parallel" architecture, where two image augmentations are run
through the network and a reconstruction loss is placed on their embeddings.

For all experiments, a simple Conv6 network spelled out by (Yali paper citation) is used as an encoder. The networks
can also be trained in a layer-wise fashion, which changes which decoder is used. In a non-layer-wise setting, the
decoder is simpler, though in the layer-wise setting, the decoder must be symmetric to the encoder so a more complex
decoder must be used.

# Experiments

All experiments consist of 2 stages:
- Unsupervised representation learning: Architecture of choice is run in an unsupervised fashion in order to orient the
network towards a quality embedding space
- Evaluation stage: New dataset is embedded and tested via Linear Evaluation or k-means clustering.

## Denoising Autoencoder

This experiment type is a single denoising autoencoder, meaning that if $x$ is input image, $\tilde{x} \sim D(x)$ where $D$ is
an augmentation distribution over the input space. The let's call the output of the network $\tilde{x}_{out}$. The loss function of this type of autoencoder is as follows:

$$
L(x) = \text{MSE}(x, \tilde{x}_{out})
$$

(ADD FIGURE)

## Parallel Autoencoder Architecture

This 
(ADD FIGURE)

## Parallel Denoising Autoencoder



## SimCLR

# Results

