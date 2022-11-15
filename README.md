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

By default, experiments are run with 200 epochs of representation learning, 150 epochs of linear evaluation with
learning rates of 0.001 and 0.01, respectively.

## Denoising Autoencoder

This experiment type is a single denoising autoencoder, meaning that if $x$ is input image, $\tilde{x} \sim D(x)$ where $D$ is
an augmentation distribution over the input space. The let's call the output of the network $\tilde{x}_{out}$. The loss function of this type of autoencoder is as follows:

$$
L(x) = \text{MSE}(x, \tilde{x}_{out})
$$

The idea behind this network architecture is that the network is forced to learn the "true" representation of the data
by learning what how the augmentations warp the input space and reversing it. 

(ADD FIGURE)

To run this experiment, use the following code:
```markdown
python main.py --ssl_type ("AE-S", "D", "NL") 
```

## Parallel Autoencoder Architecture

To run this experiment, use the following code:
```markdown
python main.py --ssl_type ("AE-P", "ND", "NL") 
```

## Parallel Denoising Autoencoder

To run this experiment, use the following code:
```markdown
python main.py --ssl_type ("AE-P", "D", "NL") 
```

## SimCLR

To run this experiment, use the following code:
```markdown
python main.py --ssl_type ("SimCLR", "ND", "NL") 
```

## Alpha Experiments

To run this experiment, use the following code:
```markdown
python main.py --exp_type strength
```


## Augmentation Strength Experiments

To run this experiment, use the following code:
```markdown
python main.py --exp_type alpha
```

# Results

