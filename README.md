# Self-Supervised Learning Notebook

## Overview

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

## Experiments

All experiments consist of 2 stages:
- Unsupervised representation learning: Architecture of choice is run in an unsupervised fashion in order to orient the
network towards a quality embedding space
- Evaluation stage: New dataset is embedded and tested via Linear Evaluation or k-means clustering.

By default, experiments are run with 400 epochs of representation learning, 300 epochs of linear evaluation with
learning rates of 0.0001 and 0.001, respectively. Experiment conditions (usl epochs, le epochs, learning rates, etc.) 
can be configured through command line parser on any experiment.

### Denoising Autoencoder

This experiment type is a single denoising autoencoder, meaning that if $x$ is input image, $\tilde{x} \sim D(x)$ where 
$D$ is an augmentation distribution over the input space. The let's call the output of the network $\tilde{x}_{out}$. 
The loss function used for this autoencoder in the notebook is:

$$
L(x) = L_\text{rec} \\
$$


$$
L(x) = ||x - p_\theta (\tilde{x})||_2
$$

The idea behind this network architecture is that the network is forced to learn the "true" representation of the data
by learning what how the augmentations warp the input space and reversing it. 

(ADD FIGURE)

To run this experiment, use the following code:
```markdown
python main.py --usl_type ae_single --denoising True
```

### Parallel Autoencoder Architecture

In this type of self-supervised learning, two images are passed through the autoencoder between each gradient step.
After the two images are run through the autoencoder, the loss function is the sum of the reconstruction losses plus a
term that penalizes the distance between the two embeddings. Generally, this loss function is an MSE loss, though it can
also be run as an L1 or even SimCLR or Barlow Twins loss on the image embeddings. This type of experiment introduces
the hyperparameter $\alpha$ which controls the weight that the embedding loss is given compared to the reconstruction
losses. The loss function used notebook for this architecture is:

$$ 
L(x) = L_\text{rec,1} + L_\text{rec,2} + \alpha L_\text{emb}
$$


$$ 
L(x) = \text{MSE}(x_1, p_\theta (x_1)) + \text{MSE}(x_2, p_\theta (x_2)) + \alpha \text{MSE}(p_\theta (x_1), 
p_\theta (x_2))
$$

Alpha is run at orders of magnitude between 0.00001 and 10. Results are presented in the results section.


To run this experiment at a specific alpha value, use the following code:
```markdown
python main.py --usl_type ae_parallel --denoising False --alpha ALPHA
```

### Parallel Denoising Autoencoder

This architecture combines the first two, running a denoising autoencoder with the parallel loss function and a penalty
on the embeddings. The loss used in the notebook for this architecture is as follows:

$$ 
L(x) = L_\text{rec,1} + L_\text{rec,2} + \alpha L_\text{emb}
$$


$$ L(x) = ||(x_ 1, p_\theta (\tilde{x}_ 1))||_ 2 + ||(x_2, p_\theta (\tilde{x}_ 2)))||_ 2 + \alpha ||(p_\theta 
(\tilde{x}_ 1)), p_\theta (\tilde{x}_ 2)))||_ 2
$$


To run this experiment, use the following code:
```markdown
python main.py --usl_type ae_parallel --denoising True
```

### SimCLR

To run this experiment, use the following code:
```markdown
python main.py --usl_type simclr
```

### Alpha Experiments

To run this experiment, use the following code:
```markdown
python main.py --exp_type alpha
```

### Augmentation Strength Experiments

To run this experiment, use the following code:
```markdown
python main.py --exp_type strength
```

## Results

