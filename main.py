import torch

from ColabExport.TrainModel import load_data, experiments, networks
from ColabExport import exp_config

experiments.print_model_architecture(networks.USL_Conv6_CIFAR1)
