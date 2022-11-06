from TrainModel import networks
from Experiments import experiments

import torch

print("========Running Network========")
print(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))

experiments.ssl_experiment_setup(model_type=networks.USL_Conv6_CIFAR1,
                                 exp_type=("SimCLR", "ND", "NL"),
                                 num_epochs_usl=200,
                                 num_epochs_le=150,
                                 save_embeddings=True)

'''
experiments.test_alpha_parallel([0.0001, 0.001, 0.01, 0.1, 0.0, 0.1, 1.0, 10])

experiments.ssl_experiment_setup(model_type=networks.USL_Conv6_CIFAR1,
                                 exp_type=("AE-S", "D", "NL"),
                                 num_epochs_usl=200,
                                 num_epochs_le=150,
                                 save_embeddings=True)

experiments.ssl_experiment_setup(model_type=networks.USL_Conv6_CIFAR1,
                                 exp_type=("AE-S", "ND", "NL"),
                                 num_epochs_usl=200,
                                 num_epochs_le=150,
                                 save_embeddings=True)

experiments.ssl_experiment_setup(model_type=networks.USL_Conv6_CIFAR_Sym,
                                 exp_type=("AE-S", "D", "L"),
                                 num_epochs_usl=200,
                                 num_epochs_le=150,
                                 save_embeddings=True)

experiments.ssl_experiment_setup(model_type=networks.USL_Conv6_CIFAR_Sym,
                                 exp_type=("AE-S", "ND", "L"),
                                 num_epochs_usl=200,
                                 num_epochs_le=150,
                                 save_embeddings=True)
'''
