from ColabExport.TrainModel import experiments, networks

import torch

print("========Running Network========")
print(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))

experiments.test_alpha_layerwise(alpha_list=[0.001])

'''
experiments.ssl_experiment_setup(model_type=networks.USL_Conv6_CIFAR_Sym,
                                 exp_type=("AE-P", "D", "NL"),
                                 num_epochs_usl=200,
                                 num_epochs_le=150,
                                 save_embeddings=True)
'''
