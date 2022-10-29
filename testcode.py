import torch

from ColabExport.TrainModel import experiments, networks

print(torch.__version__)
print(torch.cuda.get_arch_list())
print(torch.cuda.current_device())

experiments.ssl_experiment_setup(model_type=networks.USL_Conv6_CIFAR1,
                                 exp_type=("AE-S", "D", "L"),
                                 num_epochs_usl=0,
                                 num_epochs_le=1,
                                 save_embeddings=True)

