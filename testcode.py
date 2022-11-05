import torch

from TrainModel import networks
from Experiments import experiments

print(torch.__version__)
print(torch.cuda.get_arch_list())
print(torch.cuda.current_device())

experiments.ssl_experiment_setup(model_type=networks.USL_Conv6_CIFAR1,
                                 exp_type=("AE-P", "D", "NL"),
                                 num_epochs_usl=5,
                                 num_epochs_le=1)

