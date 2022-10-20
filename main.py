import torch

from ColabExport.TrainModel import load_data, experiments
from ColabExport import exp_config

experiment_config = exp_config.get_exp_config()
experiment_config['usl_type'] = 'ae_single'
experiment_config['alpha'] = 0
experiment_config['denoising'] = False
experiment_config['layerwise_training'] = True

experiment_config['num_epochs_usl'] = 0
experiment_config['num_epochs_le'] = 0
experiment_config['loaders_usl'] = load_data.get_CIFAR100(experiment_config)
experiment_config['loaders_le'] = load_data.get_CIFAR10(experiment_config)
experiment_config['print_loss_rate'] = 1
experiment_config['save_images'] = False

experiment_config = exp_config.reset_config_paths_colab(experiment_config)
print(experiment_config)

experiments.run_ssl_experiment(experiment_config, 'ae_single_run1')
