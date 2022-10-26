from ColabExport.TrainModel import experiments, networks
from ColabExport import exp_config, plots

import pandas as pd
import torch


print("========Running network 1========")
print(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
model_type = networks.USL_Conv6_CIFAR1
experiments.print_model_architecture(model_type)
experiments.ssl_experiment1(model_type)

'''
plots.produce_embedding_plots(samples_to_use=1000, load_obj="_embloaders_AE-S-D-USL_Conv6_CIFAR1.pt")
'''

'''
config = exp_config.get_exp_config()
exp_config.reset_config_paths_colab(config)

le_data_ae_s_d = pd.read_csv(config['data_save_path']+"LE_data_AE-S-D-USL_Conv6_CIFAR1.csv")
usl_data_ae_s_d = pd.read_csv(config['data_save_path']+"USL_data_AE-S-D-USL_Conv6_CIFAR1.csv")
plots.plot_usl(config, usl_data_ae_s_d, print_string='ae_s_den_')
plots.plot_lineval(config, le_data_ae_s_d, print_string='ae_s_den_')

le_data_ae_s_nd = pd.read_csv(config['data_save_path']+"LE_data_AE-S-USL_Conv6_CIFAR1.csv")
usl_data_ae_s_nd = pd.read_csv(config['data_save_path']+"USL_data_AE-S-USL_Conv6_CIFAR1.csv")
plots.plot_usl(config, usl_data_ae_s_nd, print_string='ae_s_nonden_')
plots.plot_lineval(config, le_data_ae_s_nd, print_string='ae_s_nonden_')

le_data_simclr = pd.read_csv(config['data_save_path']+"LE_data_SimCLR-<class 'ColabExport.TrainModel.networks.USL_Conv6_CIFAR1'>.csv")
usl_data_simclr = pd.read_csv(config['data_save_path']+"USL_data_SimCLR-<class 'ColabExport.TrainModel.networks.USL_Conv6_CIFAR1'>.csv")
plots.plot_usl(config, usl_data_simclr, print_string='simclr_')
plots.plot_lineval(config, le_data_simclr, print_string='simclr_')
'''