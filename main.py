from ColabExport.TrainModel import experiments, networks
from ColabExport import exp_config, plots

import pandas as pd

'''
print("========Running network 1========")
model_type = networks.USL_Conv6_CIFAR1
experiments.print_model_architecture(model_type)
experiments.ssl_experiment1(model_type)
print("\n========Running network 2========")
model_type = networks.USL_Conv6_CIFAR2
experiments.print_model_architecture(model_type)
experiments.ssl_experiment1(model_type)

print("========Running SimCLR network 1========")
model_type = networks.USL_Conv6_CIFAR1
experiments.print_model_architecture(model_type)
experiments.ssl_experiment2(model_type)

print("========Running AE Single Denoising network 1========")
model_type = networks.USL_Conv6_CIFAR1
experiments.print_model_architecture(model_type)
experiments.ssl_experiment4(model_type)

print("========Running AE Single Denoising network 1========")
model_type = networks.USL_Conv6_CIFAR1
experiments.print_model_architecture(model_type)
experiments.ssl_experiment5(model_type)

'''
#plots.produce_embedding_plots(samples_to_use=1000, load_obj="_embloaders_AE-S-D-USL_Conv6_CIFAR1.pt")
config = exp_config.get_exp_config()
exp_config.reset_config_paths_colab(config)

le_data_ae_s_d = pd.read_csv(config['data_save_path']+"LE_data_AE-S-D-USL_Conv6_CIFAR1.csv")
usl_data_ae_s_d = pd.read_csv(config['data_save_path']+"USL_data_AE-S-D-USL_Conv6_CIFAR1.csv")
plots.plot_usl(usl_data_ae_s_d)
plots.plot_lineval(le_data_ae_s_d)

le_data_ae_s_nd = pd.read_csv(config['data_save_path']+"LE_data_AE-S-USL_Conv6_CIFAR1.csv")
usl_data_ae_s_nd = pd.read_csv(config['data_save_path']+"USL_data_AE-S-USL_Conv6_CIFAR1.csv")
plots.plot_usl(usl_data_ae_s_nd)
plots.plot_lineval(le_data_ae_s_nd)

le_data_simclr = pd.read_csv(config['data_save_path']+"LE_data_SimCLR-<class 'ColabExport.TrainModel.networks.USL_Conv6_CIFAR1'>.csv")
usl_data_simclr = pd.read_csv(config['data_save_path']+"USL_data_SimCLR-<class 'ColabExport.TrainModel.networks.USL_Conv6_CIFAR1'>.csv")
plots.plot_usl(usl_data_simclr)
plots.plot_lineval(le_data_simclr)
