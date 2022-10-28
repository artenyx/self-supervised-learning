from ColabExport.TrainModel import experiments, networks

import torch

print("========Running Network========")
print("Device: "+str(torch.device("cuda:0" if torch.cuda.is_available() else "cpu")))
experiments.ssl_experiment_setup(model_type=networks.USL_Conv6_CIFAR_Sym,
                                 exp_type=("AE-S", "D", "L"),
                                 num_epochs_usl=40,
                                 num_epochs_le=150)

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
