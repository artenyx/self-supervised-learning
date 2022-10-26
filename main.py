from ColabExport.TrainModel import experiments, networks
from ColabExport import exp_config, plots

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
config1 = exp_config.get_exp_config()
config1 = exp_config.reset_config_paths_colab(config1)
data_arrays = plots.emb_loader_to_array(config1['data_save_path']+"_embloaders_AE-S-D-USL_Conv6_CIFAR1.pt")


plots.plot_pca(config1, data_arrays[0][:1000])
plots.plot_tsne(config1, data_arrays[0][:1000], data_arrays[1])


