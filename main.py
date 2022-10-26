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
config =
plots.produce_embedding_plots(samples_to_use=1000, load_obj="_embloaders_AE-S-D-USL_Conv6_CIFAR1.pt")
