from ColabExport.TrainModel import experiments, networks

model_type = networks.USL_Conv6_CIFAR2
experiments.print_model_architecture(model_type)
experiments.ssl_experiment1(model_type)
