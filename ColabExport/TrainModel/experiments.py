import pandas as pd
import torch
#from torchsummary import summary
from datetime import datetime

from ColabExport.TrainModel import train, networks, load_data, experiments
from ColabExport import exp_config, plots


def run_representation_learning(config, model):
    if config['layerwise_training']:
        assert config['usl_type'] == 'ae_single' or config['usl_type'] == 'ae_parallel'
        usl_data, usl_model = train.usl_train_network_layerwise(model, config)
    else:
        usl_data, usl_model = train.usl_train_network(model, config)
    return usl_data, usl_model


def run_linear_evaluation(config):
    if config['loaders']['loaders_embedded'] is not None:
        config['loaders']['loaders_le'] = config['loaders']['loaders_embedded']
    train_loader, test_loader = config['loaders']['loaders_le']
    le_model = networks.Linear_Evaluation_Classifier(train_loader.dataset[0][0].shape[0], config['num_classes']).to(
        config['device'])
    le_data, le_model = train.classifier_train_network(le_model, config)
    return le_data, le_model


def run_ssl_experiment(config, exp_string, rep_learning_model=None, save=True):
    config_df = pd.DataFrame.from_dict(config)
    exp_config.make_dir("ExperimentFiles")
    exp_path = "ExperimentFiles/" + exp_string + "/"
    exp_config.make_dir(exp_path)
    config["save_path"] = exp_path
    config_df.to_csv(config["save_path"]+"exp_config")

    if rep_learning_model is None:
        if config['layerwise_training']:
            rep_learning_model = networks.USL_Conv6_CIFAR_Sym(config).to(config['device'])
            #print(summary(rep_learning_model, (3, 32, 32), batch_size=256))
        else:
            rep_learning_model = networks.USL_Conv6_CIFAR1(config).to(config['device'])

    usl_data, usl_model = run_representation_learning(config, rep_learning_model)

    emb_train_loader = train.get_embedding_loader(usl_model, config, config['loaders']['loaders_le'][0])
    emb_test_loader = train.get_embedding_loader(usl_model, config, config['loaders']['loaders_le'][1])
    config['loaders']['loaders_embedded'] = emb_train_loader, emb_test_loader
    plots.produce_embedding_plots(samples_to_use=5000, config=config, get_loader_from_config=True, pca_or_tsne="tsne")

    le_data, le_model = run_linear_evaluation(config)
    plots.produce_usl_lineval_plots(config, usl_df=usl_data, lineval_df=le_data)

    if save:
        exp_config.save_data_model(config, 'USL', usl_data, usl_model)
        exp_config.save_data_model(config, 'LE', le_data, le_model)

    return usl_data, usl_model, le_data, le_model


def print_model_architecture(model_type, input_size=(3, 32, 32)):
    config = exp_config.get_exp_config()
    config['model_type'] = model_type
    model = config['model_type'](config).to(config['device'])
    #summary(model, (3, 32, 32))


def ssl_experiment_setup(model_type=networks.USL_Conv6_CIFAR1,
                         exp_type=("AE-S", "D", "NL"),
                         alpha=0.001,
                         config=None,
                         add_exp_str='',
                         num_epochs_usl=200,
                         num_epochs_le=150,
                         print_loss_rate=20,
                         save_images=True,
                         save_embeddings=False,
                         return_data=False):
    if config is None:
        config = exp_config.get_exp_config()

    if exp_type[0] == "AE-S":
        config['usl_type'] = 'ae_single'
    elif exp_type[0] == "AE-P":
        config['usl_type'] = 'ae_parallel'
        config['alpha'] = alpha
    elif exp_type[0] == "SimCLR":
        config['usl_type'] = 'ae_parallel'
    else:
        raise Exception("First element of exp_type must be AE-S, AE-P, or SimCLR.")

    if exp_type[1] == "D":
        config['denoising'] = True
    elif exp_type[1] == "ND":
        config['denoising'] = False
    else:
        raise Exception(
            "Second element of exp_type must be D or ND, representing denoising or non-denoising autoencoder.")

    if exp_type[2] == "L":
        config['layerwise_training'] = True
    elif exp_type[2] == "NL":
        config['layerwise_training'] = False
    else:
        raise Exception(
            "Third element of exp_type must be L or NL, representing layerwise training or non-layerwise training.")

    config['num_epochs_usl'] = num_epochs_usl
    config['num_epochs_le'] = num_epochs_le
    config['loaders']['loaders_usl'] = load_data.get_CIFAR100(config)
    config['loaders']['loaders_le'] = load_data.get_CIFAR10(config)
    config['save_images'] = save_images
    config['print_loss_rate'] = print_loss_rate
    config['save_embeddings'] = save_embeddings

    print(config)
    config['model_type'] = model_type
    model = config['model_type'](config).to(config['device'])
    date_time = datetime.now().strftime("%m.%d.%Y-%H:%M:%S")

    usl_data, usl_model, le_data, le_model = experiments.run_ssl_experiment(config, "-".join(exp_type) + "_" + str(date_time) + add_exp_str, rep_learning_model=model)
    if return_data:
        return usl_data, usl_model, le_data, le_model




'''
### Construction

def test_usl_lr(config, lr_list):
    for i in range(len(lr_list)):
        config['loaders']['loaders_usl'] = load_data.get_CIFAR100(config)
        print("\nTesting USL learning rate of", lr_list[i])
        config['lr_usl'] = lr_list[i]
        config = exp_config.reset_config_paths_colab(config)
        rep_learning_model = config['model'](config).to(config['device'])
        usl_data, usl_model = run_representation_learning(config, rep_learning_model)
        
        
def test_le_lr(config, lr_list):
    config['loaders']['loaders_usl'] = load_data.get_CIFAR100(config)
    config['loaders']['loaders_le'] = load_data.get_CIFAR10(config)
    config = exp_config.reset_config_paths_colab(config)
    if config['layerwise_training']:
        rep_learning_model = networks.USL_Conv6_CIFAR_Sym(config).to(config['device'])
        print(summary(rep_learning_model, (3, 32, 32), batch_size=256))
    else:
        rep_learning_model = networks.USL_Conv6_CIFAR1(config).to(config['device'])
    usl_data, usl_model = run_representation_learning(config, rep_learning_model)
    emb_train_loader = train.get_embedding_loader(usl_model, config, config['loaders']['loaders_le'][0])
    emb_test_loader = train.get_embedding_loader(usl_model, config, config['loaders']['loaders_le'][1])
    config['loaders']['loaders_embedded'] = (emb_train_loader, emb_test_loader)
    for i in range(len(lr_list)):
        print("\nTesting LE learning rate of", lr_list[i])
        config['lr_le'] = lr_list[i]
        le_data, le_model = run_linear_evaluation(config)
'''

