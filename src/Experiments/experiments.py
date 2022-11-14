import torch
import pandas as pd
from datetime import datetime

from src.TrainModel import train, networks, load_data
from src.Experiments import exp_config


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
    config_df.to_csv(config["save_path"] + "exp_config")

    if rep_learning_model is None:
        if config['layerwise_training']:
            rep_learning_model = networks.USL_Conv6_CIFAR_Sym(config).to(config['device'])
            # print(summary(rep_learning_model, (3, 32, 32), batch_size=256))
        else:
            rep_learning_model = networks.USL_Conv6_CIFAR1(config).to(config['device'])

    usl_data, usl_model = run_representation_learning(config, rep_learning_model)

    emb_train_loader = train.get_embedding_loader(usl_model, config, config['loaders']['loaders_le'][0])
    emb_test_loader = train.get_embedding_loader(usl_model, config, config['loaders']['loaders_le'][1])
    config['loaders']['loaders_embedded'] = emb_train_loader, emb_test_loader
    #plots.produce_embedding_plots(samples_to_use=1000, config=config, get_loader_from_config=True, pca_or_tsne="tsne")

    le_data, le_model = run_linear_evaluation(config)
    #plots.produce_usl_lineval_plots(config, usl_df=usl_data, lineval_df=le_data)

    if save:
        exp_config.save_data_model(config, 'USL', usl_data, usl_model)
        exp_config.save_data_model(config, 'LE', le_data, le_model)

    return usl_data, usl_model, le_data, le_model


def print_model_architecture(model_type, input_size=(3, 32, 32)):
    config = exp_config.get_exp_config()
    config['model_type'] = model_type
    model = config['model_type'](config).to(config['device'])
    # summary(model, (3, 32, 32))


def ssl_experiment_setup(exp_type=("AE-S", "D", "NL"),
                         alpha=0.001,
                         config=None,
                         add_exp_str='',
                         num_epochs_usl=200,
                         num_epochs_le=150,
                         run_test_rate_usl=1,
                         print_loss_rate=50,
                         save_images=True,
                         save_embeddings=True,
                         return_data=False,
                         strength=0.25):
    if config is None:
        config = exp_config.get_exp_config(s=strength)

    if exp_type[0] == "AE-S":
        config['usl_type'] = 'ae_single'
    elif exp_type[0] == "AE-P":
        config['usl_type'] = 'ae_parallel'
        config['alpha'] = alpha
    elif exp_type[0] == "SimCLR":
        config['usl_type'] = 'SimCLR'
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
        model_type = networks.USL_Conv6_CIFAR_Sym
    elif exp_type[2] == "NL":
        config['layerwise_training'] = False
        model_type = networks.USL_Conv6_CIFAR1
    else:
        raise Exception(
            "Third element of exp_type must be L or NL, representing layerwise training or non-layerwise training.")

    config['num_epochs_usl'] = num_epochs_usl
    config['num_epochs_le'] = num_epochs_le
    config['loaders']['loaders_usl'] = load_data.get_cifar100_usl(config)
    config['loaders']['loaders_le'] = load_data.get_cifar10_classif(config)
    config['save_images'] = save_images
    config['run_test_rate_usl'] = run_test_rate_usl
    config['print_loss_rate'] = print_loss_rate
    config['save_embeddings'] = save_embeddings
    config['exp_type'] = str(exp_type)

    print(config)
    config['model_type'] = model_type
    model = config['model_type'](config).to(config['device'])
    date_time = datetime.now().strftime("%m-%d-%Y_%H-%M-%S")
    exp_folder_name = "-".join(exp_type) + "_" + str(date_time) + add_exp_str
    usl_data, usl_model, le_data, le_model = run_ssl_experiment(config, exp_folder_name, rep_learning_model=model)
    if return_data:
        return usl_data, usl_model, le_data, le_model


def test_alpha_parallel(alpha_list=None):
    if alpha_list is None:
        alpha_list = [0.0001, 0.001, 0.01]
    for alpha0 in alpha_list:
        ssl_experiment_setup(exp_type=("AE-P", "D", "NL"),
                             alpha=alpha0,
                             add_exp_str="alpha_"+str(alpha0))
    print("COMPLETE")
    return


def test_strength_single(strength_list=None):
    if strength_list is None:
        strength_list = [0, 0.25, 0.5, 0.75, 1]
    for strength0 in strength_list:
        ssl_experiment_setup(exp_type=("AE-S", "D", "NL"),
                             strength=strength0,
                             add_exp_str='strength_'+str(strength0))
    print("COMPLETE")
    return


def classif_from_load_model(load_path, usl_model=None, num_epochs=150):
    if usl_model is None:
        usl_model = torch.load(load_path)
    config = exp_config.get_exp_config()
    config['num_epochs_le'] = num_epochs
    config['loaders']['loaders_le'] = load_data.get_cifar10_classif(config)
    emb_train_loader = train.get_embedding_loader(usl_model, config, config['loaders']['loaders_le'][0])
    emb_test_loader = train.get_embedding_loader(usl_model, config, config['loaders']['loaders_le'][1])
    config['loaders']['loaders_embedded'] = emb_train_loader, emb_test_loader

    le_data, le_model = run_linear_evaluation(config)