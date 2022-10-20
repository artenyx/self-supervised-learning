import pandas as pd
import torch
from torchsummary import summary

from ColabExport.TrainModel import train, networks, load_data, experiments
from ColabExport import exp_config


def run_representation_learning(config, model):
    if config['layerwise_training']:
        assert config['usl_type'] == 'ae_single' or config['usl_type'] == 'ae_parallel'
        usl_data, usl_model = train.usl_train_network_layerwise(model, config)
    else:
        usl_data, usl_model = train.usl_train_network(model, config)
    return usl_data, usl_model


def run_linear_evaluation(config):
    if config['loaders_embedded'] is not None:
        config['loaders_le'] = config['loaders_embedded']
    train_loader, test_loader = config['loaders_le']
    le_model = networks.Linear_Evaluation_Classifier(train_loader.dataset[0][0].shape[0], config['num_classes']).to(config['device'])
    le_data, le_model = train.classifier_train_network(le_model, config)
    return le_data, le_model


def run_ssl_experiment(config, exp_string, rep_learning_model=None, save=True):
    config_df = pd.DataFrame.from_dict(config)
    config_df.to_csv(config['data_save_path'] + 'config_' + exp_string + '.csv')
    if rep_learning_model is None:
        if config['layerwise_training']:
            rep_learning_model = networks.USL_Conv6_CIFAR_Sym(config).to(config['device'])
            print(summary(rep_learning_model, (3, 32, 32), batch_size=256))
        else:
            rep_learning_model = networks.USL_Conv6_CIFAR(config).to(config['device'])

    usl_data, usl_model = run_representation_learning(config, rep_learning_model)

    emb_train_loader = train.get_embedding_loader(usl_model, config, config['loaders_le'][0])
    emb_test_loader = train.get_embedding_loader(usl_model, config, config['loaders_le'][1])
    config['loaders_embedded'] = emb_train_loader, emb_test_loader

    le_data, le_model = run_linear_evaluation(config)

    if save:
        exp_config.save_data_model(config, 'USL', exp_string, usl_data, usl_model)
        exp_config.save_data_model(config, 'LE', exp_string, le_data, le_model)

    return usl_data, usl_model, le_data, le_model


def test_usl_lr(config, lr_list):
    for i in range(len(lr_list)):
        config['loaders_usl'] = load_data.get_CIFAR100(config)
        print("\nTesting USL learning rate of", lr_list[i])
        config['lr_usl'] = lr_list[i]
        config = exp_config.reset_config_paths_colab(config)
        if config['layerwise_training']:
            rep_learning_model = networks.USL_Conv6_CIFAR_Sym(config).to(config['device'])
            print(summary(rep_learning_model, (3, 32, 32), batch_size=256))
        else:
            rep_learning_model = networks.USL_Conv6_CIFAR(config).to(config['device'])
        usl_data, usl_model = run_representation_learning(config, rep_learning_model)


def test_le_lr(config, lr_list):
    config['loaders_usl'] = load_data.get_CIFAR100(config)
    config['loaders_le'] = load_data.get_CIFAR10(config)
    config = exp_config.reset_config_paths_colab(config)
    if config['layerwise_training']:
        rep_learning_model = networks.USL_Conv6_CIFAR_Sym(config).to(config['device'])
        print(summary(rep_learning_model, (3, 32, 32), batch_size=256))
    else:
        rep_learning_model = networks.USL_Conv6_CIFAR(config).to(config['device'])
    usl_data, usl_model = run_representation_learning(config, rep_learning_model)
    emb_train_loader = train.get_embedding_loader(usl_model, config, config['loaders_le'][0])
    emb_test_loader = train.get_embedding_loader(usl_model, config, config['loaders_le'][1])
    for i in range(len(lr_list)):
        print("\nTesting LE learning rate of", lr_list[i])
        config['lr_le'] = lr_list[i]
        le_data, le_model = run_linear_evaluation(config, (emb_train_loader, emb_test_loader))

