import torch
import torch.nn as nn
import torchvision.transforms as T
import ColabExport.TrainModel.load_data as load_data


def get_exp_config(s=0.25):

    """ Retrieve configuration for the model. """
    model_dict = {
        "usl_model1": USL_Conv6_CIFAR1,
        "usl_model2": USL_Conv6_CIFAR2,
        "usl_model3": USL_Conv6_CIFAR3
    }

    exp_config = {
        # Model Parameters
        "width": 32,
        "height": 32,
        "channels": 3,
        "latent_dim": 64,  # size of AE bottleneck #plan to remove
        "num_classes": 10,
        "representation_dim": 8192,

        # Experiment Parameters'
        "device": torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
        "denoising": None,
        "alpha": None,
        "usl_type": None,
        "layerwise_training": False,
        "loaders_usl": None,
        "loaders_le": None,
        "loaders_embedded": None,
        "batch_size": 512,
        "criterion_class": nn.CrossEntropyLoss,
        "criterion_emb_recon": nn.MSELoss,
        "optimizer_type": torch.optim.Adam,
        "optimizer": None,
        "num_epochs_usl": 200,
        "num_epochs_le": 150,
        "lr_usl": 0.001,  # verified best lr for doc 5
        "lr_le": 0.01,  # verified best lr for doc 5
        "transform": T.Compose([T.RandomCrop(24),
                               T.Resize(32),
                               T.RandomHorizontalFlip(p=0.8),
                               T.ColorJitter(brightness=0.8*s, contrast=0.8*s, saturation=0.8*s, hue=0.2*s)]),
        "transform_reduced": T.Compose([T.RandomCrop(30),
                                       T.Resize(32),
                                       T.RandomHorizontalFlip(p=0.4),
                                       T.ColorJitter()]),  # for classification reasons only
        "transform_dataloader": T.Compose([T.ToTensor()]),
        "printloss_rate": 1,  # number of epochs in between printing loss/error and saving images (if USL)

        "data_save_path": 'SavedData/',
        "model_save_path": 'SavedModels/',
        "image_save_path": 'SavedImages/'
        }
    return exp_config


def get_ae_parallel_config(denoising):
    config = get_exp_config()

    config['device'] = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    config['usl_type'] = 'ae_parallel'
    config['alpha'] = 0
    config['denoising'] = denoising

    config['num_epochs_usl'] = 10
    config['num_epochs_le'] = 10
    config['loaders_usl'] = load_data.get_CIFAR100(config)
    config['loaders_le'] = load_data.get_CIFAR10(config)

    return config


def reset_config_paths_colab(config):
    config['data_save_path'] = '/content/gdrive/My Drive/UChicago Documents/Thesis/ColabExport_SaveFiles/SavedData/'
    config['model_save_path'] = '/content/gdrive/My Drive/UChicago Documents/Thesis/ColabExport_SaveFiles/SavedModels/'
    config['image_save_path'] = '/content/gdrive/My Drive/UChicago Documents/Thesis/ColabExport_SaveFiles/SavedImages/'
    return config


def save_data_model(config, exp_type, exp_string, data, model):
    data.to_csv(config['data_save_path'] + exp_type + '_data_' + exp_string + '.csv')
    torch.save({'model.state.dict': model.state_dict()}, config['model_save_path'] + exp_type + '_model_' + exp_string + '.pt')





