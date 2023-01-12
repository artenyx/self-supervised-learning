from src.TrainModel import load_data

import torch
import torch.nn as nn
import torchvision.transforms as T
import os


def get_exp_config(s=0.25, crop_size=24):

    loaders_dict = {
        "loaders_usl": None,
        "loaders_le": None,
        "loaders_embedded": None
    }

    transforms_dict = {
        "ToTens": T.ToTensor(),
        "Crop": T.Compose([T.RandomCrop(crop_size), T.Resize(32)]),
        "HorFlip": T.RandomHorizontalFlip(p=0.8),
        "ColJit": T.ColorJitter(brightness=0.8 * s, contrast=0.8 * s, saturation=0.8 * s, hue=0.2 * s),
        "GausBlur": T.GaussianBlur(kernel_size=3),
        "Solar": T.RandomSolarize(0.75, p=0.2)
    }

    exp_config = {
        # Model Parameters
        "model_type": None,
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
        "loaders": loaders_dict,
        "batch_size": 512,
        "criterion_class": nn.CrossEntropyLoss,
        "criterion_emb": "l2",
        "criterion_recon": "l2",
        "optimizer_type": torch.optim.Adam,
        "optimizer": None,
        "scheduler_type":  torch.optim.lr_scheduler.CosineAnnealingLR,
        "scheduler": None,
        "num_epochs_usl": 200,
        "num_epochs_le": 150,
        "lr_usl": 0.001,  # verified best lr for doc 5
        "lr_le": 0.01,  # verified best lr for doc 5
        "transforms_dict": transforms_dict,
        "transforms_active": ['ToTens', 'Crop', 'HorFlip', 'ColJit'],
        "transforms_list_full": list(transforms_dict.keys()),
        "run_test_rate_usl": 1,
        "print_loss_rate": 1,  # number of epochs in between printing loss/error and saving images (if USL)
        "save_path": None,
        "num_embed_pts_plot": 1000,
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
    config['loaders']['loaders_usl'] = load_data.get_cifar100_usl(config)
    config['loaders']['loaders_le'] = load_data.get_cifar10_classif(config)

    return config


def reset_config_paths_colab(config):
    config['data_save_path'] = '/content/gdrive/My Drive/UChicago Documents/Thesis/ColabExport_SaveFiles/SavedData/'
    config['model_save_path'] = '/content/gdrive/My Drive/UChicago Documents/Thesis/ColabExport_SaveFiles/SavedModels/'
    config['image_save_path'] = '/content/gdrive/My Drive/UChicago Documents/Thesis/ColabExport_SaveFiles/SavedImages/'
    return config


def save_data_model(config, exp_type, data, model):
    data.to_csv(config['save_path'] + exp_type + '_data.csv')
    torch.save({'model.state.dict': model.state_dict()},
               config['save_path'] + exp_type + '_model_.pt')


def make_dir(folder_name):
    if not os.path.exists(folder_name):
        os.mkdir(folder_name)
    else:
        print("Directory " + folder_name + " exists")

