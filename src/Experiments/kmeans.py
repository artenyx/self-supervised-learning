import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

from src.Experiments import plots, experiments, exp_config
from src.TrainModel import train, networks, load_data


def kmeans_from_load_model(args=None, load_path=None, usl_model=None, n_clusters=10):
    config = exp_config.get_exp_config()
    if usl_model is None:
        usl_model = networks.USL_Conv6_CIFAR1(config=config).to(config['device'])
        load_path = args.usl_load_path if args is not None else load_path
        assert load_path is not None, "Trained USL Model load path not provided. Please provide either arg parser or direct path."
        usl_model.load_state_dict(torch.load(load_path)['model.state.dict'])
    config['loaders']['loaders_le'] = load_data.get_cifar10_classif(config)

    embs_train, __ = train.get_embedding_loader(usl_model, config, config['loaders']['loaders_le'][0], return_as_list=True)
    embs_train = pd.DataFrame(embs_train.numpy())
    kmeans_model_train = KMeans(n_clusters=n_clusters).fit(embs_train)

    embs_test, __ = train.get_embedding_loader(usl_model, config, config['loaders']['loaders_le'][1], return_as_list=True)
    embs_test = pd.DataFrame(embs_test.numpy())
    kmeans_model_test = KMeans(n_clusters=n_clusters).fit(embs_test)
    return kmeans_model_train.inertia_/len(embs_train), kmeans_model_test.inertia_/len(embs_test)


def kmeans_run_dir(run_dir_path, clusters=10):
    files = list(plots.listdir_nohidden(run_dir_path, False))
    if "USL_model_.pt" not in files:
        raise Exception("No trained model to create embedding dataset with.")
    kmeans_data = kmeans_from_load_model(load_path=run_dir_path + "/USL_model_.pt", n_clusters=clusters)
    return kmeans_data


def kmeans_exp_dir(exp_dir_path, clusters=10, save=True):
    files = list(plots.listdir_nohidden(exp_dir_path, True))
    kmeans_data_exp = []
    for f in files:
        inertia_run_train, inertia_run_test = kmeans_run_dir(exp_dir_path + "/" + f, clusters=clusters)
        kmeans_data_exp.append((f, inertia_run_train, inertia_run_test))

    if save:
        kmeans_data_exp_df = pd.DataFrame(kmeans_data_exp)
        kmeans_data_exp_df.to_csv(exp_dir_path + "/000_plots/usl/kmeans.csv")
    return kmeans_data_exp


