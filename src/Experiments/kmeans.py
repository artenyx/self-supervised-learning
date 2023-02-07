import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

from src.Experiments import plots, experiments, exp_config
from src.TrainModel import train, networks, load_data


def kmeans_from_load_model(args=None, load_path=None, usl_model=None, simclr=False, n_clusters=10):
    config = exp_config.get_exp_config()
    if simclr:
        config['usl_type'] = "simclr"
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


def knn_from_load_model(args=None, load_path=None, usl_model=None, simclr=False, n_neighbors=10):
    config = exp_config.get_exp_config()
    if simclr:
        config['usl_type'] = "simclr"
    if usl_model is None:
        usl_model = networks.USL_Conv6_CIFAR1(config=config).to(config['device'])
        load_path = args.usl_load_path if args is not None else load_path
        assert load_path is not None, "Trained USL Model load path not provided. Please provide either arg parser or direct path."
        usl_model.load_state_dict(torch.load(load_path)['model.state.dict'])
    config['loaders']['loaders_le'] = load_data.get_cifar10_classif(config)

    embs_train, embs_train_targs = train.get_embedding_loader(usl_model, config, config['loaders']['loaders_le'][0], return_as_list=True)
    embs_train = embs_train.numpy()
    embs_train_targs = embs_train_targs.numpy()

    knn_train = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn_train.fit(embs_train, embs_train_targs.ravel())
    embs_train_pred = knn_train.predict(embs_train)
    score_train = metrics.accuracy_score(embs_train_targs, embs_train_pred)

    embs_test, embs_test_targs = train.get_embedding_loader(usl_model, config, config['loaders']['loaders_le'][1], return_as_list=True)
    embs_test = embs_test.numpy()
    embs_test_targs = embs_test_targs.numpy()

    knn_test = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn_test.fit(embs_test, embs_test_targs.ravel())
    embs_test_pred = knn_test.predict(embs_test)
    score_test = metrics.accuracy_score(embs_test_targs, embs_test_pred)
    return score_train, score_test


def kmeans_knn_run_dir(run_dir_path, clusters=10, knn=False):
    print(run_dir_path)
    files = list(plots.listdir_nohidden(run_dir_path, False))
    if "USL_model_.pt" not in files:
        print(run_dir_path + " does not contain a trained model to create embedding dataset with. Skipping this run.")
        return None, None
    simclr = True if "simclr_" in run_dir_path else False
    print(str(simclr) + " " + run_dir_path)
    if knn:
        data = knn_from_load_model(load_path=run_dir_path + "/USL_model_.pt", simclr=simclr)
    else:
        data = kmeans_from_load_model(load_path=run_dir_path + "/USL_model_.pt", simclr=simclr, n_clusters=clusters)
    return data


def kmeans_knn_exp_dir(exp_dir_path, clusters=10, save=True, overwrite=False, knn=False):
    plots_list = list(plots.listdir_nohidden(exp_dir_path + "/000_plots/usl", dir_only=False))
    if "kmeans.csv" in plots_list and not knn:
        print("kmeans.csv exists. Skip exp.")
        return
    if "knn.csv" in plots_list and knn:
        print("knn.csv exists. Skip exp.")
        return

    files = list(plots.listdir_nohidden(exp_dir_path, dir_only=True))
    data_exp = []
    for f in files:
        score_train, score_test = kmeans_knn_run_dir(exp_dir_path + "/" + f, clusters=clusters, knn=knn)
        data_exp.append((f, score_train, score_test))
    if save:
        kmeans_data_exp_df = pd.DataFrame(data_exp)
        kmeans_data_exp_df.to_csv(exp_dir_path + "/000_plots/usl/kmeans.csv" if not knn else "/000_plots/usl/knn.csv")
    return


def kmeans_knn_all_exps(args, all_exp_dir_path="/home/geraldkwhite/SSLProject/200E_Scheduler", clusters=10, knn=True):
    files = list(plots.listdir_nohidden(all_exp_dir_path, True))
    for f in files:
        print(all_exp_dir_path + "/" + f)
        kmeans_knn_exp_dir(all_exp_dir_path + "/" + f, clusters=clusters, save=True, knn=knn)
        try:
            kmeans_knn_exp_dir(all_exp_dir_path + "/" + f, clusters=clusters, save=True, knn=knn)
        except:
            print("Issue with file " + f)
