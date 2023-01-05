import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

from src.Experiments import plots, experiments
from src.TrainModel import train


def kmeans_embedding(emb_loader, clusters=10):
    kmeans_model = KMeans(n_clusters=3, random_state=32932)
    kmeans_predict = kmeans_model.fit_predict(emb_df)


def kmeans_run_dir(run_dir_path, clusters=10):
    files = list(plots.listdir_nohidden(run_dir_path, True))
    if "USL_model_.pt" not in files:
        raise Exception("No trained model to create embedding dataset with.")
    kmeans_data = experiments.kmeans_from_load_model(load_path=run_dir_path + "/USL_model_.pt", n_clusters=clusters)
    return kmeans_data


def kmeans_exp_dir(exp_dir_path, clusters=10, save=True):
    files = list(plots.listdir_nohidden(exp_dir_path, True))
    kmeans_data_exp = []
    for f in files:
        inertia_run_train, inertia_run_test = kmeans_run_dir(exp_dir_path + "/" + f, clusters=clusters)
        kmeans_data_exp.append((f, inertia_run_train, inertia_run_test))

    if save:
        kmeans_data_exp_df = pd.DataFrame(kmeans_data_exp)
        kmeans_data_exp_df.to_csv(exp_dir_path + "/000_plots/kmeans.csv")
    return kmeans_data_exp










