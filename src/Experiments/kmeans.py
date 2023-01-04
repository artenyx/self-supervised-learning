import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

from src.Experiments import plots
from src.TrainModel import train


def kmeans_embedding(emb_loader, clusters=10):
    kmeans_model = KMeans(n_clusters=3, random_state=32932)
    kmeans_predict = kmeans_model.fit_predict(emb_df)


def kmeans_inner(parent_run_folder, clusters=10):
    files = list(plots.listdir_nohidden(parent_run_folder, True))
    if "USL_model_.pt" not in files:
        raise Exception("No trained model to create embedding dataset with.")
    usl_model = torch.load(parent_run_folder + "/USL_model_.pt")
    usl_model.eval()
    embeddings = []
    captions = []

    train.get_embedding_loader(usl_model)




