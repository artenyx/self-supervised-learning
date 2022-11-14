import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import os

from src.Experiments import exp_config


def plot_pca(config, dataset_array, print_string=''):
    pca = PCA(n_components=100)

    dataset_array_pca = pca.fit_transform(dataset_array)
    exp_var_pca = pca.explained_variance_ratio_
    cum_sum_eigenvalues = np.cumsum(exp_var_pca)

    plt.bar(range(0, len(exp_var_pca)), exp_var_pca, alpha=0.5, align='center', label='Individual explained variance')
    plt.step(range(0, len(cum_sum_eigenvalues)), cum_sum_eigenvalues, where='mid',
             label='Cumulative explained variance')
    plt.ylabel('Explained variance ratio')
    plt.xlabel('Principal component index')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig(config['save_path'] + print_string + 'pca_fig.png')
    plt.show()
    return


def plot_tsne(config, dataset_array, target_array, print_string=''):
    tsne = TSNE(n_components=2, verbose=1, random_state=123)
    z = tsne.fit_transform(dataset_array)

    df = pd.DataFrame()
    df["y"] = target_array
    df["comp-1"] = z[:, 0]
    df["comp-2"] = z[:, 1]

    plot = sns.scatterplot(x="comp-1", y="comp-2", hue=df.y.tolist(), palette=sns.color_palette("hls", 10),
                           data=df).set(title="Embedding t-SNE Projection")
    plt.savefig(config['save_path'] + print_string + 'tsne_fig.png')
    plt.show()
    return


def emb_loader_to_array(emb_dataset_train, emb_dataset_test):
    emb_dataset_array_train = np.array([tup[0].cpu().detach().numpy() for tup in emb_dataset_train])
    emb_dataset_array_train = emb_dataset_array_train.reshape((-1, np.prod(emb_dataset_array_train.shape[1:])))

    emb_dataset_array_test = np.array([tup[0].cpu().detach().numpy() for tup in emb_dataset_test])
    emb_dataset_array_test = emb_dataset_array_test.reshape((-1, np.prod(emb_dataset_array_test.shape[1:])))

    emb_target_array_train = np.array([tup[1].cpu().detach().numpy() for tup in emb_dataset_train])
    emb_target_array_test = np.array([tup[1].cpu().detach().numpy() for tup in emb_dataset_test])

    return emb_dataset_array_train, emb_target_array_train, emb_dataset_array_test, emb_target_array_test


def produce_embedding_plots(samples_to_use=1000, 
                            config=None, 
                            load_obj=None, 
                            get_loader_from_config=False,
                            pca_or_tsne="both"):
    if config is None and get_loader_from_config:
        raise Exception("Must supply config since get_loader_from_config is True.")
    if load_obj is not None and get_loader_from_config:
        raise Warning("Load path will not be used since get_loader_from_config is True.")
    if config is None and not get_loader_from_config:
        config = exp_config.get_exp_config()
        config = exp_config.reset_config_paths_colab(config)
        loaders = torch.load(config['data_save_path'] + load_obj)
        emb_dataset_train = loaders["embedding_train_loader"].dataset
        emb_dataset_test = loaders["embedding_test_loader"].dataset
    elif config is not None and not get_loader_from_config:
        loaders = torch.load(config['data_save_path'] + load_obj)
        emb_dataset_train = loaders["embedding_train_loader"].dataset
        emb_dataset_test = loaders["embedding_test_loader"].dataset
    elif config is not None and get_loader_from_config:
        loaders = config["loaders"]["loaders_embedded"]
        emb_dataset_train = loaders[0].dataset
        emb_dataset_test = loaders[1].dataset
    else:
        raise Exception("Check function.")

    data_arrays = emb_loader_to_array(emb_dataset_train, emb_dataset_test)
    n = samples_to_use

    if config["save_embeddings"]:
        emb_train_array_save = pd.concat([pd.DataFrame(data_arrays[1]), pd.DataFrame(data_arrays[0])], axis=1)[:n]
        emb_test_array_save = pd.concat([pd.DataFrame(data_arrays[3]), pd.DataFrame(data_arrays[2])], axis=1)[:n]
        emb_train_array_save.to_csv(config["save_path"] + "embedding_array_train.csv")
        emb_test_array_save.to_csv(config["save_path"] + "embedding_array_test.csv")
    if pca_or_tsne == "pca" or pca_or_tsne == "both":
        plot_pca(config, data_arrays[0][:n], print_string='train_')
        plot_pca(config, data_arrays[2][:n], print_string='test_')
    if pca_or_tsne == "tsne" or pca_or_tsne == "both":
        plot_tsne(config, data_arrays[0][:n], data_arrays[1][:n], print_string='train_')
        plot_tsne(config, data_arrays[2][:n], data_arrays[3][:n], print_string='test_')
    return


def plot_lineval(config, le_data, to_epoch=None, print_string=""):
    if to_epoch is None:
        n = len(le_data["Epoch Number"][1:]) - 1
    else:
        n = to_epoch
    plt.plot(le_data["Epoch Number"][1:n], le_data["Train Error"][1:n], label="Train Error")
    plt.plot(le_data["Epoch Number"][1:n], le_data["Test Error"][1:n], label="Test Error")
    plt.xlabel("Epoch")
    plt.ylabel("Error")
    plt.legend()
    plt.savefig(config['save_path'] + print_string + 'LE_train_test_error.png')
    plt.show()
    return


def plot_usl(config, usl_data, print_string=""):
    if config["layerwise_training"]:
        usl_data = usl_data.loc[usl_data["Total Train Loss"] != 0]
        usl_data["Epoch Number"] = range(len(usl_data))
        plt.plot(usl_data["Epoch Number"], usl_data["Total Train Loss"], label="Train Loss")
    else:
        plt.plot(usl_data["Epoch Number"][1:], usl_data["Total Train Loss"][1:], label="Train Loss")
    usl_data_test = usl_data.loc[usl_data["Total Test Loss"] != 0]
    plt.plot(usl_data_test["Epoch Number"][1:], usl_data_test["Total Test Loss"][1:], label="Test Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(config['save_path'] + print_string + 'USL_train_test_loss.png')
    plt.show()
    return


def produce_usl_lineval_plots(config, usl_df=None, lineval_df=None, load_path=None):
    if (usl_df is None or lineval_df is None) and load_path is None:
        raise Exception("If no load path supplied, must supply experiment dataframes for usl and le.")
    if load_path is not None:
        le_data = pd.read_csv("LE_" + load_path)
        usl_data = pd.read_csv("USL_" + load_path)
    else:
        usl_data = usl_df
        le_data = lineval_df
    plot_usl(config, usl_data)
    plot_lineval(config, le_data)
    return

def plot_exp_set(config, folder_path):
    files = os.listdir(folder_path)
    for f in files:
        subfiles = os.listdir(folder_path + "/" + f)
        for s in subfiles:
            if "usl_data" in s:
                temp_usl_data = pd.read_csv(folder_path + "/" + f + "/" + s)


