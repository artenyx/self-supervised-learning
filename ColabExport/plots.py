import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from ColabExport import exp_config


def plot_pca(config, dataset_array, print_string='pca_fig.png'):
    pca = PCA(n_components=100)

    dataset_array_pca = pca.fit_transform(dataset_array)
    exp_var_pca = pca.explained_variance_ratio_
    cum_sum_eigenvalues = np.cumsum(exp_var_pca)

    plt.bar(range(0,len(exp_var_pca)), exp_var_pca, alpha=0.5, align='center', label='Individual explained variance')
    plt.step(range(0,len(cum_sum_eigenvalues)), cum_sum_eigenvalues, where='mid', label='Cumulative explained variance')
    plt.ylabel('Explained variance ratio')
    plt.xlabel('Principal component index')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()
    plt.savefig(config['data_save_path'] + print_string)


def plot_tsne(config, dataset_array, target_array, print_string='tsne_fig.png'):
    tsne = TSNE(n_components=2, verbose=1, random_state=123)
    z = tsne.fit_transform(dataset_array)

    df = pd.DataFrame()
    df["y"] = target_array
    df["comp-1"] = z[:, 0]
    df["comp-2"] = z[:, 1]

    plot = sns.scatterplot(x="comp-1", y="comp-2", hue=df.y.tolist(), palette=sns.color_palette("hls", 10), data=df).set(title="Embedding t-SNE Projection")
    fig = plot.get_figure()
    fig.savefig(config['data_save_path'] + print_string)


def emb_loader_to_array(load_path, config=None, get_loader_from_config=False):
    if config is None:
        config = exp_config.get_exp_config()
        config = exp_config.reset_config_paths_colab(config)
        loaders = torch.load(load_path)
        emb_dataset = loaders["embedding_train_loader"].dataset
    elif get_loader_from_config:
        loaders = config["loaders"]["loaders_embedded"]
        emb_dataset = loaders[0].dataset
    else:
        loaders = torch.load(load_path)
        emb_dataset = loaders["embedding_train_loader"].dataset

    emb_dataset_array = np.array([tup[0].cpu().detach().numpy() for tup in emb_dataset])
    emb_dataset_array = emb_dataset_array.reshape((-1, np.prod(emb_dataset_array.shape[1:])))
    print(emb_dataset_array.shape)

    emb_target_array = np.array([tup[1].cpu().detach().numpy() for tup in emb_dataset])
    print(emb_target_array.shape)
    return emb_dataset_array, emb_target_array


'''  
config = exp_config.get_exp_config()
config = exp_config.reset_config_paths_colab(config)
loaders = torch.load(config['data_save_path']+"_embloaders_AE-S-D-USL_Conv6_CIFAR1.pt")
#need to create a way to extract only first element of tuples

emb_dataset = loaders["embedding_train_loader"].dataset
emb_dataset_array = np.array([tup[0].cpu().detach().numpy() for tup in emb_dataset])
emb_dataset_array = emb_dataset_array.reshape((-1, np.prod(emb_dataset_array.shape[1:])))
print(emb_dataset_array.shape)

emb_target_array = np.array([tup[1].cpu().detach().numpy() for tup in emb_dataset])
print(emb_target_array.shape)

plot_pca(config, emb_dataset_array) #, emb_target_array) 
'''
