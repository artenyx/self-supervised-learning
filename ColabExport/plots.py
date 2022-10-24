import torch
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np


def plot_pca(config, exp_string):
    emb_loaders = torch.load(config['data_save_path'] + '_embloaders_' + exp_string + '.pt')
    train_loader, test_loader = emb_loaders["embedding_train_loader"], emb_loaders["embedding_test_loader"]
    test_loader.dataset.data.reshape((-1, 3072))
    pca = PCA(n_components=100)

    train_set_pca = pca.fit_transform(test_loader)
    exp_var_pca = pca.explained_variance_ratio_
    cum_sum_eigenvalues = np.cumsum(exp_var_pca)

    plt.bar(range(0,len(exp_var_pca)), exp_var_pca, alpha=0.5, align='center', label='Individual explained variance')
    plt.step(range(0,len(cum_sum_eigenvalues)), cum_sum_eigenvalues, where='mid', label='Cumulative explained variance')
    plt.ylabel('Explained variance ratio')
    plt.xlabel('Principal component index')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()
    plt.savefig(config['data_save_path'] + '_emb_pcacumsum_' + exp_string + '.pt')
