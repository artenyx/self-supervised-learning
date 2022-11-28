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


def listdir_nohidden(folder_path, dir_only):
    for f in os.listdir(folder_path):
        if not f.startswith('.') and not f.startswith('000_'):
            if dir_only and os.path.isdir(folder_path + "/" + f):
                yield f
            elif not dir_only:
                yield f


def pp_data(df, usl=True):
    df_cp = df.copy()
    if usl:
        df_cp = df_cp.loc[(df_cp["Total Train Loss"] != 0) & (df_cp["Total Test Loss"] != 0)]
    else:
        df_cp = df_cp.loc[(df_cp["Train Error"] != 0) & (df_cp["Test Error"] != 0)]
    return df_cp


def plot_from_dicts(folder_path, data_dict, usl, xcol=None, ycols=None, pp=True):
    folder_path += "/000_plots/usl/" if usl else "/000_plots/le/"
    min_data = []
    if xcol is None:
        xcol = "Epoch Number"
    if ycols is None:
        ycols = ["Total Train Loss", "Total Test Loss"] if usl else ["Train Error", "Test Error"]
    os.makedirs(folder_path, exist_ok=True)

    epochs_list, tr_data_list, te_data_list = [], [], []
    exp_list = list(data_dict.keys())
    for exp in exp_list:
        data = pp_data(data_dict[exp], usl) if pp else data_dict[exp]
        plt.plot(data[xcol], data[ycols[0]])
        plt.plot(data[xcol], data[ycols[1]])
        min_data.append((exp, np.argmin(data[ycols[0]]), np.min(data[ycols[0]]),  np.argmin(data[ycols[1]]), np.min(data[ycols[1]])))
        plt.xlabel(xcol)
        plt.ylabel("Loss" if usl else "Error")
        plt.savefig(folder_path + exp + "_usl.png")
        plt.close()
        epochs_list.append(data[xcol])
        tr_data_list.append(data[ycols[0]])
        te_data_list.append(data[ycols[1]])
    min_data = pd.DataFrame(min_data)
    min_data = min_data.set_axis(["Exp", "Min Train Idx", "Min Train Val", "Min Test Idx", "Min Test Val"], axis=1)
    min_data.to_csv(folder_path + "min.csv")

    for i, exp in enumerate(exp_list):
        plt.plot(epochs_list[i], tr_data_list[i], label=exp)
    plt.xlabel(xcol)
    plt.ylabel("Loss" if usl else "Error")
    plt.savefig(folder_path + "usl_tr_all.png" if usl else folder_path + "le_tr_all.png")
    plt.legend()
    plt.savefig(folder_path + "usl_tr_all_leg.png" if usl else folder_path + "le_tr_all_leg.png")
    min_val = np.min(min_data['Min Train Val'])
    plt.ylim(0.9*min_val, 1.1*min_val)
    plt.legend()
    plt.savefig(folder_path + "usl_tr_all_zoom.png" if usl else folder_path + "le_tr_all_zoom.png")
    plt.close()
    for i, exp in enumerate(exp_list):
        plt.plot(epochs_list[i], te_data_list[i], label=exp)
    plt.xlabel(xcol)
    plt.ylabel("Loss" if usl else "Error")
    plt.savefig(folder_path + "usl_te_all.png" if usl else folder_path + "le_te_all.png")
    plt.legend()
    plt.savefig(folder_path + "usl_te_all_leg.png" if usl else folder_path + "le_te_all_leg.png")
    min_val = np.min(min_data['Min Test Val'])
    plt.ylim(0.9*min_val, 1.1*min_val)
    plt.legend()
    plt.savefig(folder_path + "usl_te_all_zoom.png" if usl else folder_path + "le_te_all_zoom.png")
    plt.close()


def plot_exp_set(folder_path):
    usl_data_dict = {}
    le_data_dict = {}
    exp_files = list(listdir_nohidden(path, True))
    for f in exp_files:
        subpath = folder_path + "/" + f
        subfiles = list(listdir_nohidden(subpath, False))
        for s in subfiles:
            if "USL_data" in s:
                temp_usl_data = pd.read_csv(folder_path + "/" + f + "/" + s)
                usl_data_dict[f] = temp_usl_data
            elif "LE_data" in s:
                temp_le_data = pd.read_csv(folder_path + "/" + f + "/" + s)
                le_data_dict[f] = temp_le_data

    plot_from_dicts(folder_path, usl_data_dict, True)
    plot_from_dicts(folder_path, le_data_dict, False)


if __name__ == "__main__":
    path = "/Users/jerrywhite/Documents/01 - University of Chicago/05 - Thesis/01 - Thesis Experiments/200E/Trans_simclr"
    plot_exp_set(path)
