from src.Experiments import experiments, plots, kmeans

import torch
import argparse


exp_funct_dict = {
    "alpha": experiments.alpha_exp,
    "strength": experiments.strength_exp,
    "lr_usl": experiments.usl_lr_exp,
    "transforms": experiments.transforms_exp,
    "crop_size": experiments.crop_size_exp,
    "bs": experiments.bs_exp,
    "epochs_usl": experiments.usl_epoch_exp,
    "ae_s_simclr": experiments.ae_s_simclr,
    "class_from_path": experiments.classif_from_load_model,
    "plot_folder": plots.plot_exp_set,
    "kmeans_all": kmeans.kmeans_all_exps,
    None: experiments.ssl_exp_from_args
}


def main(args):
    print("========Running Network========")
    print("Device: cuda" if torch.cuda.is_available() else "Device: cpu")
    exp_funct_dict[args.exp_type](args)
    print("All experiments completed.")
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_type", type=str, default=None, choices=list(exp_funct_dict.keys()))
    parser.add_argument("--usl_type", type=str, default=None, choices=["simclr", "ae_single", "ae_parallel"])
    parser.add_argument("--denoising", action="store_true")
    parser.add_argument("--layerwise", action="store_true")
    parser.add_argument("--usl_load_path", type=str, default=None)
    parser.add_argument("--lr_usl", type=float, default=0.0001)
    parser.add_argument("--lr_le", type=float, default=0.001)
    parser.add_argument("--epochs_usl", type=int, default=200)
    parser.add_argument("--epochs_le", type=int, default=150)
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--add_exp_str", type=str, default="")
    parser.add_argument("--run_test_rate_usl", type=int, default=1)
    parser.add_argument("--print_loss_rate", type=int, default=50)
    parser.add_argument("--save_embeddings", type=bool, default=False)
    parser.add_argument("--save_images", action="store_false")
    parser.add_argument("--return_data", action="store_false")
    parser.add_argument("--strength", type=float, default=0.25)
    parser.add_argument("--path", type=str, default=None)
    parser.add_argument("--crit_emb", type=str, default="l2", choices=["l1", "l2", "bt", "simclr", "cos"])
    parser.add_argument("--crit_recon", type=str, default="l2", choices=["l1", "l2", "bt", "simclr", "cos"])
    parser.add_argument("--crit_emb_lam", type=float, default=None)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--crop_size", type=int, default=24)
    parser.add_argument("--transforms_full", action="store_true")
    exp_args = parser.parse_args()
    exp_args.trans_active = "full" if exp_args.transforms_full else None
    main(exp_args)
