from src.Experiments import experiments, plots

import torch
import argparse

exp_funct_dict = {
    "alpha": experiments.alpha_exp_from_args,
    "ae_s_simclr": experiments.ae_s_simclr,
    "class_from_path": experiments.classif_from_load_model,
    "plot_folder": plots.plot_exp_set,
    "usl_lr_exp": experiments.usl_lr_exp,
    None: experiments.ssl_exp_from_args
}


def main(args):
    print("========Running Network========")
    print("Device: cuda" if torch.cuda.is_available() else "Device: cpu")
    if args.strength_exp:
        experiments.strength_exp_wrapper(args, exp_funct_dict[args.exp_type])
    else:
        exp_funct_dict[args.exp_type](args)
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_type", type=str, default=None, choices=list(exp_funct_dict.keys()))
    parser.add_argument("--strength_exp", type=bool, default=False)
    parser.add_argument("--usl_type", type=str, default=None, choices=["simclr", "ae_single", "ae_parallel"])
    parser.add_argument("--denoising", type=bool, default=False)
    parser.add_argument("--layerwise", type=bool, default=False)
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
    parser.add_argument("--save_images", type=bool, default=True)
    parser.add_argument("--return_data", type=bool, default=True)
    parser.add_argument("--strength", type=float, default=0.25)
    parser.add_argument("--path", type=str, default=None)
    exp_args = parser.parse_args()
    main(exp_args)
