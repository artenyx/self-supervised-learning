from src.Experiments import experiments, plots

import torch
import argparse


def main(args):
    print("========Running Network========")
    print(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
    if args.exp_type is None:
        experiments.ssl_experiment_setup(exp_type=(args.usl_type, args.denoising, args.layerwise),
                                         alpha=args.alpha,
                                         add_exp_str=args.add_exp_str,
                                         num_epochs_usl=args.epochs_usl,
                                         num_epochs_le=args.epochs_le,
                                         lr_usl=args.lr_usl,
                                         lr_le=args.lr_le,
                                         run_test_rate_usl=args.run_test_rate_usl,
                                         print_loss_rate=args.print_loss_rate,
                                         save_embeddings=args.save_embeddings,
                                         save_images=args.save_images,
                                         return_data=args.return_data,
                                         strength=args.strength
                                         )
    else:
        if args.exp_type == "alpha":
            experiments.test_alpha_parallel(args)
        elif args.exp_type == "strength":
            experiments.test_strength_single(args)
        elif args.exp_type == "ae_s_simclr":
            experiments.ae_s_simclr(args)
        elif args.exp_type == "class_from_path":
            experiments.classif_from_load_model(args)
        elif args.exp_type == "plot_folder":
            plots.plot_exp_set(args.path)
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_type", type=str, default=None, choices=["alpha", "strength", "ae_s_simclr", "plot_folder", "class_from_path"])
    parser.add_argument("--usl_type", type=str, default=None, choices=["SimCLR", "AE-S", "AE-P"])
    parser.add_argument("--denoising", type=str, default=None, choices=["D", "ND"])
    parser.add_argument("--layerwise", type=str, default=None, choices=["L", "NL"])
    parser.add_argument("--usl_load_path", type=str, default=None)
    parser.add_argument("--lr_usl", type=float, default=0.0001)
    parser.add_argument("--lr_le", type=float, default=0.001)
    parser.add_argument("--epochs_usl", type=int, default=400)
    parser.add_argument("--epochs_le", type=int, default=300)
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