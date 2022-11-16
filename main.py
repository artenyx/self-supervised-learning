from src.Experiments import experiments

import torch
import argparse


def main(args):
    print("========Running Network========")
    print(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
    if args.ssl_type is None:
        experiments.ssl_experiment_setup(exp_type=(args.usl_type, args.denoising, args.layerwise))
    else:
        if args.exp_type == "alpha":
            experiments.test_alpha_parallel([0.0001, 0.001, 0.01, 0.1, 0.0, 0.1, 1.0, 10])
        elif args.exp_type == "strength":
            experiments.test_strength_single([0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0])
        elif args.exp_type == "class_from_path":
            experiments.classif_from_load_model(load_path=args.usl_load_path)
    return


if __name__ == "main":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_type", type=str, default=None, choices=["alpha", "strength", "class_from_path"])

    parser.add_argument("--ssl_type", type=tuple, default=None, help="What kind of experiment to run, first argument "
                                                                     "should be AE-S, AE-P, or SimCLR. Second "
                                                                     "element should be D or ND, Third should be L "
                                                                     "or NL.")
    parser.add_argument("--usl_type", type=str, default=None, choices=["SimCLR", "AE-S", "AE-P"])
    parser.add_argument("--denoising", type=str, default=None, choices=["D", "ND"])
    parser.add_argument("--layerwise", type=str, default=None, choices=["L", "NL"])
    parser.add_argument("--usl_load_path", type=str, default=None)
    exp_args = parser.parse_args()
    main(exp_args)
