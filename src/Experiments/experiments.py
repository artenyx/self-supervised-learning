import torch
from datetime import datetime

from src.TrainModel import train, networks, load_data
from src.Experiments import exp_config


def run_representation_learning(config, model):
    if config['layerwise_training']:
        assert config['usl_type'] == 'ae_single' or config['usl_type'] == 'ae_parallel'
        usl_data, usl_model = train.usl_train_network_layerwise(model, config)
    else:
        usl_data, usl_model = train.usl_train_network(model, config)
    return usl_data, usl_model


def run_linear_evaluation(config):
    if config['loaders']['loaders_embedded'] is not None:
        config['loaders']['loaders_le'] = config['loaders']['loaders_embedded']
    train_loader, test_loader = config['loaders']['loaders_le']
    le_model = networks.Linear_Evaluation_Classifier(train_loader.dataset[0][0].shape[0], config['num_classes']).to(
        config['device'])
    le_data, le_model = train.classifier_train_network(le_model, config)
    return le_data, le_model


def run_ssl_experiment(config, exp_string, rep_learning_model=None, save=True):
    exp_config.make_dir("ExperimentFiles")
    exp_path = "ExperimentFiles/" + exp_string + "/"
    exp_config.make_dir(exp_path)
    config["save_path"] = exp_path

    if rep_learning_model is None:
        if config['layerwise_training']:
            rep_learning_model = networks.USL_Conv6_CIFAR_Sym(config).to(config['device'])
            # print(summary(rep_learning_model, (3, 32, 32), batch_size=256))
        else:
            rep_learning_model = networks.USL_Conv6_CIFAR1(config).to(config['device'])

    usl_data, usl_model = run_representation_learning(config, rep_learning_model)

    emb_train_loader = train.get_embedding_loader(usl_model, config, config['loaders']['loaders_le'][0])
    emb_test_loader = train.get_embedding_loader(usl_model, config, config['loaders']['loaders_le'][1])
    config['loaders']['loaders_embedded'] = emb_train_loader, emb_test_loader
    # plots.produce_embedding_plots(samples_to_use=1000, config=config, get_loader_from_config=True, pca_or_tsne="tsne")

    le_data, le_model = run_linear_evaluation(config)
    with open(exp_path + "config.txt", "w") as f:
        f.write(str(config))
    if save:
        exp_config.save_data_model(config, 'USL', usl_data, usl_model)
        exp_config.save_data_model(config, 'LE', le_data, le_model)

    return usl_data, usl_model, le_data, le_model


def print_model_architecture(model_type, input_size=(3, 32, 32)):
    config = exp_config.get_exp_config()
    config['model_type'] = model_type
    model = config['model_type'](config).to(config['device'])
    # summary(model, (3, 32, 32))


def ssl_experiment_setup(usl_type,
                         denoising,
                         layerwise,
                         alpha=None,
                         config=None,
                         add_exp_str='',
                         num_epochs_usl=200,
                         num_epochs_le=150,
                         lr_usl=0.001,
                         lr_le=0.01,
                         run_test_rate_usl=1,
                         print_loss_rate=50,
                         save_images=True,
                         save_embeddings=True,
                         return_data=False,
                         strength=0.25,
                         crit_emb="l2",
                         crit_emb_lam=None,
                         crit_recon="l2",
                         seed=1234,
                         batch_size=512,
                         trans_active=None,
                         crop_size=24):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    if config is None:
        config = exp_config.get_exp_config(s=strength, crop_size=crop_size)
    config['batch_size'] = batch_size
    config['usl_type'] = usl_type
    assert usl_type == "ae_single" or usl_type == "ae_parallel" or usl_type == "simclr" or usl_type == "simsiam", "Wrong USL type."
    if config['usl_type'] == 'ae_parallel':
        config['alpha'] = alpha
    config['denoising'] = denoising
    config['layerwise_training'] = layerwise
    if layerwise:
        model_type = networks.USL_Conv6_CIFAR_Sym
    else:
        model_type = networks.USL_Conv6_CIFAR1

    config['num_epochs_usl'] = num_epochs_usl
    config['num_epochs_le'] = num_epochs_le

    config['save_images'] = save_images
    config['run_test_rate_usl'] = run_test_rate_usl
    config['print_loss_rate'] = print_loss_rate
    config['save_embeddings'] = save_embeddings
    exp_type = [usl_type] if "ae" not in usl_type else [usl_type, "D" if denoising else "ND", "L" if layerwise else "NL"]
    config['exp_type'] = "-".join(exp_type)
    config['lr_usl'] = lr_usl
    config['lr_le'] = lr_le
    config['criterion_recon'] = crit_recon
    config['criterion_emb'] = crit_emb
    if crit_emb == "bt" and crit_emb_lam is None:
        config['criterion_emb_lam'] = 0.001
    elif (crit_emb == "simclr" and crit_emb_lam is None) or usl_type == "simclr":
        config['criterion_emb_lam'] = 0.5

    if trans_active == "full":
        config['transforms_active'] = config['transforms_list_full']
    elif trans_active is not None:
        config['transforms_active'] = trans_active
    config['loaders']['loaders_usl'] = load_data.get_cifar100_usl(config)
    config['loaders']['loaders_le'] = load_data.get_cifar10_classif(config)

    print(config)
    config['model_type'] = model_type
    model = config['model_type'](config).to(config['device'])
    date_time = datetime.now().strftime("%m-%d-%Y_%H-%M-%S")
    exp_folder_name = "-".join(exp_type) + "_" + str(date_time) + add_exp_str
    usl_data, usl_model, le_data, le_model = run_ssl_experiment(config, exp_folder_name, rep_learning_model=model)
    if return_data:
        return usl_data, usl_model, le_data, le_model


def ssl_exp_from_args(args):
    ssl_experiment_setup(usl_type=args.usl_type,
                         denoising=args.denoising,
                         layerwise=args.layerwise,
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
                         strength=args.strength,
                         crit_emb=args.crit_emb,
                         crit_recon=args.crit_recon,
                         batch_size=args.batch_size,
                         trans_active=args.trans_active,
                         crop_size=args.crop_size
                         )


def alpha_exp(args):
    if args.usl_type != "ae_parallel":
        print("usl_type will be reset to \"ae_parallel\" for this experiment.")
    alpha_list = [0.0, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000, 100000]
    #alpha_list = [1000, 10000, 100000]
    print("RUNNING AE_PARALLEL AT ALPHAS: " + str(alpha_list))
    print(args.layerwise)
    print(args.denoising)
    exp_str_init = args.add_exp_str
    for alpha0 in alpha_list:
        args.usl_type, args.alpha, args.add_exp_str = "ae_parallel", alpha0, exp_str_init + "alpha-"+str(alpha0)
        ssl_exp_from_args(args)


def ae_s_simclr(args):
    args.usl_type, args.denoising, args.layerwise = "ae_single", False, False
    ssl_exp_from_args(args)

    args.usl_type, args.denoising, args.layerwise = "ae_single", True, False
    ssl_exp_from_args(args)

    args.usl_type, args.denoising, args.layerwise = "ae_single", True, True
    ssl_exp_from_args(args)

    args.usl_type, args.denoising, args.layerwise = "simclr", False, False
    ssl_exp_from_args(args)


def transforms_exp(args):
    transforms_list_full = ["ToTens", "Crop", "HorFlip", "ColJit", "GausBlur", "Solar"]
    trans_test_list = [range(2), range(3), range(4), range(5), range(6), [0, 1, 2, 3], [0, 1, 2, 4]]
    for i, trans_idx in enumerate(trans_test_list):
        args.add_exp_str = "trans-" + str(i)
        args.trans_active = [transforms_list_full[j] for j in trans_idx]
        ssl_exp_from_args(args)


def crop_size_exp(args):
    crop_size_list = [4, 8, 12, 16, 20, 24, 28, 32]
    for i, crop_sz in enumerate(crop_size_list):
        args.add_exp_str = "cropsz-" + str(crop_sz)
        args.crop_size = crop_sz
        print(args.crop_size)
        ssl_exp_from_args(args)


def strength_exp(args):
    strength_list = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5]
    print("RUNNING AT STRENGTHS: " + str(strength_list))
    for strength in strength_list:
        args.add_exp_str = "strength-" + str(strength)
        args.strength = strength
        ssl_exp_from_args(args)


def bs_exp(args):
    bs_list = [1024, 512, 256, 124, 64, 8, 4]
    print("RUNNING AT BATCH SIZES: " + str(bs_list))
    for bs in bs_list:
        args.add_exp_str = "bs-" + str(bs)
        args.batch_size = bs
        ssl_exp_from_args(args)


def usl_lr_exp(args):
    lr_list = [0.000001, 0.00001, 0.0001, 0.001, 0.01]
    print("TESTING LEARNING RATES: " + str(lr_list))
    exp_str_init = args.add_exp_str
    for lr in lr_list:
        args.lr_usl = lr
        args.add_exp_str = exp_str_init + "lr-" + str(lr)
        ssl_exp_from_args(args)


def usl_epoch_exp(args):
    epochs_list = [10, 50, 100, 150, 200, 300, 400]
    print("TESTING USL N_EPOCHS: " + str(epochs_list))
    exp_str_init = args.add_exp_str
    for epochs in epochs_list:
        args.epochs_usl = epochs
        args.add_exp_str = exp_str_init + "epochs-" + str(epochs)
        ssl_exp_from_args(args)


def usl_lr_bs_exp(args):
    lr_list = [0.000001, 0.00001, 0.0001, 0.001]
    bs_list = [4096, 1024, 512, 256, 128, 64]
    print("RUNNING BS/LR EXP...")
    exp_str_init = args.add_exp_str
    for bs in bs_list:
        for lr in lr_list:
            args.lr_usl = lr
            args.batch_size = bs
            args.add_exp_str = exp_str_init + "lr-" + str(lr) + "-bs-" + str(bs)
            ssl_exp_from_args(args)


def classif_from_load_model(args, usl_model=None):
    config = exp_config.get_exp_config()
    if usl_model is None:
        usl_model = networks.USL_Conv6_CIFAR1(config=config).to(config['device'])
        usl_model.load_state_dict(torch.load(args.usl_load_path)['model.state.dict'])
    config['num_epochs_le'] = args.epochs_le
    config['loaders']['loaders_le'] = load_data.get_cifar10_classif(config)
    emb_train_loader = train.get_embedding_loader(usl_model, config, config['loaders']['loaders_le'][0])
    emb_test_loader = train.get_embedding_loader(usl_model, config, config['loaders']['loaders_le'][1])
    config['loaders']['loaders_embedded'] = emb_train_loader, emb_test_loader

    le_data, le_model = run_linear_evaluation(config)
    le_data.to_csv("ExperimentFiles/classif_from_load.csv")


