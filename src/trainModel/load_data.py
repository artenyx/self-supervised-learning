import torch
from torchvision import datasets, transforms as T


def get_cifar10_classif(config):
    batch_size = config['batch_size']
    transforms = T.Compose([config['transforms_dict'][key] for key in config['transforms_active']])

    dataset_list_train = [datasets.CIFAR10(root="data", train=True, download=True, transform=T.ToTensor())]
    dataset_list_test = [datasets.CIFAR10(root="data", train=False, download=True, transform=T.ToTensor())]
                         #datasets.CIFAR10(root="data", train=False, download=True, transform=transforms)]

    train_loader = torch.utils.data.DataLoader(list(zip(*dataset_list_train)), batch_size=batch_size, shuffle=True, num_workers=12)
    test_loader = torch.utils.data.DataLoader(list(zip(*dataset_list_train)), batch_size=batch_size, shuffle=True, num_workers=12)

    return train_loader, test_loader


def get_cifar100_usl(config):
    transforms = T.Compose([config['transforms_dict'][key] for key in config['transforms_active']])
    print(transforms)
    batch_size = config['batch_size']

    if config['usl_type'] == 'ae_single' and not config['denoising']:
        dataset_list_train = [datasets.CIFAR100(root="data", train=True, download=True, transform=T.ToTensor())]
        dataset_list_test = [datasets.CIFAR100(root="data", train=False, download=True, transform=T.ToTensor())]
    elif config['usl_type'] == 'ae_single' and config['denoising']:
        dataset_list_train = [datasets.CIFAR100(root="data", train=True, download=True, transform=T.ToTensor()),
                              datasets.CIFAR100(root="data", train=True, download=True, transform=transforms)]
        dataset_list_test = [datasets.CIFAR100(root="data", train=False, download=True, transform=T.ToTensor()),
                             datasets.CIFAR100(root="data", train=False, download=True, transform=transforms)]
    else:
        dataset_list_train = [datasets.CIFAR100(root="data", train=True, download=True, transform=T.ToTensor()),
                              datasets.CIFAR100(root="data", train=True, download=True, transform=transforms),
                              datasets.CIFAR100(root="data", train=True, download=True, transform=transforms)]
        dataset_list_test = [datasets.CIFAR100(root="data", train=False, download=True, transform=T.ToTensor()),
                             datasets.CIFAR100(root="data", train=False, download=True, transform=transforms),
                             datasets.CIFAR100(root="data", train=False, download=True, transform=transforms)]
    if batch_size < 64:
        nworkers = 6
    else:
        nworkers = 12
    train_loader = torch.utils.data.DataLoader(list(zip(*dataset_list_train)), batch_size=batch_size, shuffle=True, num_workers=nworkers)
    test_loader = torch.utils.data.DataLoader(list(zip(*dataset_list_test)), batch_size=batch_size, shuffle=True, num_workers=nworkers)
    return train_loader, test_loader
