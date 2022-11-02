import torch
from torchvision import datasets, transforms as T


def get_cifar10_classif(config):
    transform = config['transform']
    batch_size = config['batch_size']

    CIFAR10_train = datasets.CIFAR10(root="data", train=True, download=True, transform=transform)
    CIFAR10_test = datasets.CIFAR10(root="data", train=False, download=True, transform=transform)

    CIFAR10_train_loader = torch.utils.data.DataLoader(CIFAR10_train, batch_size=batch_size, shuffle=True, num_workers=12)
    CIFAR10_test_loader = torch.utils.data.DataLoader(CIFAR10_test, batch_size=batch_size, shuffle=True, num_workers=12)
    return CIFAR10_train_loader, CIFAR10_test_loader


def get_cifar100_usl(config):
    transform = config['transform']
    batch_size = config['batch_size']

    if config['usl_type'] == 'ae_single' and not config['denoising']:
        dataset_list_train = [datasets.CIFAR100(root="data", train=True, download=True, transform=T.ToTensor())]
        dataset_list_test = [datasets.CIFAR100(root="data", train=False, download=True, transform=T.ToTensor())]
    elif config['usl_type'] == 'ae_single' and config['denoising']:
        dataset_list_train = [datasets.CIFAR100(root="data", train=True, download=True, transform=T.ToTensor()),
                              datasets.CIFAR100(root="data", train=True, download=True, transform=transform)]
        dataset_list_test = [datasets.CIFAR100(root="data", train=False, download=True, transform=T.ToTensor()),
                             datasets.CIFAR100(root="data", train=False, download=True, transform=transform)]
    else:
        dataset_list_train = [datasets.CIFAR100(root="data", train=True, download=True, transform=T.ToTensor()),
                              datasets.CIFAR100(root="data", train=True, download=True, transform=transform),
                              datasets.CIFAR100(root="data", train=True, download=True, transform=transform)]
        dataset_list_test = [datasets.CIFAR100(root="data", train=False, download=True, transform=T.ToTensor()),
                             datasets.CIFAR100(root="data", train=False, download=True, transform=transform),
                             datasets.CIFAR100(root="data", train=False, download=True, transform=transform)]

    train_loader = torch.utils.data.DataLoader(dataset_list_train, batch_size=batch_size, shuffle=True, num_workers=12)
    test_loader = torch.utils.data.DataLoader(dataset_list_test, batch_size=batch_size, shuffle=True, num_workers=12)
    return train_loader, test_loader
