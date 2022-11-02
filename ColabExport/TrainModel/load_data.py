import torch
from torchvision import datasets


def get_CIFAR10(config, CIFAR10_train=None, CIFAR10_test=None):
    transform = config['transform_dataloader']
    batch_size = config['batch_size']

    if CIFAR10_train is None:
        CIFAR10_train = datasets.CIFAR10(root="data", train=True, download=True, transform=transform)
        CIFAR10_test = datasets.CIFAR10(root="data", train=False, download=True, transform=transform)

    CIFAR10_train_loader = torch.utils.data.DataLoader(CIFAR10_train, batch_size=batch_size, shuffle=True, num_workers=12)
    CIFAR10_test_loader = torch.utils.data.DataLoader(CIFAR10_test, batch_size=batch_size, shuffle=True, num_workers=12)
    return CIFAR10_train_loader, CIFAR10_test_loader


def get_CIFAR100(config, CIFAR100_train=None, CIFAR100_test=None):
    transform = config['transform_dataloader']
    batch_size = config['batch_size']

    if CIFAR100_train is None:
      CIFAR100_train = datasets.CIFAR100(root="data", train=True, download=True, transform=transform)
      CIFAR100_test = datasets.CIFAR100(root="data", train=False, download=True, transform=transform)

    CIFAR100_train_loader = torch.utils.data.DataLoader(CIFAR100_train, batch_size=batch_size, shuffle=True, num_workers=12)
    CIFAR100_test_loader = torch.utils.data.DataLoader(CIFAR100_test, batch_size=batch_size, shuffle=True, num_workers=12)
    return CIFAR100_train_loader, CIFAR100_test_loader
