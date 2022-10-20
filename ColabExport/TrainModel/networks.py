import torch
import torch.nn as nn
from collections import OrderedDict


def initialize_layer_weights(layer_seq):
    for m in layer_seq:
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            nn.init.zeros_(m.bias)


class USL_Conv6_CIFAR_OD(nn.Module):
    def __init__(self, config):
        super(USL_Conv6_CIFAR_OD, self).__init__()
        # Setting model parameters
        self.representation_dim = config['representation_dim']
        kernel_enc, kernel_dec = 3, 2

        # Importing model configuration parameters
        self.latent_dim = config['latent_dim']
        enc_layers, dec_layers = [], []

        # Encoder Layers
        enc_layers.append(
            (str(len(enc_layers)), nn.Conv2d(config['channels'], 32, kernel_size=kernel_enc, stride=1, padding=1)))
        enc_layers.append((str(len(enc_layers)), nn.Hardtanh()))
        enc_layers.append((str(len(enc_layers)), nn.Conv2d(32, 32, kernel_size=kernel_enc, stride=1, padding=1)))
        enc_layers.append((str(len(enc_layers)), nn.MaxPool2d(2, 2)))
        enc_layers.append((str(len(enc_layers)), nn.Conv2d(32, 64, kernel_size=kernel_enc, stride=1, padding=1)))
        enc_layers.append((str(len(enc_layers)), nn.Hardtanh()))
        enc_layers.append((str(len(enc_layers)), nn.Conv2d(64, 64, kernel_size=kernel_enc, stride=1, padding=1)))
        enc_layers.append((str(len(enc_layers)), nn.MaxPool2d(2, 2)))
        enc_layers.append((str(len(enc_layers)), nn.Conv2d(64, 512, kernel_size=kernel_enc, stride=1, padding=1)))
        enc_layers.append((str(len(enc_layers)), nn.MaxPool2d(2, 2)))
        enc_layers.append((str(len(enc_layers)), nn.Flatten()))

        # Projector Layers
        proj_layers = [(str(len(enc_layers)), nn.Linear(self.representation_dim, self.latent_dim))]

        if config['usl_type'] != "simclr":
            # Decoder Layers
            dec_layers.append((str(len(enc_layers) + len(dec_layers)),
                               nn.Linear(self.latent_dim, self.representation_dim)))
            dec_layers.append((str(len(enc_layers) + len(dec_layers)),
                               nn.Unflatten(1, (512, 4, 4))))
            dec_layers.append((str(len(enc_layers) + len(dec_layers)),
                               nn.ConvTranspose2d(512, 32, kernel_size=kernel_dec, stride=2, padding=0)))
            dec_layers.append((str(len(enc_layers) + len(dec_layers)),
                               nn.Hardtanh()))
            dec_layers.append((str(len(enc_layers) + len(dec_layers)),
                               nn.ConvTranspose2d(32, 32, kernel_size=kernel_dec, stride=2, padding=0)))
            dec_layers.append((str(len(enc_layers) + len(dec_layers)),
                               nn.Hardtanh()))
            dec_layers.append((str(len(enc_layers) + len(dec_layers)),
                               nn.ConvTranspose2d(32, config['channels'], kernel_size=kernel_dec, stride=2, padding=0)))
            dec_layers.append((str(len(enc_layers) + len(dec_layers)),
                               nn.Sigmoid()))

        self.embed = nn.Sequential(OrderedDict(enc_layers))
        self.encode = nn.Sequential(OrderedDict(enc_layers + proj_layers))
        self.decode = nn.Sequential(OrderedDict(dec_layers))

        # For use in layer-wise training function
        self.encoder_layers = enc_layers
        self.projector_layers = proj_layers
        self.decoder_layers = dec_layers

        initialize_layer_weights(self.encode)
        initialize_layer_weights(self.decode)

    def forward(self, inp):
        encoded = self.encode(inp)
        decoded = self.decode(encoded)
        return encoded, decoded


class Linear_Evaluation_Classifier(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(Linear_Evaluation_Classifier, self).__init__()
        layers = [nn.Linear(in_dim, out_dim)]
        self.layers = nn.Sequential(*layers)

    def forward(self, inp):
        output = self.layers(inp)
        return output


class USL_Conv6_CIFAR1(nn.Module):
    def __init__(self, config):
        super(USL_Conv6_CIFAR1, self).__init__()
        # Setting model parameters
        self.representation_dim = config['representation_dim']
        kernel_enc, kernel_dec = 3, 3

        # Importing model configuration parameters
        self.latent_dim = config['latent_dim']
        enc_layers, dec_layers = [], []

        # Encoder Layers
        enc_layers = [nn.Conv2d(config['channels'], 32, kernel_size=kernel_enc, stride=1, padding=1),
                      nn.Hardtanh(),
                      nn.Conv2d(32, 32, kernel_size=kernel_enc, stride=1, padding=1),
                      nn.MaxPool2d(2, 2),
                      nn.Conv2d(32, 64, kernel_size=kernel_enc, stride=1, padding=1),
                      nn.Hardtanh(),
                      nn.Conv2d(64, 64, kernel_size=kernel_enc, stride=1, padding=1),
                      nn.MaxPool2d(2, 2),
                      nn.Conv2d(64, 512, kernel_size=kernel_enc, stride=1, padding=1),
                      nn.MaxPool2d(2, 2),
                      nn.Flatten()]

        # Projector Layers
        proj_layers = [nn.Linear(self.representation_dim, self.latent_dim)]

        if config['usl_type'] != "simclr":
            # Decoder Layers
            dec_layers = [nn.Linear(self.latent_dim, self.representation_dim),
                          nn.Unflatten(1, (512, 4, 4)),
                          nn.ConvTranspose2d(512, 32, kernel_size=kernel_dec, stride=2, padding=1),
                          nn.Hardtanh(),
                          nn.ConvTranspose2d(32, 32, kernel_size=kernel_dec, stride=2, padding=1),
                          nn.Hardtanh(),
                          nn.ConvTranspose2d(32, config['channels'], kernel_size=kernel_dec, stride=2, padding=1),
                          nn.Sigmoid()]

        self.embed = nn.Sequential(*enc_layers)
        self.encode = nn.Sequential(*(enc_layers + proj_layers))
        self.decode = nn.Sequential(*dec_layers)

        # For use in layer-wise training function
        self.encoder_layers = enc_layers
        self.projector_layers = proj_layers
        self.decoder_layers = dec_layers

        initialize_layer_weights(self.encode)
        initialize_layer_weights(self.decode)

    def forward(self, inp):
        encoded = self.encode(inp)
        decoded = self.decode(encoded)
        return encoded, decoded


class USL_Conv6_CIFAR2(nn.Module):
    def __init__(self, config):
        super(USL_Conv6_CIFAR2, self).__init__()
        # Setting model parameters
        self.representation_dim = config['representation_dim']
        kernel_enc, kernel_dec = 3, 4

        # Importing model configuration parameters
        self.latent_dim = config['latent_dim']
        enc_layers, dec_layers = [], []

        # Encoder Layers
        enc_layers = [nn.Conv2d(config['channels'], 32, kernel_size=kernel_enc, padding=1),
                      nn.Hardtanh(),
                      nn.Conv2d(32, 32, kernel_size=kernel_enc, padding=1),
                      nn.MaxPool2d(2),
                      nn.Conv2d(32, 64, kernel_size=kernel_enc, padding=1),
                      nn.Hardtanh(),
                      nn.Conv2d(64, 64, kernel_size=kernel_enc, padding=1),
                      nn.MaxPool2d(2),
                      nn.Conv2d(64, 512, kernel_size=kernel_enc, padding=1),
                      nn.MaxPool2d(2)]

        # Projector Layers
        proj_layers = []  # Only necessary if encoder output is not compatible with decoder input

        if config['usl_type'] != "simclr":
            # Decoder Layers
            dec_layers = [nn.ConvTranspose2d(512, 32, kernel_size=kernel_dec, stride=2, padding=1),
                          nn.Hardtanh(),
                          nn.ConvTranspose2d(32, 32, kernel_size=kernel_dec, stride=2, padding=1),
                          nn.Hardtanh(),
                          nn.ConvTranspose2d(32, config['channels'], kernel_size=kernel_dec, stride=2, padding=1),
                          nn.Sigmoid()]

        self.embed = nn.Sequential(*enc_layers)
        self.encode = nn.Sequential(*(enc_layers + proj_layers))
        self.decode = nn.Sequential(*dec_layers)

        # For use in layer-wise training function
        self.encoder_layers = enc_layers
        self.projector_layers = proj_layers
        self.decoder_layers = dec_layers

        initialize_layer_weights(self.encode)
        initialize_layer_weights(self.decode)

    def forward(self, inp):
        encoded = self.encode(inp)
        decoded = self.decode(encoded)
        return encoded, decoded


class USL_Conv6_CIFAR_Sym(nn.Module):
    def __init__(self, config):
        super(USL_Conv6_CIFAR_Sym, self).__init__()

        # Setting model parameters
        self.representation_dim = config['representation_dim']
        self.latent_dim = config['latent_dim']
        enc_layers, dec_layers = [], []

        # Encoder Layers
        enc_layers = [nn.Conv2d(config['channels'], 32, kernel_size=3, stride=1, padding=1),
                      nn.Hardtanh(),
                      nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
                      nn.MaxPool2d(2, 2),
                      nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
                      nn.Hardtanh(),
                      nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
                      nn.MaxPool2d(2, 2),
                      nn.Conv2d(64, 512, kernel_size=3, stride=1, padding=1),
                      nn.MaxPool2d(2, 2),
                      nn.Flatten()]

        # Projector Layers
        proj_layers = [nn.Linear(self.representation_dim, self.latent_dim)]

        # Decoder Layers
        dec_layers = [nn.Linear(self.latent_dim, self.representation_dim),
                      nn.Unflatten(1, (512, 4, 4)),
                      nn.ConvTranspose2d(512, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
                      nn.Hardtanh(),
                      nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
                      nn.Hardtanh(),
                      nn.ConvTranspose2d(64, 32, kernel_size=3, stride=1, padding=1),
                      nn.Hardtanh(),
                      nn.ConvTranspose2d(32, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
                      nn.Hardtanh(),
                      nn.ConvTranspose2d(32, config['channels'], kernel_size=3, stride=1, padding=1),
                      nn.Sigmoid()]

        self.embed = nn.Sequential(*enc_layers)
        self.encode = nn.Sequential(*(enc_layers + proj_layers))
        self.decode = nn.Sequential(*dec_layers)

        # For use in layer-wise training function
        self.encoder_layers = enc_layers
        self.projector_layers = proj_layers
        self.decoder_layers = dec_layers

        initialize_layer_weights(self.encode)
        initialize_layer_weights(self.decode)

    def forward(self, inp):
        encoded = self.encode(inp)
        decoded = self.decode(encoded)
        return encoded, decoded


class USL_Conv6_CIFAR_LC(nn.Module):
    # USL Network where the layers are passed into the class instance, currently being used in layerwise training function
    def __init__(self, config, enc_layers, dec_layers, proj_layers=[]):
        super(USL_Conv6_CIFAR_LC, self).__init__()

        # Setting model parameters
        self.representation_dim = config['representation_dim']
        self.latent_dim = config['latent_dim']

        # Projector Layers
        self.embed = nn.Sequential(*enc_layers)
        self.encode = nn.Sequential(*(enc_layers + proj_layers))
        self.decode = nn.Sequential(*dec_layers)

        # For use in layer-wise training function
        self.encoder_layers = enc_layers
        self.projector_layers = proj_layers
        self.decoder_layers = dec_layers

        initialize_layer_weights(self.encode)
        initialize_layer_weights(self.decode)

    def forward(self, inp):
        encoded = self.encode(inp)
        decoded = self.decode(encoded)
        return encoded, decoded
