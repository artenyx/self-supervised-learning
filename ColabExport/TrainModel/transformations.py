import torch
import torch.nn.functional as F


def deform_data(config, x_in, factor=.6):
    h = x_in.shape[2]
    w = x_in.shape[3]
    nn = x_in.shape[0]
    u = ((torch.rand(nn, 6) - .5) * factor).to(config['device'])
    # Amplify the shift part of the
    # u[:,[2,5]]*=2
    if True: # no skew?
        u[:, [1, 3]] = 0
    rr = torch.zeros(nn, 6).to(config['device'])
    rr[:, 0] = 1
    rr[:, 4] = 1
    theta = (u + rr).view(-1, 2, 3)
    grid = F.affine_grid(theta, [nn, 1, h, w], align_corners=True)
    x_out = F.grid_sample(x_in, grid, padding_mode='zeros', align_corners=True)

    return x_out
