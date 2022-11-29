import torch
import torch.nn as nn
from torchvision.utils import save_image
from torch.nn.functional import normalize


def standardize(tensor):
    # Vectorize each example
    tensor = tensor.reshape(tensor.shape[0], -1)
    sd = torch.sqrt(torch.sum(tensor * tensor, dim=1)).reshape(-1, 1)
    tensor = tensor / (sd + 0.001)
    return tensor


def simclr_loss_func(embedding1, embedding2, lam=0.5):
    assert embedding1.shape == embedding2.shape
    batch_size = embedding1.shape[0]

    # Standardize 64 dim outputs of original and deformed images
    embedding1_stand = standardize(embedding1)
    embedding2_stand = standardize(embedding2)
    # Compute 3 covariance matrices - 0-1, 0-0, 1-1.
    cov12 = torch.mm(embedding1_stand, embedding2_stand.transpose(0, 1))  # COV
    cov11 = torch.mm(embedding1_stand, embedding1_stand.transpose(0, 1))  # COV0
    cov22 = torch.mm(embedding2_stand, embedding2_stand.transpose(0, 1))  # COV1
    # Diagonals of covariances.
    d12 = torch.diag(cov12)  # v
    d11 = torch.diag(cov11)  # v0
    d22 = torch.diag(cov22)  # v1
    # Mulitnomial logistic loss just computed on positive match examples, with all other examples as a separate class.
    lecov = torch.log(
        torch.exp(torch.logsumexp(cov12, dim=1)) + torch.exp(torch.logsumexp(cov11 - torch.diag(d11), dim=1)))
    lecov += torch.log(
        torch.exp(torch.logsumexp(cov12, dim=1)) + torch.exp(torch.logsumexp(cov22 - torch.diag(d22), dim=1)))
    lecov = lam * lecov - d12

    loss = torch.mean(lecov)

    '''
    # Accuracy
      if torch.cuda.is_available():
        ID = 2. * torch.eye(batch_size).to('cuda') - 1.
      else:
        ID = 2. *torch.eye(batch_size) - 1
      icov=ID*cov12
      acc=torch.sum((icov>0).type(torch.float))/ batch_size
      '''

    return loss


def barlow_twins_loss_func(embedding1, embedding2, lam):
    # normalizing embeddings
    embedding1 = normalize(embedding1)
    embedding2 = normalize(embedding2)

    # embedding should be shape [batchsize,latent_dim]
    batchsize = embedding1.shape[0]
    latent_dim = embedding1.shape[1]

    # latent_dim square matrix
    c = torch.mm(embedding1.T, embedding2) / batchsize

    # creating cross correlation matrix and squaring entries
    if torch.cuda.is_available():
        eye = torch.eye(latent_dim, device="cuda")
    else:
        eye = torch.eye(latent_dim)
    c_diff = torch.pow(c - eye, 2)

    # multiplying off diagonal entries by alpha parameter
    # torch.off_diagonal(c_diff).mul_(alpha)
    c_diff_copy = lam * c_diff
    c_diff_copy.diag = c_diff.diag

    loss = torch.sum(c_diff_copy)
    return loss


loss_dict = {
    "l2": nn.MSELoss(),
    "l1": nn.L1Loss(),
    "bt": barlow_twins_loss_func,
    "simclr": simclr_loss_func,
    "cos": nn.CosineEmbeddingLoss(),
}


def simclr_run_loss(model, config, img1, img2):
    encoding1, __ = model(img1)
    encoding2, __ = model(img2)
    loss_total = simclr_loss_func(encoding1, encoding2, lam=config['criterion_emb_lam'])

    return None, None, None, loss_total


def ae_parallel_run_loss(model, config, epoch, img0, img1, img2):

    criterion_recon = loss_dict[config['criterion_recon']]
    criterion_emb = loss_dict[config['criterion_emb']]

    encoding1, decoding1 = model(img1)
    encoding2, decoding2 = model(img2)
    if config['save_image_flag'] and config['save_images']:
        save_inp_rec_images(config, epoch, img0, [img1, img2], [decoding1, decoding2])

    if config['denoising']:
        loss_img1 = criterion_recon(img0, decoding1)
        loss_img2 = criterion_recon(img0, decoding2)
    else:
        loss_img1 = criterion_recon(img1, decoding1)
        loss_img2 = criterion_recon(img2, decoding2)

    if config['criterion_emb'] == "bt" or config['criterion_emb'] == "simclr":
        loss_emb = criterion_emb(encoding1, encoding2, lam=config['criterion_emb_lam'])
    else:
        loss_emb = criterion_emb(encoding1, encoding2)

    loss_total = torch.sum(loss_img1 + loss_img2 + config['alpha'] * loss_emb / config['latent_dim'])

    return loss_img1, loss_img2, loss_emb, loss_total


def ae_single_run_loss(model, config, epoch, img0, img1):

    criterion_recon = loss_dict[config['criterion_recon']]

    encoding1, decoding1 = model(img1)
    if config['save_image_flag'] and config['save_images']:
        save_inp_rec_images(config, epoch, img0, [img1], [decoding1])

    if config['denoising']:
        loss_img1 = criterion_recon(img0, decoding1)
    else:
        loss_img1 = criterion_recon(img1, decoding1)

    loss_total = torch.sum(loss_img1)

    return loss_img1, None, None, loss_total


def loss_add(loss_img1, loss_img2, loss_emb, loss_total, tot_loss_img1, tot_loss_img2, tot_loss_emb, tot_loss_total):
    if loss_img1 is None:
        tot_loss_img1 = 0.0
    else:
        tot_loss_img1 += loss_img1.item()
    if loss_img2 is None:
        tot_loss_img2 = 0.0
    else:
        tot_loss_img2 += loss_img2.item()
    if loss_emb is None:
        tot_loss_emb = 0.0
    else:
        tot_loss_emb += loss_emb.item()
    if loss_total is None:
        tot_loss_total = 0.0
    else:
        tot_loss_total += loss_total.item()
    return tot_loss_img1, tot_loss_img2, tot_loss_emb, tot_loss_total


def save_inp_rec_images(config, epoch, orig_img, img_list, rec_list):
    orig_img = orig_img.view(orig_img.shape).cpu().data
    save_image(orig_img, '{}epoch{:03d}_orig.png'.format(config['save_path'], epoch+1))
    for i, img in enumerate(img_list):
        img = img.view(orig_img.shape).cpu().data
        rec = rec_list[i].view(orig_img.shape).cpu().data
        save_image(img, '{}epoch{:03d}_inp{}.png'.format(config['save_path'], epoch+1, i + 1))
        save_image(rec, '{}epoch{:03d}_rec{}.png'.format(config['save_path'], epoch+1, i + 1))
    config['save_image_flag'] = False

