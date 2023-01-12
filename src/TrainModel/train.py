import torch
import torch.nn as nn
import time
import numpy as np
import pandas as pd

from src.TrainModel import losses, networks


def print_epoch_data(usl, epochs, epoch_data_train, epoch_data_test):
    if usl:
        print('Epoch {:02d}/{:02d} Time {:01f}/{:01f}'.format(epoch_data_train[0], epochs, epoch_data_train[5], epoch_data_test[5]))
        print('Image 1 Train: {:02f} || Test: {:02f}'.format(epoch_data_train[1], epoch_data_test[1]))
        print('Image 2 Train: {:02f} || Test: {:02f}'.format(epoch_data_train[2] or 0, epoch_data_test[2] or 0))
        print('Embedding Train: {:02f} || Test: {:02f}'.format(epoch_data_train[3] or 0, epoch_data_test[3] or 0))
        print('Total Train: {:02f} || Test: {:02f}'.format(epoch_data_train[4], epoch_data_test[4]))
    else:
        print('Epoch {:02d}/{:02d} Train Time {:01f}/{:01f}'.format(epoch_data_train[0], epochs, epoch_data_train[3], epoch_data_test[3]))
        print('Train Error: {:02f} || Test Error: {:02f}'.format(epoch_data_train[1], epoch_data_test[1]))
        print('Train Loss: {:02f} || Test Loss: {:02f}'.format(epoch_data_train[2], epoch_data_test[2]))


def get_column_names(usl):
    if usl:
        column_names = ["Epoch Number", "Image 1 Train Loss", "Image 2 Train Loss",
                        "Embedding Train Loss", "Total Train Loss", "Time Elapsed Train",
                        "Epoch Number", "Image 1 Test Loss", "Image 2 Test Loss",
                        "Embedding Test Loss", "Total Test Loss", "Time Elapsed Test"]
    else:
        column_names = ["Epoch Number", "Train Error", "Train Loss", "Time Elapsed Train",
                        "Epoch Number", "Test Error", "Test Loss", "Time Elapsed Test"]
    return column_names


def usl_run_epoch(model, config, loader, epoch, grad):
    t0 = time.time()
    optimizer = config['optimizer']
    if config['scheduler'] is not None:
        scheduler = config['scheduler']

    tot_loss_img1, tot_loss_img2, tot_loss_emb, tot_loss_total = np.zeros(4)
    first = True
    for data in loader:
        if first and (epoch == 0 or (epoch + 1) % config['print_loss_rate'] == 0):
            config['save_image_flag'] = True
            first = False
        if config['usl_type'] == "ae_single" and not config['denoising']:
            img0, targ = data[0]
            img0 = img0.to(config['device'])
            img1 = img0.to(config['device'])
        elif config['usl_type'] == "ae_single" and config['denoising']:
            (img0, targ), (img1, __) = data
            img0 = img0.to(config['device'])
            img1 = img1.to(config['device'])
        else:
            (img0, targ), (img1, __), (img2, __) = data
            img0 = img0.to(config['device'])
            img1 = img1.to(config['device'])
            img2 = img2.to(config['device'])

        optimizer.zero_grad()
        if config['usl_type'] == "ae_single":
            loss_img1, loss_img2, loss_emb, loss_total = losses.ae_single_run_loss(model, config, epoch, img0, img1)
        elif config['usl_type'] == "ae_parallel":
            loss_img1, loss_img2, loss_emb, loss_total = losses.ae_parallel_run_loss(model, config, epoch, img0, img1, img2)
        elif config['usl_type'] == "simclr":
            loss_img1, loss_img2, loss_emb, loss_total = losses.simclr_run_loss(model, config, img1, img2)
        elif config['usl_type'] == "simsiam":
            loss_img1, loss_img2, loss_emb, loss_total = losses.simsiam_run_loss(model, config, img1, img2)
        if grad:
            loss_total.backward()
            optimizer.step()
            if config['scheduler'] is not None:
                scheduler.step()
        tot_loss_img1, tot_loss_img2, tot_loss_emb, tot_loss_total = losses.loss_add(loss_img1, loss_img2, loss_emb, loss_total, tot_loss_img1, tot_loss_img2, tot_loss_emb, tot_loss_total)
    avg_loss_img1 = (tot_loss_img1 or 0) / len(loader)
    avg_loss_img2 = (tot_loss_img2 or 0) / len(loader)
    avg_loss_emb = (tot_loss_emb or 0) / len(loader)
    avg_loss_total = (tot_loss_total or 0) / len(loader)
    t1 = time.time() - t0
    return epoch + 1, avg_loss_img1, avg_loss_img2, avg_loss_emb, avg_loss_total, t1


def usl_train_network(model, config):
    if config['denoising'] is None or config['device'] is None or config['usl_type'] is None:
        raise Exception("Parameters denoising, device, and usl_type must all be configured. Acceptable usl_"
                        "type settings are ae_single, ae_parallel, and simclr.")

    config['optimizer'] = config['optimizer_type'](model.parameters(), lr=config['lr_usl'])
    if config['scheduler_type'] is not None:
        config['scheduler'] = config['scheduler_type'](config['optimizer'], config['num_epochs_usl'], eta_min=0, last_epoch=-1)
    train_loader, test_loader = config['loaders']['loaders_usl']
    train_data, test_data = [np.zeros(6)], [np.zeros(6)]
    for epoch in range(config['num_epochs_usl']):
        train_data.append(usl_run_epoch(model, config, train_loader, epoch, True))
        if epoch == 0 or (epoch + 1) % config['run_test_rate_usl'] == 0:
            test_data.append(usl_run_epoch(model, config, test_loader, epoch, False))
        else:
            test_data.append(np.zeros(6))
        if epoch == 0 or (epoch + 1) % config['print_loss_rate'] == 0:
            print_epoch_data(True, config['num_epochs_usl'], train_data[-1], test_data[-1])

    data = pd.concat([pd.DataFrame(train_data), pd.DataFrame(test_data)], axis=1)
    data = data.set_axis(get_column_names(True), axis=1)
    return data, model


def classifier_run_epoch(model, config, loader, epoch, grad):
    t0 = time.time()
    optimizer = config['optimizer']
    criterion = config['criterion_class']()

    tot_correct, tot_samples, tot_loss = np.zeros(3)
    for img, target in loader:
        img = img.to(config['device'])
        target = target.to(config['device'])
        optimizer.zero_grad()
        output = model(img)
        index = torch.argmax(output, 1)
        loss = criterion(output, target)
        if grad:
            loss.backward()
            optimizer.step()
        tot_correct += (index == target).float().sum()
        tot_samples += img.shape[0]
        tot_loss += loss.item()
        err = 1 - tot_correct / tot_samples
    avg_loss = tot_loss / len(loader)
    t1 = time.time() - t0
    return epoch + 1, err.item(), avg_loss, t1


def classifier_train_network(model, config):
    if config['device'] is None:
        raise Exception("Device must be configured in exp_config.")
    config['optimizer'] = config['optimizer_type'](model.parameters(), lr=config['lr_le'])
    train_loader, test_loader = config['loaders']['loaders_le']

    train_data, test_data = [np.zeros(4)], [np.zeros(4)]
    for epoch in range(config['num_epochs_le']):
        train_data.append(classifier_run_epoch(model, config, train_loader, epoch, True))
        test_data.append(classifier_run_epoch(model, config, test_loader, epoch, False))
        if epoch == 0 or (epoch + 1) % config['print_loss_rate'] == 0:
            print_epoch_data(False, config['num_epochs_le'], train_data[-1], test_data[-1])
    data = pd.concat([pd.DataFrame(train_data), pd.DataFrame(test_data)], axis=1)
    data = data.set_axis(get_column_names(False), axis=1)
    return data, model


def get_embedding_loader(model, config, loader, return_as_list=False):
    model.eval()
    embeddings = []
    targets = []
    with torch.no_grad():
        for img, targ in loader:
            img = img.to(config['device'])
            embed = model.embed(img)
            if len(embed.shape) != 1:
                embed = nn.Flatten()(embed)
            embeddings.append(embed)
            targets.append(targ)
    embeddings = torch.cat(embeddings, dim=0).detach().cpu()
    targets = torch.cat(targets).detach().cpu()
    if not return_as_list:
        embedding_loader = torch.utils.data.DataLoader(list(zip(embeddings, targets)), batch_size=config['batch_size'],
                                                       shuffle=False, num_workers=12)
    else:
        embedding_loader = [embeddings, targets]
    return embedding_loader


def usl_train_network_layerwise(model, config):
    # Take in model that will be taken to be final model desired
    # Essentially a model creation function, trains model to given epochs at each step, then adds layers on each side to
    # create new model and pass into train_usl function

    # Change so that function just changes the grad_status to true when a layer is to be updated
    encoder_layers = model.encoder_layers + model.projector_layers
    decoder_layers = model.decoder_layers

    current_enc_layers, current_dec_layers = [], []
    total_data = pd.DataFrame()

    while len(encoder_layers) != 0:
        current_enc_layers.append(encoder_layers.pop(0))
        current_enc_layers.append(encoder_layers.pop(0))
        current_dec_layers.insert(0, decoder_layers.pop(-1))
        current_dec_layers.insert(0, decoder_layers.pop(-1))

        current_model = networks.USL_Conv6_CIFAR_LC(config, current_enc_layers, current_dec_layers)
        current_data, current_model = usl_train_network(current_model, config)
        total_data = pd.concat([total_data, current_data])
        current_enc_layers, current_dec_layers = (current_model.encoder_layers + current_model.projector_layers, current_model.decoder_layers)

        # for layer in current_enc_layers + current_dec_layers:
        #    print(layer)
        #    for param in layer.parameters():
        #        print(param.requires_grad)

        for layer in current_enc_layers + current_dec_layers:
            for param in layer.parameters():
                param.requires_grad = False

    for layer in current_enc_layers + current_dec_layers:
        for param in layer.parameters():
            param.requires_grad = True
    current_model = networks.USL_Conv6_CIFAR_LC(config, current_enc_layers[:-1], current_dec_layers, proj_layers=[current_enc_layers[-1]])

    return total_data, current_model
