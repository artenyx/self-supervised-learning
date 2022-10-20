
#SET-UP

## IMPORTS / MOUNTING DRIVE


import torch, math, copy
from torchvision import datasets, transforms, models
from torchsummary import summary
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import torch.optim as optim
from torchvision.utils import save_image
import torchvision.transforms as T
import time
from sklearn.model_selection import train_test_split
from collections import OrderedDict
import numpy as np
from PIL import Image
import pandas as pd
import os

#Mounting Google Drive

def make_dir(Name = 'AE_images'):
    if not os.path.exists(Name):
      os.makedirs(Name)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

## SETTING MODEL CONFIGURATION:"""

def get_model_config(s = 0.25):
  """ Retrieve configuration for the model. """

  model_config = {
    "width": 32,
    "height": 32,
    "channels": 3,
    "latent_dim": 64, #size of AE bottleneck
    "batch_size": 512,
    "criterion_class": nn.CrossEntropyLoss,
    "criterion_emb_recon": nn.MSELoss,
    "optimizer_type": torch.optim.Adam,
    "optimizer":None,
    "num_epochs_ssl": 200,
    "num_epochs_lineval":150,
    "train_err_thresh":0.00,
    "layer_dim": 512, # not used
    "num_classes": 10,
    "lr_ssl":0.001, #verified best lr for doc 5
    "lr_lineval":0.01, #verified best lr for doc 5
    "directory":"AE_img_",
    "linear":False,
    "transform":T.Compose([T.RandomCrop(24),
                           T.Resize(32),
                           T.RandomHorizontalFlip(p = 0.8),
                           T.ColorJitter(brightness = 0.8*s,contrast = 0.8*s,saturation = 0.8*s, hue = 0.2*s)]),
    "transform_reduced":T.Compose([T.RandomCrop(30),
                                   T.Resize(32),
                                   T.RandomHorizontalFlip(p = 0.4),
                                   T.ColorJitter()]), #for classification reasons only
    "transform_dataloader":T.Compose([T.ToTensor()]),
    "printloss_rate":25,  #number of epochs in between printing loss/error and saving images (if USL)
    "data_path":'/content/drive/My Drive/UChicago Documents/Thesis/Data/08_',
    "model_path":'/content/drive/My Drive/UChicago Documents/Thesis/Models/'
  }
  return model_config

"""## GETTING DATASETS/LOADERS:"""

def get_CIFAR10(config,dataset_size = 50000,CIFAR10_train = None,CIFAR10_test = None):
  transform_CIFAR = config['transform_dataloader']
  batch_size = config['batch_size']

  if CIFAR10_train == None:
    CIFAR10_train = datasets.CIFAR10(root = "data",train = True,download = True, transform = transform_CIFAR)
    CIFAR10_test = datasets.CIFAR10(root = "data",train = False,download = True, transform = transform_CIFAR)

  CIFAR10_train_loader = torch.utils.data.DataLoader(CIFAR10_train,batch_size = batch_size,shuffle = True,num_workers = 1)
  CIFAR10_test_loader = torch.utils.data.DataLoader(CIFAR10_test,batch_size = batch_size,shuffle = True,num_workers = 1)
  return CIFAR10_train_loader,CIFAR10_test_loader

def get_CIFAR100(config,dataset_size = 50000,CIFAR100_train = None,CIFAR100_test = None):
  transform_CIFAR = config['transform_dataloader']
  batch_size = config['batch_size']

  if CIFAR100_train == None:
    CIFAR100_train = datasets.CIFAR100(root = "data",train = True,download = True, transform = transform_CIFAR)
    CIFAR100_test = datasets.CIFAR100(root = "data",train = False,download = True, transform = transform_CIFAR)

  CIFAR10_train_loader = torch.utils.data.DataLoader(CIFAR100_train,batch_size = batch_size,shuffle = True,num_workers = 1)
  CIFAR10_test_loader = torch.utils.data.DataLoader(CIFAR100_test,batch_size = batch_size,shuffle = True,num_workers = 1)
  return CIFAR10_train_loader,CIFAR10_test_loader

"""## SSL PROCEDURES:

### Autoencoder Representation Learning:
"""

def AE_train_epoch(epoch,model,config,train_loader,train_directory,denoising,alpha):

  criterion_recon = config['criterion_emb_recon']()
  criterion_emb = config['criterion_emb_recon']()
  optimizer = config['optimizer']
  transform = config['transform']
  latent_dim = config['latent_dim']
  printloss_rate = config['printloss_rate']

  loss_img1,loss_img2,loss_emb,loss_total = np.zeros(4)
  first = True
  for img,target in train_loader:
    if torch.cuda.is_available():
      img = img.cuda()
    img0 = img
    img1 = transforms.Lambda(lambda img: torch.stack([transform(img_) for img_ in img]))(img)
    img2 = img #transforms.Lambda(lambda img: torch.stack([transform(img_) for img_ in img]))(img)


    optimizer.zero_grad()
    emb1,output1 = model(img1)
    emb2,output2 = model(img2)
    if denoising:
      batch_loss_img1 = criterion_recon(img0,output1)
      batch_loss_img2 = torch.zeros(1).cuda() #criterion_recon(img0,output2)
    else:
      batch_loss_img1 = criterion_recon(img1,output1)
      batch_loss_img2 = criterion_recon(img2,output2)
    batch_loss_emb = criterion_emb(emb1,emb2)
    batch_loss_total = torch.sum(batch_loss_img1 + alpha * batch_loss_emb/latent_dim + batch_loss_img2)
    batch_loss_total.backward()
    optimizer.step()

    loss_img1 += batch_loss_img1.item()
    loss_img2 += batch_loss_img2.item()
    loss_emb += batch_loss_emb.item()
    loss_total += batch_loss_total.item()

    if first and (epoch == 0 or (epoch+1) % printloss_rate == 0):
      img0 = img0.view(img.shape).cpu().data
      img1 = img1.view(img.shape).cpu().data
      img2 = img2.view(img.shape).cpu().data
      output1 = output1.view(img.shape).cpu().data
      output2 = output2.view(img.shape).cpu().data
      save_image(img0, '{}/epoch{:03d}_inp0.png'.format(train_directory,epoch+1))
      save_image(img1, '{}/epoch{:03d}_inp1.png'.format(train_directory,epoch+1))
      save_image(img2, '{}/epoch{:03d}_inp2.png'.format(train_directory,epoch+1))
      save_image(output1, '{}/epoch{:03d}_rec1.png'.format(train_directory,epoch+1))
      save_image(output2, '{}/epoch{:03d}_rec2.png'.format(train_directory,epoch+1))
    first = False
  loss_img1 = loss_img1 / len(train_loader)
  loss_img2 = loss_img2 / len(train_loader)
  loss_emb = loss_emb / len(train_loader)
  loss_total = loss_total / len(train_loader)
  return loss_img1,loss_img2,loss_emb,loss_total

def AE_test_epoch(epoch,model,config,test_loader,test_directory,denoising,alpha):

  criterion_recon = config['criterion_emb_recon']()
  criterion_emb = config['criterion_emb_recon']()
  linear = config['linear']
  transform = config['transform']
  latent_dim = config['latent_dim']

  loss_img1,loss_img2,loss_emb,loss_total = np.zeros(4)
  first = True
  model.eval()
  with torch.no_grad():
    for img,target in test_loader:
      if torch.cuda.is_available():
        img = img.cuda()
      img0 = img
      img1 = transforms.Lambda(lambda img: torch.stack([transform(img_) for img_ in img]))(img)
      img2 = img #transforms.Lambda(lambda img: torch.stack([transform(img_) for img_ in img]))(img)


      emb1,output1 = model(img1)
      emb2,output2 = model(img2)
      if denoising:
        batch_loss_img1 = criterion_recon(img0,output1)
        batch_loss_img2 = torch.zeros(1).cuda() #criterion_recon(img0,output2)
      else:
        batch_loss_img1 = criterion_recon(img1,output1)
        batch_loss_img2 = criterion_recon(img2,output2)
      batch_loss_emb = criterion_emb(emb1,emb2)
      batch_loss_total = torch.sum(batch_loss_img1+alpha * batch_loss_emb/latent_dim + batch_loss_img2)

      loss_img1 += batch_loss_img1.item()
      loss_img2 += batch_loss_img2.item()
      loss_emb += batch_loss_emb.item()
      loss_total += batch_loss_total.item()

      if first and (epoch == 0 or (epoch+1) % 20 == 0):
        img0 = img0.view(img.shape).cpu().data
        img1 = img1.view(img.shape).cpu().data
        img2 = img2.view(img.shape).cpu().data
        output1 = output1.view(img.shape).cpu().data
        output2 = output2.view(img.shape).cpu().data
        save_image(img0, '{}/epoch{:03d}_inp0.png'.format(test_directory,epoch+1))
        save_image(img1, '{}/epoch{:03d}_inp1.png'.format(test_directory,epoch+1))
        save_image(img2, '{}/epoch{:03d}_inp2.png'.format(test_directory,epoch+1))
        save_image(output1, '{}/epoch{:03d}_rec1.png'.format(test_directory,epoch+1))
        save_image(output2, '{}/epoch{:03d}_rec2.png'.format(test_directory,epoch+1))
      first = False

  loss_img1 = loss_img1 / len(test_loader)
  loss_img2 = loss_img2 / len(test_loader)
  loss_emb = loss_emb / len(test_loader)
  loss_total = loss_total / len(test_loader)
  return loss_img1,loss_img2,loss_emb,loss_total

def AE_pretrain(model,config,denoising,alpha):

  config['optimizer']=config['optimizer_type'](model.parameters(),lr = config['lr_ssl'])
  train_loader,test_loader = get_CIFAR100(config)
  directory = config['directory']
  epochs = config['num_epochs_ssl']
  printloss_rate = config['printloss_rate']
  train_directory,test_directory = directory + '_train',directory + '_test'

  make_dir(train_directory)
  make_dir(test_directory)

  train_list_img1,train_list_img2,train_list_emb,train_list_tot,test_list_img1,test_list_img2,test_list_emb,test_list_tot,epoch_list,time_list = [],[],[],[],[],[],[],[],[],[]
  for epoch in range(epochs):
    t1 = time.time()
    train_loss_img1,train_loss_img2,train_loss_emb,train_loss_tot = AE_train_epoch(epoch,model,config,train_loader,train_directory,denoising,alpha)
    epoch_list.append(epoch)
    train_list_img1.append(train_loss_img1)
    train_list_img2.append(train_loss_img2)
    train_list_emb.append(train_loss_emb)
    train_list_tot.append(train_loss_tot)
    test_loss_img1,test_loss_img2,test_loss_emb,test_loss_tot = AE_test_epoch(epoch,model,config,test_loader,test_directory,denoising,alpha)
    test_list_img1.append(test_loss_img1)
    test_list_img2.append(test_loss_img2)
    test_list_emb.append(test_loss_emb)
    test_list_tot.append(test_loss_tot)
    time_list.append(time.time()-t1)
    if epoch == 0 or (epoch+1) % printloss_rate == 0:
      print('Epoch {:02d}/{:02d} Time {:01f}'.format(epoch+1, epochs,time.time()-t1))
      print('Image 1 Train: {:02f} || Test: {:02f}'.format(train_loss_img1, test_loss_img1))
      print('Image 2 Train: {:02f} || Test: {:02f}'.format(train_loss_img2, test_loss_img2))
      print('Embedding Train: {:02f} || Test: {:02f}'.format(train_loss_emb, test_loss_emb))
      print('Total Train: {:02f} || Test: {:02f}'.format(train_loss_tot, test_loss_tot))

  data = {'Epoch':epoch_list,'Alpha Value':np.repeat(alpha,len(epoch_list)),'Time per Epoch':time_list,'Train Loss - Image 1':train_list_img1,'Test Loss - Image 1':test_list_img1,'Train Loss - Image 2':train_list_img2,'Test Loss - Image 2':test_list_img2,'Embedding Loss - Train':train_list_emb,'Embedding Loss - Test':test_list_emb,'Total Loss - Train':train_list_tot,'Total Loss - Test':test_list_tot}
  data = pd.DataFrame(data)
  return data, model

"""### SimCLR Representation Learning:"""

def Standardize(input):
  #Vectorize each example
  input = input.reshape(input.shape[0],-1)
  sd = torch.sqrt(torch.sum(input * input,dim = 1)).reshape(-1,1)
  input = input/(sd + 0.001)
  return input

def SimCLR_Loss(batch_repr1,batch_repr2):
  assert batch_repr1.shape == batch_repr2.shape
  batch_size = batch_repr1.shape[0]

  # Standardize 64 dim outputs of original and deformed images
  batch_repr1_stand = Standardize(batch_repr1)
  batch_repr2_stand = Standardize(batch_repr2)
  # Compute 3 covariance matrices - 0-1, 0-0, 1-1.
  COV12 = torch.mm(batch_repr1_stand,batch_repr2_stand.transpose(0,1)) #COV
  COV11 = torch.mm(batch_repr1_stand,batch_repr1_stand.transpose(0,1)) #COV0
  COV22 = torch.mm(batch_repr2_stand,batch_repr2_stand.transpose(0,1)) #COV1
  # Diagonals of covariances.
  d12 = torch.diag(COV12) #v
  d11 = torch.diag(COV11) #v0
  d22 = torch.diag(COV22) #v1
  # Mulitnomial logistic loss just computed on positive match examples, with all other examples as a separate class.
  lecov = torch.log(torch.exp(torch.logsumexp(COV12,dim=1)) + torch.exp(torch.logsumexp(COV11-torch.diag(d11),dim=1)))
  lecov += torch.log(torch.exp(torch.logsumexp(COV12,dim=1)) + torch.exp(torch.logsumexp(COV22-torch.diag(d22),dim=1)))
  lecov = .5*(lecov) - d12

  loss = torch.mean(lecov)

  '''
  # Accuracy
  if torch.cuda.is_available():
    ID = 2. * torch.eye(batch_size).to('cuda') - 1.
  else:
    ID = 2. *torch.eye(batch_size) - 1
  icov=ID*COV12
  acc=torch.sum((icov>0).type(torch.float))/ batch_size
  '''

  return loss

def EncProj_train_epoch(epoch,model,config,train_loader,train_directory):

  linear = model.FC_only
  transform = config['transform']
  latent_dim = config['latent_dim']
  printloss_rate = config['printloss_rate']
  optimizer=config['optimizer']

  loss_total = 0
  for img,_ in train_loader:
    img=img.to(device)
    img0 = img
    img1 = transforms.Lambda(lambda img: torch.stack([transform(img_) for img_ in img]))(img)
    img2 = img0 #transforms.Lambda(lambda img: torch.stack([transform(img_) for img_ in img]))(img)

    optimizer.zero_grad()
    output1 = model(img1)
    output2 = model(img2)
    loss = SimCLR_Loss(output1,output2)
    loss.backward()
    optimizer.step()

    loss_total += loss.item()

  loss_total = loss_total / len(train_loader)
  return loss_total

def EncProj_test_epoch(epoch,model,config,test_loader,test_directory):

  linear = config['linear']
  transform = config['transform']
  printloss_rate = config['printloss_rate']


  loss_total = 0
  model.eval()
  with torch.no_grad():
    for img,_ in test_loader:
      img = img.to(device)
      img0 = img
      img1 = transforms.Lambda(lambda img: torch.stack([transform(img_) for img_ in img]))(img)
      img2 = img0 #transforms.Lambda(lambda img: torch.stack([transform(img_) for img_ in img]))(img)


      output1 = model(img1)
      output2 = model(img2)
      loss = SimCLR_Loss(output1,output2)

      loss_total += loss.item()

  loss_total = loss_total / len(test_loader)
  return loss_total

def EncProj_get_embedding(model,test_loader):

  model.eval()
  output_all=[]
  targs=[]
  with torch.no_grad():
    for img,targ in test_loader:
      img = img.to(device)
      output = model.embedd(img)
      output_all.append(output)
      targs.append(targ)
  output_all=torch.cat(output_all,dim=0).detach().cpu()
  targs=torch.cat(targs).detach().cpu()
  return output_all,targs

def EncProj_Train(model,config):
  train_loader,test_loader = get_CIFAR100(config)
  directory = config['directory']
  epochs = config['num_epochs_ssl']
  printloss_rate = config['printloss_rate']
  config['optimizer']=config['optimizer_type'](model.parameters(),lr = config['lr_ssl'])
  train_directory = directory + '_train'
  test_directory = directory + '_test'

  make_dir(train_directory)
  make_dir(test_directory)

  train_list_tot,test_list_tot,time_list,epoch_list = [],[],[],[]
  for epoch in range(epochs):
    t1=time.time()
    train_loss_tot = EncProj_train_epoch(epoch,model,config, train_loader,train_directory)
    train_list_tot.append(train_loss_tot)
    test_loss_tot = EncProj_test_epoch(epoch,model,config, test_loader,test_directory)
    test_list_tot.append(test_loss_tot)
    time_list.append(time.time()-t1)
    epoch_list.append(epoch+1)
    if epoch == 0 or (epoch+1) % printloss_rate == 0:
      print('Epoch {:02d}/{:02d}. Time {:f}'.format(epoch+1, epochs,time.time()-t1))
      print('Total Train: {:02f} || Test: {:02f}'.format(train_loss_tot, test_loss_tot))

  data = {'Epoch':epoch_list,'Time per Epoch':time_list,'Total Loss - Train':train_list_tot,'Total Loss - Test':test_list_tot}
  data = pd.DataFrame(data)

  return data, model

"""## LINEAR EVALUATION PROCEDURE:

"""

def classif_train_epoch(epoch,model,config,train_loader):

  optimizer=config['optimizer']
  criterion = config['criterion_class']()
  linear = config['linear']
  transform = config['transform']

  total_correct,total_samples = np.zeros(2)
  for data, target in train_loader:
    data=data.to(device)
    target=target.to(device)
    if linear:
      data = data.reshape(-1,data.shape[1]*data.shape[2]*data.shape[3])
    optimizer.zero_grad()
    output = model(data)
    index = torch.argmax(output,1)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()
    total_correct += (index == target).float().sum()
    total_samples += data.shape[0]
    train_err = 1 - total_correct/total_samples
  return train_err.item(), loss.item()

def classif_test_epoch(epoch,model,config, test_loader,transform_test_imgs):

  criterion = config['criterion_class']()
  linear = config['linear']
  transform = config['transform_reduced']

  total_correct,total_samples = np.zeros(2)
  model.eval()
  with torch.no_grad():
    for data, target in test_loader:
      data=data.to(device)
      target=target.to(device)
      if transform_test_imgs:
        data = transforms.Lambda(lambda img: torch.stack([transform(img_) for img_ in img]))(data)
      if linear:
        data = data.reshape(-1,data.shape[1]*data.shape[2]*data.shape[3])
      output = model(data)
      index = torch.argmax(output,1)
      total_correct += (index == target).float().sum()
      total_samples += data.shape[0]
    test_err = 1 - total_correct/total_samples
  return test_err.item()

def Classifier_Training(model,config,train_loader,test_loader,alpha = None):

  epochs = config['num_epochs_lineval']
  printloss_rate = config['printloss_rate']
  config['optimizer'] = config['optimizer_type'](model.parameters(),lr=config['lr_lineval'])
  train_err_thresh = config['train_err_thresh']
  train_err_list,train_loss_list,test_err_trans_list,test_err_notrans_list,epoch_list,time_list,alpha_list = [],[],[],[],[],[],[]
  epoch,train_err = 0,1.0

  while (epoch < epochs) and (train_err > train_err_thresh):
    t1 = time.time()
    train_err, train_loss = classif_train_epoch(epoch,model,config,train_loader)
    test_err_trans = 0 #classif_test_epoch(epoch,model,config,test_loader,True)
    test_err_notrans = classif_test_epoch(epoch,model,config,test_loader,False)

    train_err_list.append(train_err)
    train_loss_list.append(train_loss)
    test_err_trans_list.append(test_err_trans)
    test_err_notrans_list.append(test_err_notrans)
    epoch_list.append(epoch+1)
    time_list.append(time.time() - t1)
    alpha_list.append(alpha)

    if epoch == 0 or (epoch+1) % printloss_rate == 0:
      print('Epoch {:03d}/{:03d} Time {:.1f}\nTrain Error {:.2f}% || Train loss {:.5f} || Test Error w/ Trans {:.2f}% || Test Error w/o Trans {:.2f}%'.format(epoch+1, epochs, time.time()-t1, train_err*100, train_loss, test_err_trans*100, test_err_notrans*100))
    epoch += 1

  data = {'Epoch':epoch_list,'Alpha Value':alpha_list,'Time per Epoch':time_list,'Train Err':train_err_list,'Train Loss':train_loss_list,'Test Err with Transformations':test_err_trans_list,'Test Err without Transformations':test_err_notrans_list}
  data = pd.DataFrame(data)
  return data

"""## MODEL CLASSES:"""

class PrintLayer(nn.Module):
    def __init__(self):
        super(PrintLayer, self).__init__()

    def forward(self, x):
        # Do your print / debug stuff here
        print(x.shape)
        return x

class Linear_class(nn.Module):
  def __init__(self,indim,num_classes):
    super(Linear_class,self).__init__()

    self.layers=nn.Sequential(*[nn.Linear(indim,num_classes),nn.Dropout(0.05)])

  def forward(self, input):

    output=self.layers(input)

    return output


class Conv6_LC(nn.Module):
  def __init__(self,ssl_type):
    super(Conv6_LC,self).__init__()
    #Setting model parameters
    self.ssl_type = ssl_type
    self.representation_dim = 8192
    self.FC_only = False
    kernel_enc = 3
    kernel_dec = 2

    #Importing model configuration parameters
    config = get_model_config()
    input_shape = config.get("width") * config.get("height") * config.get("channels")
    latent_dim = config.get("latent_dim")
    input_channels = config.get("channels")
    enclayers,declayers = [],[]

    #Creating ordered dictionary of layers to organize structure
    enclayers = [(str(len(enclayers)),nn.Conv2d(input_channels,32,kernel_size = kernel_enc,stride = 1,padding = 1))]
    enclayers.append((str(len(enclayers)),nn.Hardtanh()))
    enclayers.append((str(len(enclayers)),nn.Conv2d(32,32,kernel_size = kernel_enc,stride = 1,padding = 1)))
    enclayers.append((str(len(enclayers)),nn.MaxPool2d(2,2)))
    enclayers.append((str(len(enclayers)),nn.Conv2d(32,64,kernel_size = kernel_enc,stride = 1,padding = 1)))
    enclayers.append((str(len(enclayers)),nn.Hardtanh()))
    enclayers.append((str(len(enclayers)),nn.Conv2d(64,64,kernel_size = kernel_enc,stride = 1,padding = 1)))
    enclayers.append((str(len(enclayers)),nn.MaxPool2d(2,2)))
    enclayers.append((str(len(enclayers)),nn.Conv2d(64,512,kernel_size = kernel_enc,stride = 1,padding = 1)))
    enclayers.append((str(len(enclayers)),nn.MaxPool2d(2,2)))
    enclayers.append((str(len(enclayers)),nn.Flatten()))
    self.embedd=nn.Sequential(OrderedDict(enclayers))
    #enclayers.append((str(len(enclayers)),nn.Linear(self.representation_dim,latent_dim)))

    #declayers = [(str(len(enclayers)+len(declayers)),nn.Linear(latent_dim,self.representation_dim))]
    declayers.append((str(len(enclayers)+len(declayers)),nn.Unflatten(1,(512,4,4))))
    declayers.append((str(len(enclayers)+len(declayers)),nn.ConvTranspose2d(512,32,kernel_size = kernel_dec,stride = 2,padding = 0)))
    declayers.append((str(len(enclayers)+len(declayers)),nn.Hardtanh()))
    declayers.append((str(len(enclayers)+len(declayers)),nn.ConvTranspose2d(32,32,kernel_size = kernel_dec,stride = 2,padding = 0)))
    declayers.append((str(len(enclayers)+len(declayers)),nn.Hardtanh()))
    declayers.append((str(len(enclayers)+len(declayers)),nn.ConvTranspose2d(32,input_channels,kernel_size = kernel_dec,stride = 2,padding = 0)))
    declayers.append((str(len(enclayers)+len(declayers)),nn.Sigmoid()))
    layers = enclayers + declayers

    self.encode = nn.Sequential(OrderedDict(enclayers))
    self.decode = nn.Sequential(OrderedDict(declayers))
    self.encoder_nlayers = len(enclayers)
    self.decoder_nlayers = len(declayers)

    #For grad_status function
    if self.ssl_type == "autoencoder":
      self.layers = nn.Sequential(OrderedDict(enclayers + declayers))
    elif self.ssl_type == "simclr":
      self.layers = nn.Sequential(OrderedDict(enclayers))
    for m in self.layers:
      if isinstance(m, nn.Conv2d) or isinstance(m,nn.ConvTranspose2d) or isinstance(m,nn.Linear):
        nn.init.xavier_normal_(m.weight)
        nn.init.zeros_(m.bias)

  def forward(self,input):
      if self.ssl_type == "autoencoder":
        encoded = self.encode(input)
        decoded = self.decode(encoded)
        return encoded,decoded
      elif self.ssl_type == "simclr":
        encoded = self.encode(input)
        return encoded

def save_model_data(model,data,model_path,data_path,config):
  with open(data_path, 'w', encoding = 'utf-8-sig') as f:
    data.to_csv(f)
  torch.save({'config': config,'model.state.dict': model.state_dict()}, model_path)

def load_model(path,mtype):
  AA=torch.load(path)
  config=AA['config']
  model = Conv6_LC(ssl_type = mtype)
  model.load_state_dict(AA['model.state.dict'])

  return model,config

def get_embedding(model_path,model_type,model = None,config = None):
  if model == None:
    model, config=load_model(model_path,model_type)
  model = model.cuda()
  train_loader,test_loader = get_CIFAR10(config)
  out_test, targ_test=EncProj_get_embedding(model,test_loader)
  out_train, targ_train=EncProj_get_embedding(model,train_loader)
  test_loader= torch.utils.data.DataLoader(list(zip(out_test,targ_test)),batch_size = config['batch_size'],shuffle = False,num_workers = 1)
  train_loader= torch.utils.data.DataLoader(list(zip(out_train,targ_train)),batch_size = config['batch_size'],shuffle = True,num_workers = 1)
  return train_loader, test_loader

"""##EXPERIMENT FUNCTIONS:"""

def RunExp(ssl_type,config,prefix_before_name = '',denoising = False,alphas = [0.0],save = True):
  data_seed = config['data_path'] + prefix_before_name
  model_seed = config['model_path'] + prefix_before_name
  data_ssl = pd.DataFrame()
  data_classif = pd.DataFrame()
  ssl_epochs = config['num_epochs_ssl']


  if not ssl_type == 'autoencoder' and len(alphas) > 1:
    print('SimCLR SSL selected, alpha list will not be used.')
    alphas = [0.0]

  for alpha in alphas:
    ###Representation Learning
    print('\n----Representation Learning----')
    if ssl_type == 'autoencoder':
      SSLNet = Conv6_LC(ssl_type = ssl_type).cuda()
      config['num_epochs_ssl'] = ssl_epochs
      if denoising:
        print('Running denoising autoencoder SSL experiment with alpha = ' + str(alpha))
        current_data_path = data_seed + 'DenoisingAE_alpha'+str(alpha)+'_epochs'+str(config['num_epochs_ssl'])
        current_model_path = model_seed + 'DenoisingAE_alpha'+str(alpha)+'_epochs'+str(config['num_epochs_ssl'])+'.pt'
        data_ssl, SSLNet = AE_pretrain(SSLNet,config,denoising = True,alpha = alpha)
      else:
        print('Running autoencoder SSL experiment with alpha = ' + str(alpha))
        current_data_path = data_seed + 'AE_alpha'+str(alpha)+'_epochs'+str(config['num_epochs_ssl'])
        current_model_path = model_seed + 'AE_alpha'+str(alpha)+'_epochs'+str(config['num_epochs_ssl'])+'.pt'
        data_ssl, SSLNet = AE_pretrain(SSLNet,config,denoising = False,alpha = alpha)
    elif ssl_type == 'simclr':
      print('Running SimCLR SSL experiment.')
      SSLNet = Conv6_LC(ssl_type = ssl_type).cuda()
      config['num_epochs_ssl'] = ssl_epochs
      current_data_path = data_seed + 'SimCLR_epochs'+str(config['num_epochs_ssl'])
      current_model_path = model_seed + 'SimCLR_epochs'+str(config['num_epochs_ssl'])+'.pt'
      data_ssl, SSLNet = EncProj_Train(SSLNet,config)
    elif ssl_type == 'randclass':
      print('Running classifier on random embeddings.')
      ssl_type = 'simclr'
      SSLNet = Conv6_LC(ssl_type = ssl_type).cuda() #choice of simclr is arbitrary
      config['num_epochs_ssl'] = 0
      current_data_path = data_seed + 'Rand'
      current_model_path = model_seed + 'Rand'+'.pt'
      data_ssl, SSLNet = EncProj_Train(SSLNet,config)
    else:
      raise Exception('Not a supported SSL type. Supported strings are simclr and autoencoder. Use autoencoder ssl_type with denoising = True for a denoising autoencoder.')

    ###Saving Model
    if save:
      torch.save({'config': config,'model.state.dict': SSLNet.state_dict()}, current_model_path)
      CIFAR10_emb_train_loader, CIFAR10_emb_test_loader= get_embedding(current_model_path,ssl_type,model = SSLNet,config = config)
    else:
      CIFAR10_emb_train_loader, CIFAR10_emb_test_loader= get_embedding(current_model_path,ssl_type,model = SSLNet,config = config)

    LinearEval_Model = Linear_class(SSLNet.representation_dim,config['num_classes']).cuda()
    print(LinearEval_Model)

    ###Running Linear Evaluation Stage
    print('----Linear Evaluation----')
    data_classif = Classifier_Training(LinearEval_Model, config, CIFAR10_emb_train_loader, CIFAR10_emb_test_loader, alpha = alpha)

    ###Saving data and config
    if save:
      with open(current_data_path + '_reptr.csv', 'w', encoding = 'utf-8-sig') as f:
        data_ssl.to_csv(f)
      with open(current_data_path + '_lineval.csv', 'w', encoding = 'utf-8-sig') as f:
        data_classif.to_csv(f)
      config_df = pd.DataFrame(config,index = [0])
      with open(current_data_path + '_config.csv', 'w', encoding = 'utf-8-sig') as f:
        config_df.to_csv(f)

  print('Experiment completed.')

def RunAllExps(config,alphas = [0.0],save = True):
  RunExp('autoencoder',config,denoising = False,alphas = alphas,save = save)
  RunExp('autoencoder',config,denoising = True,alphas = alphas,save = save)
  RunExp('simclr',config,denoising = False,alphas = [0.0],save = save)

"""# EXPERIMENTS:

###Single Experiment:
"""

strength_list = [0.0,0.25,0.5,1]

for s in strength_list:
  config = get_model_config(s = s)
  config['lr_ssl'] = 0.001 #0.001 best for den AE, 0.0001 best for simclr (and vanilla ae?)
  config['lr_lineval'] = 0.001

  config['num_epochs_ssl'] = 200
  config['num_epochs_lineval'] = 100

  config['printloss_rate'] = 50

  ssl_types = ['simclr','autoencoder']

  alphas = [0.0]

  #Run Experiment
  config['criterion_emb_recon'] = nn.MSELoss
  RunExp('autoencoder',config,'_colorjitstrength_'+str(s),denoising = True,alphas = alphas,save = True)

  config['lr_ssl'] = 0.0001
  RunExp('simclr',config,'_colorjitstrength_'+str(s),denoising = False,alphas = alphas,save = True)

"""###All Experiments:"""

#Experiment Settings

config = get_model_config()
config['lr_ssl'] = 0.001
config['lr_lineval'] = 0.001

config['num_epochs_ssl'] = 300
config['num_epochs_lineval'] = 300

config['printloss_rate'] = 50

alphas = [0.0,0.001,0.01,0.1,1.0]

#Run Experiment
RunAllExps(config,alphas = alphas,save = True)

"""###SimCLR Learning Rate Testing:"""

#best simclr lr = 0.001

#Experiment Settings
torch.manual_seed(123)
torch.cuda.manual_seed(123)

config = get_model_config()
config['lr_lineval'] = 0.001

config['num_epochs_ssl'] = 50
config['num_epochs_lineval'] = 100

config['printloss_rate'] = 20
lrs = [0.0001,0.001,0.01,0.1]

for lr in lrs:
  config['lr_ssl'] = lr

  ssl_types = ['simclr','autoencoder']
  ssl_type = ssl_types[0]

  alphas = [0.0,0.001,0.01,0.1,1.0]

  #Run Experiment
  RunExp(ssl_type,config,denoising = True,alphas = alphas,save = False)

"""###AE Learning Rate Testing:"""

#best AE lr = 0.001 6/25 denoising L2 emb/ recon loss
#best AE lr = 0.001 6/25 denoising L1 emb/ recon loss

#Experiment Settings
torch.manual_seed(123)
torch.cuda.manual_seed(123)

config = get_model_config()
config['lr_lineval'] = 0.001

config['num_epochs_ssl'] = 50
config['num_epochs_lineval'] = 100
config['criterion_emb_recon'] = nn.L1Loss


config['printloss_rate'] = 20
lrs = [0.0001,0.001,0.01,0.1]

for lr in lrs:
  config['lr_ssl'] = lr
  ssl_types = ['simclr','autoencoder']
  ssl_type = ssl_types[1]

  alphas = [0.0]

  #Run Experiment
  RunExp(ssl_type,config,denoising = True,alphas = alphas,save = False)

#best AE lr = 0.001 6/25

#Experiment Settings
torch.manual_seed(123)
torch.cuda.manual_seed(123)

config = get_model_config()
config['lr_lineval'] = 0.001

config['num_epochs_ssl'] = 50
config['num_epochs_lineval'] = 100

config['printloss_rate'] = 20
lrs = [0.00001,0.0001]

for lr in lrs:
  config['lr_ssl'] = lr
  ssl_types = ['simclr','autoencoder']
  ssl_type = ssl_types[1]

  alphas = [0.0]

  #Run Experiment
  RunExp(ssl_type,config,denoising = False,alphas = alphas,save = False)

"""###LinEval Learning Rate Testing"""

#best lineval lr = 0.001

#Experiment Settings
torch.manual_seed(123)
torch.cuda.manual_seed(123)

config = get_model_config()
config['lr_ssl'] = 0.001

config['num_epochs_ssl'] = 5
config['num_epochs_lineval'] = 100

config['printloss_rate'] = 20
lrs = [0.0001,0.001,0.01,0.1]

for lr in lrs:
  config['lr_lineval'] = lr
  ssl_types = ['simclr','autoencoder']
  ssl_type = ssl_types[1]

  alphas = [0.0]

  #Run Experiment
  RunExp(ssl_type,config,denoising = True,alphas = alphas,save = False)

print(lrs)