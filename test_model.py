import os
import torch
import torchvision
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from load_LIDC_data import LIDC_IDRI
# from probabilistic_unet import ProbabilisticUnet
from vnd_unet import VNDUnet
from utils import l2_regularisation
import csv
import time

DATA = 'LIDC_IDRI'

num_filters = [32,64,128,192]
latent_dim = 6
no_convs_fcomb = 4
beta = 10.
nsamples = 4

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dataset = LIDC_IDRI(dataset_location = '/home/nandcui/data/plidc-punet/', plot=True)

# num of cross validation
K = 3
# retrieve the indices
splits = []
for i in range(K):

    split = []
    with open('splits/split_{}.csv'.format(i), 'r') as f:
        reader = csv.reader(f)
        for i, line in enumerate(reader):
            line = [int(elem) for elem in line]
            split.append(line)
    splits.append(split)

def test(fold=0, dataloader=None, savefig=False):
    net.eval()
    test_loss = 0
    for step, (patch, mask, _) in enumerate(dataloader): 

        patch = patch.to(device)
        mask = mask.to(device)

        mask = mask.permute(1, 0, 2, 3)
        mask = mask.reshape(-1, *mask.shape[2:])
        mask = torch.unsqueeze(mask,1)
        net.forward(patch, mask, training=False)
        recons = []

        for fix_len_ in range(4):
            for i in range(nsamples):
                recons.append(net.sample(testing=True, fix_len=fix_len_+1))

        recons = torch.cat(recons)

        torchvision.utils.save_image(patch, 
                        image_path + str(fold) + '/' + 'patch_' + str(step) + '.png',
                        normalize=True,
                        nrow=32)
        torchvision.utils.save_image(mask, 
                        image_path + str(fold) + '/' + 'mask_' + str(step) + '.png',
                        normalize=True,
                        nrow=32)    
        torchvision.utils.save_image((torch.sigmoid(recons) >= 0.5).float(), 
                        image_path + str(fold) + '/' + 'recons_' + str(step) + '.png',
                        normalize=True,
                        nrow=32)
        exit()

        # elbo = net.elbo(mask, hard=hard)
        # test_loss -= elbo.item()

    # test_loss /= len(test_loader)
    # print(TAG + 'Testing elbo: ', test_loss)
    return test_loss

path = 'checkpoint/LIDC_IDRI/beta-10.0_regw-1e-05_wd-0_lr-0.0001_seed-1_hard-0/'
image_path = path + 'prediction_images/'
hard = False

if not os.path.isdir(image_path):
    os.makedirs(image_path)

results = {}
# iterate the K fold 
for i in range(K):
    if not os.path.isdir(image_path + str(i) + '/'):
        os.makedirs(image_path + str(i) + '/')

    # Dataloaders
    TAG = '({}-fold-{}) '.format(K, i+1) 
    train_indices, val_indices, test_indices = splits[i]

    val_sampler = SubsetRandomSampler(val_indices)
    test_sampler = SubsetRandomSampler(test_indices)

    val_loader = DataLoader(dataset, batch_size=32, sampler=val_sampler)
    test_loader = DataLoader(dataset, batch_size=32, sampler=test_sampler)

    # init probabilistic UNET or its variants
    print(TAG + 'num_filters: {}, latent dimension: {}, Fcomb layer number: {}, KLD weight: {}.'.format(
                num_filters, latent_dim, no_convs_fcomb, beta))

    net = VNDUnet(input_channels=1, num_classes=1, num_filters=num_filters, latent_dim=latent_dim, no_convs_fcomb=no_convs_fcomb, beta=beta)
    net.to(device)

    net.load_state_dict(torch.load(path + str(i) + '.pth').state_dict())

    te_loss = test(fold=i, dataloader=test_loader, savefig=True)
