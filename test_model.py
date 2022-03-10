import os
import torch
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


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dataset = LIDC_IDRI(dataset_location = '/home/nandcui/data/plidc-punet/')

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

def validation(loss_dict):
    net.eval()
    for step, (patch, mask, _) in enumerate(val_loader): 
        patch = patch.to(device)
        mask = mask.to(device)
        mask = torch.unsqueeze(mask,1)
        net.forward(patch, mask, training=True)
        elbo = net.elbo(mask)
        loss_dict['val_elbo'] -= elbo.item()

    loss_dict['val_elbo'] /= len(val_loader)
    return loss_dict

def test(savefig=False):
    net.eval()
    test_loss = 0
    for step, (patch, mask, _) in enumerate(test_loader): 
        patch = patch.to(device)
        mask = mask.to(device)
        mask = torch.unsqueeze(mask,1)
        net.forward(patch, mask, training=True)
        elbo = net.elbo(mask)
        test_loss -= elbo.item()

        print(mask.shape)
    test_loss /= len(test_loader)
    print(TAG + 'Testing elbo: ', test_loss)
    return test_loss

path = 'checkpoint/LIDC_IDRI/beta-10.0_regw-1e-05_wd-0_lr-0.0001_seed-1/summary.csv'

results = {}
# iterate the K fold 
for i in range(K):
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

    net.load_stat_dict(torch.load('path'))

    te_loss = test(savefig=True)
