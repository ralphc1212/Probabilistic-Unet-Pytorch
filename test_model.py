import os
import torch
import torchvision
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler, SequentialSampler
from load_LIDC_data import LIDC_IDRI
# from probabilistic_unet import ProbabilisticUnet
from vnd_unet import VNDUnet
from utils import l2_regularisation
import csv
import time
from metric import get_energy_distance_components, calc_energy_distances

DATA = 'LIDC_IDRI'

num_filters = [32,64,128,192]
latent_dim = 6
no_convs_fcomb = 4
beta = 10.
nsamples = 4

FIX_LEN = 2

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

    patches = []
    masks = []
    recons = []
    for step, (patch, mask, _) in enumerate(dataloader): 

        patch = patch.to(device)
        mask = mask.to(device)

        mask = mask.permute(1, 0, 2, 3)
        mask = mask.reshape(-1, *mask.shape[2:])
        mask = torch.unsqueeze(mask,1)
        net.forward(patch, mask, training=False)

        recon = []

        if savefig:
            for fix_len_ in range(4):
                for i in range(nsamples):
                    recon.append(net.sample(testing=True, fix_len=fix_len_))
        else:
            for i in range(nsamples):
                recon.append(net.sample(testing=True, fix_len=FIX_LEN))

        recon = torch.cat(recon)

        recon = (torch.sigmoid(recon) >= 0.5).float()

        if savefig:
            torchvision.utils.save_image(patch, 
                            image_path + str(fold) + '/' + str(step) + '_patch' + '.png',
                            normalize=True,
                            nrow=32)
            torchvision.utils.save_image(mask, 
                            image_path + str(fold) + '/' + str(step) + '_mask' + '.png',
                            normalize=True,
                            nrow=32)    
            torchvision.utils.save_image((torch.sigmoid(recon) >= 0.5).float(), 
                            image_path + str(fold) + '/' + str(step) + '_recons' + '.png',
                            normalize=True,
                            nrow=32)

        patches.append(patch.detach())
        masks.append(mask.detach())
        recons.append(recon.detach())

    patches = torch.cat(patches)
    masks = torch.cat(masks)
    recons = torch.cat(recons)

        # elbo = net.elbo(mask, hard=hard)
        # test_loss -= elbo.item()

    # test_loss /= len(test_loader)
    # print(TAG + 'Testing elbo: ', test_loss)
    return patches.cpu().numpy(), masks.cpu().numpy(), recons.cpu().numpy()

path = 'checkpoint/LIDC_IDRI/beta-10.0_regw-1e-05_wd-1e-06_lr-0.0001_seed-1_hard-1/'
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

    val_sampler = SequentialSampler(val_indices)
    test_sampler = SequentialSampler(test_indices)

    val_loader = DataLoader(dataset, batch_size=32, sampler=val_sampler)
    test_loader = DataLoader(dataset, batch_size=32, sampler=test_sampler)

    # init probabilistic UNET or its variants
    print(TAG + 'num_filters: {}, latent dimension: {}, Fcomb layer number: {}, KLD weight: {}.'.format(
                num_filters, latent_dim, no_convs_fcomb, beta))

    net = VNDUnet(input_channels=1, num_classes=1, num_filters=num_filters, latent_dim=latent_dim, no_convs_fcomb=no_convs_fcomb, beta=beta)
    net.to(device)

    net.load_state_dict(torch.load(path + str(i) + '.pth').state_dict())

    patches, masks, recons = test(fold=i, dataloader=test_loader, savefig=False)

    total_dist_dict = {'YS': [], 'SS': [], 'YY': []}
    for img_idx in range(patches.shape[0]):
        dist_dict = get_energy_distance_components(masks[int(img_idx * 4) : int((img_idx + 1) * 4)], 
            recons[int(img_idx * 4) : int((img_idx + 1) * 4)], 2)
        total_dist_dict['YS'].append(np.expand_dims(dist_dict['YS'], 0))
        total_dist_dict['SS'].append(np.expand_dims(dist_dict['SS'], 0))
        total_dist_dict['YY'].append(np.expand_dims(dist_dict['YY'], 0))

    total_dist_dict['YS'] = np.concatenate(total_dist_dict['YS'])
    total_dist_dict['SS'] = np.concatenate(total_dist_dict['SS'])
    total_dist_dict['YY'] = np.concatenate(total_dist_dict['YY'])

    dist = calc_energy_distances(total_dist_dict)
    print(dist.mean())
    # print(dist_dict)

