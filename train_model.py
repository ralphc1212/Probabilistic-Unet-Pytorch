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

num_filters = [32,64,128,192]
latent_dim = 6
no_convs_fcomb = 4
beta = 1.
reg_weight = 1e-5
weight_decay = 0
init_lr = 1e-4
epochs = 100

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dataset = LIDC_IDRI(dataset_location = '/home/nandcui/data/plidc-punet/')
# dataset_size = len(dataset)
# indices = list(range(dataset_size))
# split = int(np.floor(0.1 * dataset_size))
# np.random.shuffle(indices)

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

def train(loss_dict):
    net.train()
    for step, (patch, mask, _) in enumerate(train_loader): 
        patch = patch.to(device)
        mask = mask.to(device)
        mask = torch.unsqueeze(mask,1)
        net.forward(patch, mask, training=True)
        elbo = net.elbo(mask)
        reg_loss = l2_regularisation(net.posterior) + l2_regularisation(net.prior) + l2_regularisation(net.fcomb.layers)
        loss = - elbo + reg_weight * reg_loss
        loss_dict['tr_elbo'] -= elbo.item()
        loss_dict['tr_loss'] += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    loss_dict['tr_elbo'] /= len(train_loader)
    loss_dict['tr_loss'] /= len(train_loader)
    return loss_dict

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

def test():
    net.eval()
    test_loss = 0
    for step, (patch, mask, _) in enumerate(test_loader): 
        patch = patch.to(device)
        mask = mask.to(device)
        mask = torch.unsqueeze(mask,1)
        net.forward(patch, mask, training=True)
        elbo = net.elbo(mask)
        test_loss -= elbo.item()

    test_loss /= len(test_loader)
    print(TAG + 'Testing elbo: ', test_loss)
    return loss_dict

# iterate the K fold 
for i in range(K):
    # Dataloaders
    TAG = '({}-fold-{}) '.format(K, i+1) 
    train_indices, val_indices, test_indices = splits[i]

    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)
    test_sampler = SubsetRandomSampler(test_indices)

    train_loader = DataLoader(dataset, batch_size=32, sampler=train_sampler)
    val_loader = DataLoader(dataset, batch_size=32, sampler=val_sampler)
    test_loader = DataLoader(dataset, batch_size=32, sampler=test_sampler)
    print(TAG + "Number of training/validation/test patches:", (len(train_indices),len(val_indices),len(test_indices)))

    # init probabilistic UNET or its variants
    print(TAG + 'num_filters: {}, latent dimension: {}, Fcomb layer number: {}, KLD weight: {}.'.format(
                num_filters, latent_dim, no_convs_fcomb, beta))

    # net = ProbabilisticUnet(input_channels=1, num_classes=1, num_filters=num_filters, 
                            # latent_dim=latent_dim, no_convs_fcomb=no_convs_fcomb, beta=beta)

    net = VNDUnet(input_channels=1, num_classes=1, num_filters=num_filters, latent_dim=latent_dim, no_convs_fcomb=no_convs_fcomb, beta=beta)
    net.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=init_lr, weight_decay=weight_decay)

    # train_loss, testing_loss, time, reg_loss
    # save best model w\ best loss, 

    best_val_elbo = 0
    for epoch in range(epochs):
        time_start = time.time()

        loss_dict = {'tr_elbo': 0, 'tr_loss': 0, 'val_elbo': 0}
        loss_dict = train(loss_dict)
        loss_dict = validation(loss_dict)
        print(TAG + 'Epoch: {}, Trainnig ELBO: {}, Training loss: {}, Validation ELBO: {}.'.format(
                        epoch, loss_dict['tr_elbo'], loss_dict['tr_loss'], loss_dict['val_elbo']))
