import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from load_LIDC_data import LIDC_IDRI
# from probabilistic_unet import ProbabilisticUnet
from vnd_unet import VNDUnet
from utils import l2_regularisation
import csv

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dataset = LIDC_IDRI(dataset_location = '/home/nandcui/data/plidc-punet/')
# dataset_size = len(dataset)
# indices = list(range(dataset_size))
# split = int(np.floor(0.1 * dataset_size))
# np.random.shuffle(indices)

# num of cross validation
K = 3
splits = []
for i in range(K):

    split = []
    with open('splits/split_{}.csv'.format(i), 'r') as f:
        reader = csv.reader(f)
        for i, line in enumerate(reader):
            line = [int(elem) for elem in line]
            split.append(line)
    splits.append(split)

print(splits)
exit()

train_indices, test_indices = indices[split:], indices[:split]
train_sampler = SubsetRandomSampler(train_indices)
test_sampler = SubsetRandomSampler(test_indices)
train_loader = DataLoader(dataset, batch_size=32, sampler=train_sampler)
test_loader = DataLoader(dataset, batch_size=32, sampler=test_sampler)
print("Number of training/test patches:", (len(train_indices),len(test_indices)))

# start: lr, weight_decay, kld_weight, optimizer
# train_loss, testing_loss, time, reg_loss
# save best model w\ best loss, 

# net = ProbabilisticUnet(input_channels=1, num_classes=1, num_filters=[32,64,128,192], latent_dim=6, no_convs_fcomb=4, beta=10.0)
net = VNDUnet(input_channels=1, num_classes=1, num_filters=[32,64,128,192], latent_dim=6, no_convs_fcomb=4, beta=10.0)
net.to(device)
optimizer = torch.optim.Adam(net.parameters(), lr=1e-4, weight_decay=0)
epochs = 10
for epoch in range(epochs):
    for step, (patch, mask, _) in enumerate(train_loader): 
        patch = patch.to(device)
        mask = mask.to(device)
        mask = torch.unsqueeze(mask,1)
        net.forward(patch, mask, training=True)
        elbo = net.elbo(mask)
        reg_loss = l2_regularisation(net.posterior) + l2_regularisation(net.prior) + l2_regularisation(net.fcomb.layers)
        loss = - elbo + 1e-5 * reg_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
