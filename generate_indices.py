from load_LIDC_data import LIDC_IDRI
dataset = LIDC_IDRI(dataset_location = '/home/nandcui/data/plidc-punet/')
dataset_size = len(dataset)
indices = list(range(dataset_size))

# generate 10-fold cross validation set
import numpy as np
import csv

nsplits = 10

split = int(np.floor(0.1 * dataset_size))

for i in range(nsplits):
	np.random.shuffle(indices)
	train = indices[:-int(2*split)]
	val = indices[-int(2*split):-int(split)]
	test = indices[-int(split):]

	split_list = [train, val, test]
	with open('splits/split_{}.csv'.format(i), 'wb') as f:
		writer = csv.writer(f)
		writer.writerows(split_list)
