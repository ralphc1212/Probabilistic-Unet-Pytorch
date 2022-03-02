from load_LIDC_data import LIDC_IDRI
dataset = LIDC_IDRI(dataset_location = '/home/nandcui/data/plidc-punet/')
dataset_size = len(dataset)
indices = list(range(dataset_size))


print(dataset_size)
print(indices)