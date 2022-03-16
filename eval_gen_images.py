import torch
import numpy as np
import torchvision
from metric import calc_energy_distances, get_energy_distance_components

path = '/Users/ralphc/Dropbox/experiments/pami_2022/'
image_path = path + 'prediction_images_hard/'

# load three fold
for i in  range(3):

	for step in range(48):
		# '/' + str(step) + '_patch' + '.png'
		categories = [str(i) + '/' + str(step) + '_patch' + '.png', 
					  str(i) + '/' + str(step) + '_mask' + '.png', 
					  str(i) + '/' + str(step) + '_recons' + '.png']

		patch = torchvision.io.read_image(image_path + categories[0])
		mask = torchvision.io.read_image(image_path + categories[1])
		recon = torchvision.io.read_image(image_path + categories[2])

		print(patch)
		print(mask)
		print(recon)

		print(patch.shape)
		print(mask.shape)
		print(recon.shape)
		exit()