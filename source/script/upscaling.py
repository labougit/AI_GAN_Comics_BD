import torch
import torch.nn as nn
import os

#Upscale Image
from super_image import EdsrModel, ImageLoader
from PIL import Image
import requests

# To ignore warning
import warnings
import sys

if __name__ == '__main__':

	arg  = sys.argv[1]
	output = sys.argv[2]
	warnings.simplefilter("ignore", UserWarning)
	with torch.no_grad():
		model = EdsrModel.from_pretrained('eugenesiow/edsr', scale=2)

		folder_path = str(arg)
		      
		img = Image.open(folder_path)

		inputs = ImageLoader.load_image(img)
		preds = model(inputs)

		ImageLoader.save_image(preds, output)

