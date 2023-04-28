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

# For plotting
import numpy as np
import matplotlib.pyplot as plt

# For utilities
import time, sys, os

# For conversion
import cv2
import opencv_transforms.transforms as TF
import dataloader
import torchvision.transforms as transforms

# For everything
import torch
import torch.nn as nn
import torchvision.utils as vutils

# For our model
import mymodels
import torchvision.models


if __name__ == '__main__':

	warnings.simplefilter("ignore", UserWarning)
	with torch.no_grad():
		print(torch.cuda.is_available())
	device = 'cuda' if torch.cuda.is_available() else 'cpu'

	if device=='cuda':
	    print("The gpu to be used : {}".format(torch.cuda.get_device_name(0)))
	else:
	    print("No gpu detected")
	with torch.no_grad():
	    netC2S = mymodels.Color2Sketch(pretrained=True).to(device)
	    netC2S.eval()
	# batch_size. number of cluster
	batch_size = 1
	ncluster = 9

	# Validation 
	print('Loading Validation data...', end=' ')
	val_transforms = TF.Compose([
	    TF.Resize(512),
	    ])
	val_imagefolder = dataloader.PairImageFolder('./dataset/val_void', val_transforms, netC2S, ncluster)
	val_loader = torch.utils.data.DataLoader(val_imagefolder, batch_size=batch_size, shuffle=False)
	print("Done!")
	print("Validation data size : {}".format(len(val_imagefolder)))


	# Test
	print('Loading Test data...', end=' ')
	test_transforms = TF.Compose([
	    TF.Resize(512),
	    ])
	test_imagefolder = dataloader.GetImageFolder('./dataset/test_void', test_transforms, netC2S, ncluster)
	test_loader = torch.utils.data.DataLoader(test_imagefolder, batch_size=batch_size, shuffle=False)
	print("Done!")

	print("size:",format(test_imagefolder))
	# Reference
	print('Loading Reference data...', end=' ')
	refer_transforms = TF.Compose([
	    TF.Resize(512),
	    ])
	refer_imagefolder = dataloader.GetImageFolder('./dataset/reference_void', refer_transforms, netC2S, ncluster)
	refer_loader = torch.utils.data.DataLoader(refer_imagefolder, batch_size=1, shuffle=False)
	refer_batch = next(iter(refer_loader))
	print("Done!")
	print("Reference data size : {}".format(len(refer_imagefolder)))
	temp_batch_iter = iter(refer_loader)
	temp_batch = next(temp_batch_iter)
	edge = temp_batch[0]
	# Convertir le tenseur en une forme appropriée
	temp_tensor = edge.squeeze().permute(0,1,2)

	color = temp_batch[1]
	color_palette = temp_batch[2]

	color_list = temp_batch[2]
	# A : Edge, B : Color
	nc = 3 * (ncluster + 1)
	print(type(nc))
	netG = mymodels.Sketch2Color(nc=nc, pretrained=True).to(device) 

	num_params = sum(p.numel() for p in netG.parameters() if p.requires_grad)
	print('Number of parameters: %d' % (num_params))

	torch.backends.cudnn.benchmark = True
	temp_batch_iter=iter(test_loader)
	import torch.nn.functional as F


	netG.eval()
	temp_batch = next(temp_batch_iter)

	from PIL import Image
	from PIL import Image, ImageOps

	# Charger l'image avec PIL
	img = temp_batch[0].squeeze()

	# Convertir l'image en un tableau NumPy
	img_np = np.array(img)

	# Définir les transformations à appliquer à l'image
	transform = TF.Compose([
	    TF.Resize((512,512)),
	    TF.ToTensor(),             # convertir l'image en un tensor PyTorch
	])

	# Appliquer les transformations à l'image
	edge = transform(img_np)
	edge_tmp = edge.unsqueeze(0)
	# Convertir le tensor en une image PIL
	#edge_img = TF.ToPILImage(edge_tmp.squeeze())

	# Afficher l'image avec matplotlib
	#plt.imshow(edge_tmp)
	#plt.show()

	with torch.no_grad():
	    
	    #notre edge
	    edge = edge_tmp
	    edge = edge.to(device)  # Déplacez edge sur le GPU
	    encode = netC2S.forward(edge)

	    #affichage de l'image après l'autoencoder
	    encode = encode.permute(0, 2, 3, 1)
	    encode = encode.detach().cpu().numpy()
	    img = (encode[0] * 255).astype('uint8')
	    cv2.imwrite('./outputs/sketch.png', img)
	    #edge de base
	    #edge = temp_batch[0].to(device)

	    real = temp_batch[1].to(device)
	    
	    reference = refer_batch[1].to(device)
	    
	    
	    color_palette = refer_batch[2]

		

	    input_tensor = torch.cat([edge.cpu()]+color_palette, dim=1).to(device)
	    fake = netG(input_tensor)
	    result = torch.cat((reference, edge, fake), dim=-1).cpu()
	    
	    
	    #result_for_save = torch.cat(fake, dim=-1)
	    output_for_save = vutils.make_grid(fake, nrow=1, padding=5, normalize=True).cpu().permute(1,2,0).numpy()
		    
	    # Save images to file
	    save_path = 'outputs/'
	    save_name = 'img-{}.jpg'.format("image_genere")
	    plt.imsave(arr=output_for_save, fname='{}{}'.format(save_path, save_name))
		    
	    # Convertir le tensor en une image PIL
	    to_pil = transforms.ToPILImage()
	    pil_image = to_pil(fake[0])

	    folder_path = './dataset/test_void/test'

	    # Obtenez la liste des fichiers dans le dossier
	    files = os.listdir(folder_path)

	    # Bouclez à travers tous les fichiers dans le dossier
	    for file in files:
	    	if file.endswith('.jpg') or file.endswith('.jpeg') or file.endswith('.png'):
	    		image = Image.open(os.path.join(folder_path, file))
	    		width_ori,height_ori = image.size
	    #Upscale Image
	    image_ori = image
	    from super_image import EdsrModel, ImageLoader
	    from PIL import Image
	    import requests
	    
	    model = EdsrModel.from_pretrained('eugenesiow/edsr', scale=2)   
	    img = Image.open("outputs/img-image_genere.jpg")
	    sketch = Image.open("outputs/sketch.png")
	    
	    inputs = ImageLoader.load_image(img)
	    inputs2 = ImageLoader.load_image(sketch)
	    preds2 = model(inputs2)
	    preds = model(inputs)

	    ImageLoader.save_image(preds, './outputs/scaled_2x.png')
	    ImageLoader.save_image(preds2, './outputs/sketch_scaled_2x.png')
	    # Charger l'image
	    image = Image.open("outputs/scaled_2x.png")
	    image_sketch = Image.open("outputs/sketch_scaled_2x.png")
	    new_size = (width_ori, height_ori)
	    image_resized = image.resize(new_size)
	    image_resized_sketch = image_sketch.resize(new_size)
	    # Enregistrer l'image dans un fichier
	    #image_resized.show()
	    image_resized.save('./outputs/image_final_redimensionné.png')
	    image_resized_sketch.save('./outputs/sketch_redimensionné.png')
	    # Charger les deux images
	    image2 = Image.open("outputs/image_final_redimensionné.png")
	    sketch = Image.open("outputs/sketch_redimensionné.png")
	    # Récupérer les dimensions des images
	    largeur, hauteur = image_sketch.size

	    # Créer une nouvelle image avec une largeur deux fois plus grande que les images originales
	    nouvelle_image = Image.new('RGB', (largeur*3, height_ori))
	    # Coller les deux images côte à côte
	    nouvelle_image.paste(sketch, (0,0))
	    nouvelle_image.paste(image2, (largeur,0))
	    nouvelle_image.paste(image_ori, (largeur*2,0))
	    nouvelle_image.save('./outputs/s_p_o.png')
	    # Afficher l'image résultante
	    nouvelle_image.show()
