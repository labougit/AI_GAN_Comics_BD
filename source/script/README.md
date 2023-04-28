# Folder Scripts
A simple repositorie where you can find all scripts used to create the dataset: [Kaggle Comics and BD dataset](https://www.kaggle.com/datasets/mrarmonius/bd-and-comics).

List of scripts used to obtain dataset Comics published on Kaggle.

Launch order:
- rename_script
- bd color to sketch
- rescale_image
- extratc_color
- train_val

## Rename script
A simple script which rename all files with number and keep only jpg and png extension.

## Bd color to sketch
The script uses helper.py.
The aim of this script is to resize the image in 512 square and to keep the ratio of this image. Moreover the script create a sketch version of the raw image and paste it horizontally with the raw image.

## rescale_image
The script add Pads if needed to have the image in the format of 1024x512 pixels. It add padding equally to the raw image and sketch image.

## extract color
The script extract a palette of 16 colors from the raw image. An dwrite in 4 regions of 4 colours each in JSON file.

## train_val.
Split data between train val value with a ratio of 20% for val and 80 % for train.

## upscaling
Simple script which uses super_image librarie and pre-training weights to upscale your image. It uses an AI model.
