# -*- coding: utf-8 -*-

import rawpy
import imageio
#import numpy as np
#np.set_printoptions(threshold=np.inf)

def rgb2ycc(img, delta):
	for i in range(img.shape[0]-1):
		for j in range(img.shape[1]-1):
			R = img[i][j][0]
			G = img[i][j][1]
			B = img[i][j][2]
			Y = 0.299*R + 0.587*G + 0.114*B
			Cr = (R-Y)*0.713 + delta
			Cb = (B-Y)*0.564 + delta
			img[i][j][0] = Y
			img[i][j][1] = Cr
			img[i][j][2] = Cb

#			[0.299, 0.587, 0.144]

	return img

def ycc2rgb(img, delta):
	R = Y + 1.403*(Cr-delta)
	G = Y - 0.714*(Cr-delta) - 0.344*(Cb-delta)
	B = Y + 1.773*(Cb-delta)

def rgb2gray(rgb):

    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return gray

'''
raw_image = 'DSC_0214.NEF'

raw = rawpy.imread(raw_image)
rgb = raw.postprocess(use_camera_wb = True)
imageio.imwrite('default.tiff', rgb)

'''

im = imageio.imread('./resources/DSC_0214.tif')
color_format = '16bit';

if color_format == '8bit':
	# For 8-bit images
	delta = 128
elif color_format == '16bit':
	# For 16-bit images
	delta = 32768

#print(rgb2ycc(im, delta))
print(rgb2gray(im))
#print(im.shape)
#print(range(im.shape[0]-1))
