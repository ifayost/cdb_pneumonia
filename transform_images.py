import sys
import os
import numpy as np
from PIL import Image
from tqdm import tqdm

def crop_images(im, shapes):
	height = im.size[0]
	width = im.size[1]
	
	res_fact_h = 1 - (2/10)*(height/np.max(shapes[:,0]))
	res_fact_w = 1 - (3.5/10)*(width/np.max(shapes[:,1]))
	
	return im.crop((((1-res_fact_h)/2)*height,
					((1-res_fact_w)/2)*width,
					(1-((1-res_fact_h)/2))*height, 
					(1-((1-res_fact_w)/2))*width))


if __name__ == '__main__':

	path = os.path.join(sys.argv[1], 'train')
	shapes = []
	list_path = []
	for j in os.listdir(path):
		if j != '.DS_Store':
			for k in os.listdir(os.path.join(path,j)):
				if os.path.join(path,j,k)[-4:] == 'jpeg':
					im_path = os.path.join(path,j,k)
					list_path.append(im_path)
					im = Image.open(im_path)
					size = im.size
					shapes.append(size)
	shapes = np.array(shapes)
	
	
	for i in tqdm(list_path):
		im = Image.open(i)
		im = crop_images(im, shapes)
		im.save(i)	