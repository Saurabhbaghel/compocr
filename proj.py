import numpy as np
from google.colab.patches import cv2_imshow
import random
import os
import matplotlib.pyplot as plt
from doctr.io import DocumentFile
from doctr.models import ocr_predictor
from argparse import ArgumentParser



def detection(img_path):
	img_name = args.path_img
	os.environ['USE_TORCH'] = '1'
	predictor = ocr_predictor()
	doc = DocumentFile.from_images(img_name)
	result = predictor(doc)
	json_export = result.export()

	return json_export


def recognition()




def main(args):
	
	image = cv.imread(img_name)
	h, w = image.shape[:-1]
	i=0
	for block in json_export['pages'][0]['blocks']:
	  for line in  block['lines']:
	    i+=1
	    p1,p2 = line['geometry']
	    x1,y1 = int(p1[0]*w), int(p1[1]*h)
	    x2,y1 = int(p2[0]*w), y1
	    x2,y2 = x2, int(p2[1]*h)
	    x1,y2 = x1, y2
	    p1=(x1,y1)
	    p2=(x2,y2)
	    
	    # print(p1,p2)
	    file_name = f'{i}.jpg'
	    file = image[x1:x2,y1:y2,:]
	    # cv.imwrite(file_name,file)
	    thickness = 2
	    color = (0,0,255)
	    image = cv.rectangle(image,p1,p2,color,thickness)

	cv2_imshow(image)


def arg_parser():
	parser = ArgumentParser()
	parser.add_argument('--path_img',help='address of the image file')
	args = parser.parse_args()

	return arg


if __name__ == 'main':
	
	args=arg_parser()
	main(args)



