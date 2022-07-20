from doctr.models.zoo import ocr_predictor
from doctr.io import DocumentFile
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import torch
import os


os.environ['USE_TORCH'] = '1'
# im = cv.imread('textfiles/Sanskrit2.jpg')
# ten = torch.from_numpy(im)
# ten.resize_((1,1024,1024,3))
# im=np.resize(im,(600,800,3))
im = DocumentFile.from_images(['/home/s/Desktop/Self/Job/IITBombay/Project/DocTR/compocr/textfiles/Sanskrit2.jpg'])
model = ocr_predictor(pretrained=True)
# print(im.shape)

out = model([im])
# plt.imshow(out)
print(out.export())
# plt.imshow(out)
# print(ten.size())