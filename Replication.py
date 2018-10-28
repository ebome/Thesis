import skimage
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import os
import glob


# Read images from folder
img_dir = 'D:\AMME4111\#Code\#Partitioned'# The Directory of all images 
data_path = os.path.join(img_dir,'*g')
files = glob.glob(data_path)
CFdata = []
for f1 in files:
    img = cv2.imread(f1)
    CFdata.append(img) 
    # CFdata is a list with size=139 and each element a np-array(3 channel image,uint8)

# print image.shape
for i in range(len(CFdata)):
    CFdata[i] = cv2.cvtColor(CFdata[i], cv2.COLOR_BGR2GRAY)
##################################################
flags=[i for i in dir(cv2) if i.startswith('COLOR_')] 
print (flags)
##################################################
