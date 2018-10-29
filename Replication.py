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
testImage=CFdata[10]
norm_image = cv2.normalize(testImage, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

# apply band-pass spatial filter
'''
通过一维高斯核生成二维高斯核:先获取两个一维高斯核，
而后对后一个高斯核进行转置，而后第一个高斯核核第二个高斯核通过矩阵相乘就可以得到一个二维高斯核了
https://blog.csdn.net/qq_16013649/article/details/78784791

'''
def gaussian_kernel_2d_opencv(kernel_size = 0,sigma = 0.5): 
    # kernel_size is aperture size. It should be odd (1 3 5...) and positive.
    # Gaussian kernel size can be zero's and then they are computed from sigma
    # returen Gaussian filter coefficients
    kx = cv2.getGaussianKernel(kernel_size,sigma)
    ky = cv2.getGaussianKernel(kernel_size,sigma)
    return np.multiply(kx,np.transpose(ky)) 
