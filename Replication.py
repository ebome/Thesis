import cv2

from skimage.transform import rotate
from skimage.feature import local_binary_pattern
from skimage import data, io
from skimage.color import label2rgb
import skimage
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import os
import glob

#################################################################################
# Read images from folder
img_dir = 'D:\AMME4111\#Code\#Partitioned'# The Directory of all images 
data_path = os.path.join(img_dir,'*g')
files = glob.glob(data_path)
CFdata = []
for f1 in files:
    img = cv2.imread(f1)
    CFdata.append(img) 
    # CFdata is a list with size=139 and each element a np-array(3 channel image,uint8)

# Print image.shape
for i in range(len(CFdata)):
    CFdata[i] = cv2.cvtColor(CFdata[i], cv2.COLOR_BGR2GRAY) 
    # in cv2, image is read as BGR rather than RGB
'''
# Check all flags in cv2

flags=[i for i in dir(cv2) if i.startswith('COLOR_')] 
print (flags)
'''

testImage=CFdata[32]
norm_image = cv2.normalize(testImage, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

#################################################################################
# Apply band-pass spatial filter
''' 2D Gaussian kernel is used
'''
high_kernel_size = (11, 11); # ksize MUST be odd numbers
low_kernel_size = (5,5);
sigma = 0; # if both sigmas are zeros, they are computed from ksize.width and ksize.height
blur_HIGH = cv2.GaussianBlur(norm_image,high_kernel_size,sigma) 

var_0 = norm_image - blur_HIGH #  L-G_h(L)
var_1 = abs(norm_image - blur_HIGH)  #  |L-G_h(L)|
blur_HIGH_denominater = cv2.GaussianBlur(var_1,high_kernel_size,sigma) 
# blur_HIGH,blur_LOW,var_0,var_1, blur_HIGH_denominater are all np array with float type

# Prevent 0/0 when do bit-wise division
var_3 = np.divide( var_0, blur_HIGH_denominater, out=np.zeros_like(var_0), where=blur_HIGH_denominater!=0 )

HIGH_PASS_filter = 0.25*var_3+1 # element-wise summation and multiplication

BAND_PASS_filter = cv2.GaussianBlur(HIGH_PASS_filter,low_kernel_size,sigma)

'''
Ensure no negative values after filtering

# mini=BAND_PASS_filter[BAND_PASS_filter != 0].min()
# print(mini)
'''

anti_norm_image = cv2.normalize(BAND_PASS_filter, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX) # normalize L into region [0,1] again

'''
plt.subplot(121)
plt.imshow(BAND_PASS_filter,cmap='gray')
plt.title('BAND_PASS_filter')
plt.subplot(122)
plt.imshow(anti_norm_image,cmap='gray')
plt.title('anti_norm_image') 
'''


#################################################################################
# FEATURE EXTRACTION SECTION
#################################################################################
# Tamura feature vectors


anti_norm_image




#################################################################################
# LBP feature vectors

''' settings for LBP '''
radius = 3
n_points = 8 * radius

# 仅通过cvtcolor()函数是没有办法将灰色图直接转化为RGB图像的 (https://blog.csdn.net/llh_1178/article/details/77833447?utm_source=blogxgwz7)

lbp = local_binary_pattern(anti_norm_image, n_points, radius)

plt.imshow(lbp, cmap='gray')
plt.show()



#################################################################################
# GLCM feature vectors






#################################################################################
# Gabor feature vectors

