import cv2
import math
from scipy import signal
from skimage.feature import local_binary_pattern
from PIL import Image
import skimage
import numpy as np
import matplotlib.pyplot as plt

import os
import glob

'''img = cv2.imread("F:\images\Lena.jpg", 1)
cv2.imshow("1", img)
cv2.waitKey()'''
#################################################################################
# Read images from folder
img_dir = r'C:\Users\Yang\Desktop\AMME4111\#Code\#Partitioned'# The Directory of all images 
data_path = os.path.join(img_dir,'*g')
files = glob.glob(data_path)
CFdata = []
for f1 in files:
    img = cv2.imread(f1)
    CFdata.append(img) 
    # CFdata is a list with size=139 and each element a np-array(3 channel image,uint8)


for i in range(len(CFdata)):
    CFdata[i] = cv2.cvtColor(CFdata[i], cv2.COLOR_BGR2GRAY) 
    # in cv2, image is read as BGR rather than RGB

'''
# Check all flags in cv2
flags=[i for i in dir(cv2) if i.startswith('COLOR_')] 
print (flags)
'''

testImage=CFdata[1]
norm_image = skimage.img_as_float64(testImage)
plt.imshow(norm_image);plt.title('normed image')
# Convert an image to double-precision (64-bit) floating point format, with values in [0, 1]

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

HIGH_PASS_filter = 0.25*var_3+0.5 # element-wise summation and multiplication

BAND_PASS_filter = cv2.GaussianBlur(HIGH_PASS_filter,low_kernel_size,sigma)

'''Ensure no negative values after filtering
'''
'''
mini=BAND_PASS_filter[BAND_PASS_filter != 0].min()
print(mini)
'''

anti_norm_image = skimage.img_as_ubyte(BAND_PASS_filter)
# Convert an image to unsigned byte format, with values in [0, 255]


plt.imshow(anti_norm_image,cmap='gray')


#################################################################################
# FEATURE EXTRACTION SECTION
#################################################################################
# Tamura feature vectors: coarseness, directionality, contrast
# -----------------------------------------------------------------------------
# FIRST: Coarseness
def averageOverNeighborhoods(x,y,k,img):
    imgHeight, imgWidth = img.shape
    result = 0.0
    border = 2**(2 * k)
    
    for i in range(0, math.ceil(border)): 
        
        for j in range(0, math.ceil(border)):
            var12 = x - int( (2**(k - 1)) ) + i
            var12 = int(var12)
            var13 = y - int( (2**(k - 1)) ) + j
            var13 = int(var13)
            if var12 < 0 :
               var12 = 0
            if var13 < 0 :
               var13 = 0
            if (var12 >= imgWidth) :
               var12 = imgWidth - 1
            if (var13 >= imgHeight):
               var13 = imgHeight - 1
            # up to now var12 and var13 are int,
            # in original code, the image has to be processed from RGB into grayScales, but here we do not need 
            result = result + anti_norm_image[var12,var13]

    result = ( 1.0 / (2**(2 * k)) ) * result
    return result



def differencesBetweenNeighborhoodsHorizontal(x,y,k,img): # x, y, k are int
    a = averageOverNeighborhoods(x + int( (2**(k - 1))  ), y, k,img)
    b = averageOverNeighborhoods(x - int( (2**(k - 1))  ), y, k,img)
    result = abs(a - b)
    return result
 
def differencesBetweenNeighborhoodsVertical(x,y,k,img):
    a = averageOverNeighborhoods(x, y + int(  (2**(k - 1))  ), k,img)
    b = averageOverNeighborhoods(x, y - int(  (2**(k - 1))  ), k,img)
    result = abs(a - b)
    return result

def sizeLeadDiffValue(x,y,img): # x and y are int
    result = 0.0
    maxK = 1
    for k in range(0, 3): 
        horizon = differencesBetweenNeighborhoodsHorizontal(x, y, k,img)
        vertical = differencesBetweenNeighborhoodsVertical(x, y, k,img)
        tmp = max(horizon, vertical);
        if result < tmp:
            maxK = k
            result = tmp
    return maxK;

def coarseness(n0,n1,img): # n0,n1 are integers
    result = 0.0
    for i in range(1, n0 - 1): 
        for j in range(1, n1 - 1):
            double = sizeLeadDiffValue(i, j,img)
            result = result + 2**double
    result = ( 1.0 / (n0 * n1) ) * result
    return result



imgHeight, imgWidth = anti_norm_image.shape

f=coarseness(imgWidth,imgHeight,anti_norm_image)

# -----------------------------------------------------------------------------
# SECOND: Contrast
def contrast(graypic):  
    graypic = graypic.astype(float)  # convert data type from uint8 to float
    x = graypic.reshape(-1)     # Convert 2D arrray in 1D  
    a=(x-np.mean(x))
    M4 = np.mean( np.power(a, 4) )  # Fourth moment
    delta2 = np.var(x,ddof=1) # Variance  
    alfa4 = M4/(delta2**2) # Kurtosis  
    delta = np.std(x,ddof=1) # Standard deviation  
    Fcon = delta/(alfa4**(1/4)) # Contrast  
    return Fcon   # It is a number

# -----------------------------------------------------------------------------
# THIRD: Directionality
# sita is the angular matrxi for each pixel  
def directionality(graypic):  
    h,w = graypic.shape   

    # Sobel Kernels 
    GradientH = np.array([[-1,0,1],[-1,0,1],[-1,0,1]])  
    GradientV = np.array([[1,1,1],[0,0,0],[-1,-1,-1]])
    
    # Convolution; Get valid Matrix 
    MHconv = signal.convolve2d(graypic,GradientH)  
    MH=MHconv[3:(h+1),3:(w+1)]  
    MVconv=signal.convolve2d(graypic,GradientV) 
    MV=MVconv[3:(h+1),3:(w+1)] 
    # get the modulus
    MG=(abs(MH)+abs(MV))/2  
    
    # size of valid matrix 
    validH = h-2  
    validW = w-2 
    sita = np.zeros(shape=(validH,validW))
    MH[MH == 0] = 1 # since we cannot let MV/MH have zero in MH
    # The dirction of each pixel  
    for i in range(0,validH):  
        for j in range(0,validW):
            temp=MV[i,j]/MH[i,j]
            sita[i,j] = np.arctan( temp ) + (np.pi/2)

    n=16  
    t=12  
    Nsita = np.zeros(shape=(1,n))  
    # Calculte the histogram of directionality  
    for i in range(0,validH):  
        for j in range(0,validW):  
            for k in range(0,n):  
                if sita[i,j]>=(2*(k-1)*np.pi/2/n) and sita[i,j]<((2*(k-1)+1)*np.pi/2/n) and MG[i,j]>=t :  
                    Nsita[:,k]=Nsita[:,k]+1 
 
    HD = np.zeros(shape=(1,n))
    for k in range(0,n):  
        HD[:,k]=Nsita[:,k]/(Nsita.sum()) # sum all the elements in Nsita
 
    # HD is histogram 
    FIp = np.argmax(HD) # Find index of maximum value in HD
    Fdir=0  
    for k in range(0,n):  
        Fdir = Fdir + np.square((k-FIp)) * HD[:,k] # Modification of equations for simpler computation 

    return [Fdir,sita] 

FFFFf=contrast(anti_norm_image)


#################################################################################
# LBP feature vectors --> returns 10 numbers as bins

''' settings for LBP '''
radius = 3
n_points = 8 * radius

# https://blog.csdn.net/llh_1178/article/details/77833447?utm_source=blogxgwz7

lbp = local_binary_pattern(anti_norm_image, n_points, radius)
lbpHist, _ = np.histogram(lbp); # _ means ignore another returned value "bin_edges"

plt.imshow(lbp, cmap='gray');plt.title('local binary pattern')

#################################################################################
# GLCM feature vectors

# Define the maximal grey level
gray_level = 16

def maxGrayLevel(img_gray):
    max_gray_level=0
    (height,width)=img_gray.shape
    for y in range(height):
        for x in range(width):
            if img_gray[y][x] > max_gray_level:
                max_gray_level = img_gray[y][x]
    return max_gray_level+1

def getGlcm(img_gray,d_x,d_y):
    srcdata=img_gray.copy()
    ret=[[0.0 for i in range(gray_level)] for j in range(gray_level)]
    (height,width) = img_gray.shape

    max_gray_level=maxGrayLevel(img_gray)

    # If the grey-level is large than 16，we could shrink the level to 16 for reducing computation time
    if max_gray_level > gray_level:
        for j in range(height):
            for i in range(width):
                srcdata[j][i] = srcdata[j][i]*gray_level / max_gray_level

    for j in range(height-d_y):
        for i in range(width-d_x):
             rows = srcdata[j-1][i-1]
             cols = srcdata[j + d_y-1][i+d_x-1]
             ret[rows][cols]+=1.0

    for i in range(gray_level):
        for j in range(gray_level):
            ret[i][j]/=float(height*width)

    return ret # ret is the 16*16 GLCM 

def feature_computer(p):
    Con=0.0
    Eng=0.0
    Asm=0.0
    Idm=0.0
    for i in range(gray_level):
        for j in range(gray_level):
            Con+=(i-j)*(i-j)*p[i][j]
            Asm+=p[i][j]*p[i][j]
            Idm+=p[i][j]/(1+(i-j)*(i-j))
            if p[i][j]>0.0:
                Eng+=p[i][j]*math.log(p[i][j])
    return Asm,Con,-Eng,Idm
img_gray= anti_norm_image

glcm_0=getGlcm(img_gray, 1,0)
glcm_1=getGlcm(img_gray, 0,1)
glcm_2=getGlcm(img_gray, 1,1)
glcm_3=getGlcm(img_gray, -1,1)

asm0,con0,eng0,idm0=feature_computer(glcm_0)
asm1,con1,eng1,idm1=feature_computer(glcm_0)
asm2,con2,eng2,idm2=feature_computer(glcm_0)
asm3,con3,eng3,idm3=feature_computer(glcm_0)


# https://blog.csdn.net/kmsj0x00/article/details/79463376 

#################################################################################
# Gabor feature vectors




#################################################################################
# SIFT



#################################################################################
# CLASSIFICATION
#################################################################################
# SVM
