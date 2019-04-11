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

testImage=CFdata[1]
norm_image = skimage.img_as_float64(testImage)
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
def Tamura(imag):
    #feature=[0,0,0,0,0]   
    Fcrs = coarseness(imag,kmax=4)
    Fcon = contrast(imag)
    f = directionality(imag) # return [Fdir,sita]
    feature = [ Fcrs, Fcon, f[0] ]
    return feature

# FIRST: Coarseness
# returns a float
def coarseness(graypic,kmax=4): # graphicis the input imahe. 2^kmax is the maximal window  
    height,width = graypic.shape   # get the size of image 
    A=np.zeros(shape=(height,width,2**kmax)) # A is the average grey level matrix in 3D
    
    # Calculte the average grey-level within 2^k neighborhood range for each pixel.  
    for i in range( (2**(kmax-1)+1), (height-2**(kmax-1)+1) ): # stop is not included in py
        for j in range( (2**(kmax-1)+1), (width-2**(kmax-1)+1) ):
            for k in range(1,kmax):
                a = anti_norm_image[ (i-2**(k-1)):(i+2**(k-1)) , (j-2**(k-1)):(j+2**(k-1)) ]
                A[i,j,k] = np.mean(a) # the mean of all the element in matrix
        
    # Calclute the differnece between the vertical and horizontal window's Ak for each pixel  
    Eh=np.zeros(shape=(height,width,2**kmax)) 
    Ev=np.zeros(shape=(height,width,2**kmax)) 
    for i in range( (2**(kmax-1)), (height-2**(kmax-1)) ): # stop is not included in py
        for j in range( (2**(kmax-1)), (width-2**(kmax-1)) ):
            for k in range(0,kmax):
                a = np.absolute(  A[(i+2**(k-1)),j,k] - A[(i-2**(k-1)),j]   ) # a is a vertical list
                b = np.absolute(  A[i,(j+2**(k-1)),k] - A[i,(j-2**(k-1))]   ) # b is a vertical list
                Eh[i,j]=a 
                Ev[i,j]=b  

    # Maximized k by calculating E  
    Sbest=np.zeros(shape=(height,width))
    maxkk=0
    for i in range( (2**(kmax-1)), (height-2**(kmax-1)) ): # stop is not included in py
        for j in range( (2**(kmax-1)), (width-2**(kmax-1)) ):
            maxEh = max(Eh[i,j,:] )  # Eh[i,j,:] is a vector
            maxEv = max(Ev[i,j,:] )  
            if maxEh > maxEv:  
                maxkk = maxEh  
            else:  
                maxkk = maxEv  
            Sbest[i,j]=2**maxkk # The best window size for each pixel is 2^maxkk  

    # The coarseness for the whole image is the average of all value in Sbest
    Fcrs = np.mean(Sbest)  
    return Fcrs # It is a number

# f=coarseness(anti_norm_image)


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

# f = directionality(anti_norm_image)
#################################################################################
# LBP feature vectors --> returns 10 numbers as bins

''' settings for LBP '''
radius = 3
n_points = 8 * radius

# https://blog.csdn.net/llh_1178/article/details/77833447?utm_source=blogxgwz7

lbp = local_binary_pattern(anti_norm_image, n_points, radius)
lbpHist, _ = np.histogram(lbp); # _ means ignore another returned value "bin_edges"

plt.imshow(lbp, cmap='gray')
plt.show()

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
    return [Asm,Con,-Eng,Idm]

img_gray= anti_norm_image

glcm_0=getGlcm(img_gray, 1,0)
glcm_1=getGlcm(img_gray, 0,1)
glcm_2=getGlcm(img_gray, 1,1)
glcm_3=getGlcm(img_gray, -1,1)

glcm_feature0=feature_computer(glcm_0)
glcm_feature1=feature_computer(glcm_1)
glcm_feature2=feature_computer(glcm_2)
glcm_feature3=feature_computer(glcm_3)
print(glcm_feature0,glcm_feature1,glcm_feature2,glcm_feature3)

# https://blog.csdn.net/kmsj0x00/article/details/79463376 

#################################################################################
# Gabor feature vectors




#################################################################################
# SIFT



#################################################################################
# CLASSIFICATION
#################################################################################
# SVM
