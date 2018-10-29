import cv2

from skimage.feature import local_binary_pattern
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
# Tamura feature vectors: coarseness, directionality, contrast
def Tamura(imag):
    #feature=[0,0,0,0,0]   
    Fcrs = coarseness(imag,kmax=4)
    Fcon = contrast(imag)
    [Fdir,sita] = directionality(imag)
    feature = [ Fcrs, Fcon,Fdir ]
    return feature

# 第一个指标 Coarseness，粗糙度

def coarseness(graypic,kmax=4) # graphic为待处理的灰度图像，2^kmax为最大窗口  
    height,width = graypic.shape   # 获取图片大小  
    A=np.zeros(shape=(height,width,2**kmax)); # 平均灰度值矩阵A: 3D matrix 
    #计算有效可计算范围内每个点的2^k邻域内的平均灰度值  
    for i in range( (2**(kmax-1)+1), (height-2**(kmax-1)+1) ): # stop is not included in py
        for j in range( (2**(kmax-1)+1), (width-2**(kmax-1)+1) ):
            for k in range(1,kmax):
                
                
                
                
                
                A(i,j,k)=mean2(graypic(i-2**(k-1):i+2**(k-1)-1,j-2**(k-1):j+2**(k-1)-1));  
     
        
        
        
    # 对每个像素点计算在水平和垂直方向上不重叠窗口之间的Ak差  
    for i in range( (2**(kmax-1)+1), (height-2**(kmax-1)+1) ): # stop is not included in py
        for j in range( (2**(kmax-1)+1), (width-2**(kmax-1)+1) ):
            for k in range(1,kmax):
                Eh(i,j,k)=abs(A(i+2**(k-1),j,k)-A(i-2**(k-1),j));  
                Ev(i,j,k)=abs(A(i,j+2**(k-1),k)-A(i,j-2**(k-1)));  

    # 对每个像素点计算使E达到最大值的k  
    for i in range( (2**(kmax-1)+1), (height-2**(kmax-1)+1) ): # stop is not included in py
        for j in range( (2**(kmax-1)+1), (width-2**(kmax-1)+1) ):
            [maxEh,p]=max(Eh(i,j,:));  
            [maxEv,q]=max(Ev(i,j,:));  
            if maxEh>maxEv  
                maxkk=p;  
            else  
                maxkk=q;  
            end  
        Sbest(i,j)=2**maxkk; # 每个像素点的最优窗口大小为2^maxkk  
    end  
end  
    # 所有Sbest的均值作为整幅图片的粗糙度  
    Fcrs=mean2(Sbest);  
    return Fcrs





# 第二个指标 Contrast，对比度
# 注意这个函数因为涉及到方差，要求输入类型为double，因此我这里在源代码上做了适当的修改  
def contrast(graypic) # graypic为待处理的灰度图片  
    graypic=double(graypic); # 这一句我自己做了修改，否则原博文中的代码不能直接运行  
    x=graypic(:); # 二维向量一维化  
    M4=mean((x-mean(x)).^4); # 四阶矩  
    delta2=var(x,1); # 方差  
    alfa4=M4/(delta2**2); # 峰度  
    delta=std(x,1); # 标准差  
    Fcon=delta/(alfa4**(1/4)); # 对比度  
    return Fcon  



# 第三个指标 Directionality，方向度
%sita为各像素点的角度矩阵，在线性度中会用到，所以这里作为结果返回  
function [Fdir,sita]=directionality(graypic)  
[h w]=size(graypic);  
%两个方向的卷积矩阵  
GradientH=[-1 0 1;-1 0 1;-1 0 1];  
GradientV=[ 1 1 1;0 0 0;-1 -1 -1];  
%卷积，取有效结果矩阵  
MHconv=conv2(graypic,GradientH);  
MH=MHconv(3:h,3:w);  
MVconv=conv2(graypic,GradientV);  
MV=MVconv(3:h,3:w);  
%向量模  
MG=(abs(MH)+abs(MV))./2;  
%有效矩阵大小  
validH=h-2;  
validW=w-2;  
%各像素点的方向  
for i=1:validH  
    for j=1:validW  
        sita(i,j)=atan(MV(i,j)/MH(i,j))+(pi/2);  
    end  
end  
n=16;  
t=12;  
Nsita=zeros(1,n);  
%构造方向的统计直方图  
for i=1:validH  
    for j=1:validW  
        for k=1:n  
            if sita(i,j)>=(2*(k-1)*pi/2/n) && sita(i,j)<((2*(k-1)+1)*pi/2/n) && MG(i,j)>=t  
                Nsita(k)=Nsita(k)+1;  
            end  
        end  
    end  
end  
for k=1:n  
    HD(k)=Nsita(k)/sum(Nsita(:));  
end  
%假设每幅图片只有一个方向峰值，为计算方便简化了原著  
[maxvalue,FIp]=max(HD);  
Fdir=0;  
for k=1:n  
    Fdir=Fdir+(k-FIp)^2*HD(k);%公式与原著有改动  
end  
end  

  





#################################################################################
# LBP feature vectors --> now is 10

''' settings for LBP '''
radius = 3
n_points = 8 * radius

# 仅通过cvtcolor()函数是没有办法将灰色图直接转化为RGB图像的 
# https://blog.csdn.net/llh_1178/article/details/77833447?utm_source=blogxgwz7

lbp = local_binary_pattern(anti_norm_image, n_points, radius)
lbpHist, _ = np.histogram(lbp); # _ means ignore another returned value "bin_edges"

plt.imshow(lbp, cmap='gray')
plt.show()

#################################################################################
# GLCM feature vectors






#################################################################################
# Gabor feature vectors


#################################################################################
# SIFT



#################################################################################
# CLASSIFICATION
#################################################################################
# SVM
