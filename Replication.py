from skimage import io,exposure
import matplotlib.pyplot as plt

img = io.imread("099.png", 0) 
plt.figure("hist",figsize=(8,8))

arr=img.flatten()
plt.subplot(221)
plt.imshow(img,plt.cm.gray)  #Original image
plt.subplot(222)
plt.hist(arr, bins=256, normed=1,edgecolor='None',facecolor='red') # Original histogram 

img1=exposure.equalize_hist(img)
arr1=img1.flatten()
plt.subplot(223)
plt.imshow(img1,plt.cm.gray)  #Equalized image
plt.subplot(224)
plt.hist(arr1, bins=256, density=1,edgecolor='None',facecolor='red') # Equakized histogram

plt.show()