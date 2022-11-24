import os
import cv2
import logging
import tensorflow as tf
import face_recognition
import math
import numpy as np
import dlib
import tensorflow
import PythonMagick
import matplotlib
import matplotlib.pyplot as plt
from scipy import linalg,ndimage,signal
from numpy import array,zeros
from scipy.ndimage import filters
from PIL import Image 
from skimage import util, img_as_float, io
from pylab import *

font_set = matplotlib.font_manager.FontProperties(fname=r"c:\windows\fonts\msyh.ttc", size=8)

def make_pyramid(ori_img, down_times):
    now_img = ori_img.copy()
    gaussian_pyramid = []
    gaussian_pyramid.append(now_img)
    for i in range(down_times):
        now_img = cv2.pyrDown(now_img)
        gaussian_pyramid.append(now_img)
    return gaussian_pyramid
 
img = cv2.imread("4.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
pyramid = make_pyramid(img,4)
 
fig, axes = plt.subplots(1, 5, dpi=300)
for idx, img in enumerate(pyramid):
    ax = axes[idx]
    ax.tick_params(direction='in', top=False, bottom=False, left=False, right=False, labelsize=4)
    ax.set_xlabel(f"第{idx}层图像",fontproperties=font_set,fontsize=6)
    ax.imshow(img)
plt.show()


def AddSaltAndPepperNosie(img, pro):
    noise = np.random.uniform(0, 255, img[:, :, 0].shape)
    mask = noise < pro * 255
    mask = np.expand_dims(mask, axis=2)
    mask = np.repeat(mask, 3, axis=2)
    img = img * (1 - mask)
    mask = noise > 255 - pro * 255
    mask = np.expand_dims(mask, axis=2)
    mask = np.repeat(mask, 3, axis=2)
    img = 255 * mask + img * (1 - mask)
    return img
def AddGaussNoise(img, sigma, mean=0):
    noise = np.random.normal(mean, sigma, img.shape).astype(np.float)
    img = img + noise
    img = np.clip(img, 0, 255).astype(np.uint8)
    return img
img = cv2.imread('4.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.figure(dpi=200)
plt.legend(prop={'family':'SimHei','size':15})

plt.subplot(1, 3, 1)
plt.title("原图",fontproperties=font_set)
plt.imshow(img)

noiseImgGauss = AddGaussNoise(img, 20, 0)
plt.subplot(1, 3, 2)
plt.title("高斯噪声",fontproperties=font_set)
plt.imshow(noiseImgGauss)

noiseImgSalt = AddSaltAndPepperNosie(img, 0.1)
plt.subplot(1, 3, 3)
plt.title("椒盐噪声",fontproperties=font_set)
plt.imshow(noiseImgSalt)

plt.figure(dpi=400)
plt.rc('font', family='Times New Roman', size=4)
img_gauss_gauss = cv2.GaussianBlur(noiseImgGauss, (5, 5), 5)
plt.subplot(1,4,1)
plt.title("高斯噪声+高斯平滑",fontproperties=font_set)
plt.imshow(img_gauss_gauss)
img_gauss_mid = cv2.medianBlur(np.uint8(noiseImgGauss), 3)
plt.subplot(1,4,2)
plt.title("高斯噪声+均值滤波",fontproperties=font_set)
plt.imshow(img_gauss_mid)
img_salt_gauss = cv2.GaussianBlur(np.uint8(noiseImgSalt), (5, 5), 5)
plt.subplot(1,4,3)
plt.imshow(img_salt_gauss)
plt.title("椒盐噪声+高斯平滑",fontproperties=font_set)
img_salt_mid = cv2.medianBlur(np.uint8(noiseImgSalt), 3)
plt.subplot(1,4,4)
plt.imshow(img_salt_mid)
plt.title("椒盐噪声+均值滤波",fontproperties=font_set)
plt.show()

plt.figure(dpi=400)

plt.subplot(1,3,1)
plt.title("高斯噪声",fontproperties=font_set)
plt.imshow(noiseImgGauss)
plt.subplot(1,3,2)
plt.title("高斯噪声+高斯平滑",fontproperties=font_set)
plt.imshow(img_gauss_gauss)
plt.subplot(1,3,3)
plt.title("高斯噪声+均值滤波",fontproperties=font_set)
plt.imshow(img_gauss_mid)
plt.show()
plt.figure(dpi=400)
plt.subplot(1,3,1)
plt.imshow(noiseImgSalt)
plt.title("椒盐噪声",fontproperties=font_set)
plt.subplot(1,3,2)
plt.imshow(img_salt_gauss)
plt.title("椒盐噪声+高斯平滑",fontproperties=font_set)
plt.subplot(1,3,3)
plt.imshow(img_salt_mid)
plt.title("椒盐噪声+均值滤波",fontproperties=font_set)
plt.show()

def calc_rmse(img1, img2):
    rmse = np.mean((img1 - img2) ** 2)
    rmse = np.sqrt(rmse)
    return rmse
rmuse1 = calc_rmse(noiseImgGauss, img)
print(rmuse1)
rmuse2 = calc_rmse(noiseImgSalt, img)
print(rmuse2)
rmuse3 = calc_rmse(img_gauss_gauss, img)
print(rmuse3)
rmuse4 = calc_rmse(img_gauss_mid, img)
print(rmuse4)
rmuse5 = calc_rmse(img_salt_gauss, img)
print(rmuse5)
rmuse6 = calc_rmse(img_salt_mid, img)
print(rmuse6)

img = cv2.imread('3.jpg')  
plt.figure(dpi=400)

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.subplot(2,2,1)
plt.title("原图",fontproperties=font_set)
plt.imshow(img)

plt.subplot(2,2,3)
img2 = cv2.Laplacian(img, cv2.CV_8U) 
plt.title("拉普拉斯滤波",fontproperties=font_set)
plt.imshow(img2)

img3 = cv2.Sobel(img, cv2.CV_8U, 0, 1)
plt.subplot(2,2,2)
plt.title("Sobel算子",fontproperties=font_set)
plt.imshow(img3)

img4 = cv2.Canny(img,cv2.CV_8U,200, 300)
plt.subplot(2,2,4)
plt.title("Canny滤波",fontproperties=font_set)
plt.imshow(img4)
img5=img+img2

img = cv2.imread('3.jpg') 
plt.figure(dpi=400)
#x梯度
imgx=cv2.Sobel(img,cv2.CV_8U,1,0)
imgx=Image.fromarray(cv2.cvtColor(imgx, cv2.COLOR_BGR2RGB))
imgx=array(imgx)
imgx = cv2.normalize(imgx, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
heat_imgx = cv2.applyColorMap(imgx, cv2.COLORMAP_HOT)
plt.subplot(2, 2, 1)
plt.title("X方向梯度",fontproperties=font_set)
plt.imshow(heat_imgx)
#y梯度
imgy=cv2.Sobel(img,cv2.CV_8U,0,1)
imgy=Image.fromarray(imgy)
imgy=array(imgy)
imgy = cv2.normalize(imgy, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
heat_imgy = cv2.applyColorMap(imgy, cv2.COLORMAP_HOT)
plt.subplot(2, 2, 2)
plt.title("Y方向梯度",fontproperties=font_set)
plt.imshow(heat_imgy)
#计算幅度
A = cv2.magnitude(np.float32(imgx), np.float32(imgy))
A = cv2.normalize(A, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
A_heat = cv2.applyColorMap(A, cv2.COLORMAP_HOT)
plt.subplot(2, 2, 3)
plt.title("梯度幅度",fontproperties=font_set)
plt.imshow(A_heat)
# 计算角度
an = cv2.phase(np.float32(imgx), np.float32(imgy), True)
an = cv2.normalize(an, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
an_heat = cv2.applyColorMap(an, cv2.COLORMAP_HOT)
plt.subplot(2, 2, 4)
plt.title("梯度角度",fontproperties=font_set)
plt.imshow(an_heat)
plt.show()


img = cv2.imread('3.jpg') 
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.figure(dpi=400)
plt.subplot(1,2,1)
plt.title("原图",fontproperties=font_set)
plt.imshow(img)
plt.subplot(1,2,2)
plt.title("原图+拉普拉斯滤波",fontproperties=font_set)
plt.imshow(img5)
plt.show()
