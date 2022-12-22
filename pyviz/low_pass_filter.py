import numpy as np
import cv2
import matplotlib.pyplot as plt

def butterworth_band_resistant_filter(source, center, radius=10, w=5, n=1):
    """
    create butterworth band resistant filter, equation 4.150
    param: source: input, source image
    param: center: input, the center of the filter, where is the lowest value, (0, 0) is top left corner, source.shape[:2] is 
                   center of the source image
    param: radius: input, int, the radius of circle of the band pass filter, default is 10
    param: w:      input, int, the width of the band of the filter, default is 5
    param: n:      input, int, order of the butter worth fuction, 
    return a [0, 1] value butterworth band resistant filter
    """    
    epsilon = 1e-8
    N, M = source.shape[:2]
    
    u = np.arange(M)
    v = np.arange(N)
    
    u, v = np.meshgrid(u, v)
    
    D = np.sqrt((u - center[1]//2)**2 + (v - center[0]//2)**2)
    C0 = radius
    
    temp = (D * w) / ((D**2 - C0**2) + epsilon)
    
    kernel = 1 / (1 + temp ** (2*n)) 
    return kernel

def butterworth_low_pass_filter(img, center, radius=5, n=1):
    """
    create butterworth low pass filter 
    param: source: input, source image
    param: center: input, the center of the filter, where is the lowest value, (0, 0) is top left corner, source.shape[:2] is 
                   center of the source image
    param: radius: input, the radius of the lowest value, greater value, bigger blocker out range, if the radius is 0, then all
                   value is 0
    param: n: input, float, the order of the filter, if n is small, then the BLPF will be close to GLPF, and more smooth from low
              frequency to high freqency.if n is large, will close to ILPF
    return a [0, 1] value filter
    """  
    epsilon = 1e-8
    M, N = img.shape[1], img.shape[0]
    
    u = np.arange(M)
    v = np.arange(N)
    
    u, v = np.meshgrid(u, v)
    
    D = np.sqrt((u - center[1]//2)**2 + (v - center[0]//2)**2)
    D0 = radius
    kernel = (1 / (1 + (D / (D0 + epsilon))**(2*n)))
    
    return kernel

def pad_image(img, mode='constant'):
    """
    pad image into PxQ shape, orginal is in the top left corner.
    param: img: input image
    param: mode: input, str, is numpy pad parameter, default is 'constant'. for more detail please refere to Numpy pad
    return PxQ shape image padded with zeros or other values
    """
    dst = np.pad(img, ((0, img.shape[0]), (0, img.shape[1])), mode=mode)
    return dst    

def centralized_2d(arr):
    """
    centralized 2d array $f(x, y) (-1)^{x + y}$, about 4.5 times faster than index, 160 times faster than loop,
    """
    
    def get_1_minus1(img):
        """
        get 1 when image index is even, -1 while index is odd, same shape as input image, need this array to multiply with input image
        to get centralized image for DFT
        Parameter: img: input, here we only need img shape to create the array
        return such a [[1, -1, 1], [-1, 1, -1]] array, example is 3x3
        """
        dst = np.ones(img.shape)
        dst[1::2, ::2] = -1
        dst[::2, 1::2] = -1
        return dst

    #==================中心化=============================
    mask = get_1_minus1(arr)
    dst = arr * mask

    return dst

def spectrum_fft(fft):
    """
    return FFT spectrum
    """
    return np.sqrt(np.power(fft.real, 2) + np.power(fft.imag, 2))

# 陷波滤波器处理周期噪声，用巴特沃斯低通滤波器得到的效果比目前的陷波滤波器效果还要好
img_ori = cv2.imread('diff_1/raw_data/case1/scat/img_haze1.png', 0)
img_ori = cv2.equalizeHist(img_ori)
M, N = img_ori.shape[:2]

fp = pad_image(img_ori, mode='reflect')
fp_cen = centralized_2d(fp)
fft = np.fft.fft2(fp_cen)

# 中心化后的频谱
spectrum_fshift = spectrum_fft(fft)
spectrum_log = np.log(1 + spectrum_fshift)

# 滤波器
n = 15
r = 20
H = butterworth_low_pass_filter(fp, fp.shape, radius=180, n=4)
# BNRF_1 = butterworth_notch_resistant_filter(fp, radius=r, uk=355, vk=0, n=n)
# BNRF_2 = butterworth_notch_resistant_filter(fp, radius=r, uk=0, vk=355, n=n)
# BNRF_3 = butterworth_notch_resistant_filter(fp, radius=r, uk=250, vk=250, n=n)
# BNRF_4 = butterworth_notch_resistant_filter(fp, radius=r, uk=-250, vk=250, n=n)
# BNRF = BNRF_1 * BNRF_2 * BNRF_3 * BNRF_4  * H
BNRF = H

fft_filter = fft * BNRF

# 滤波后的频谱
spectrum_filter = spectrum_fft(fft_filter)
spectrum_filter_log = np.log(1 + spectrum_filter)

# 傅里叶反变换
ifft = np.fft.ifft2(fft_filter)

# 去中心化反变换的图像，并取左上角的图像
img_new = centralized_2d(ifft.real)[:M, :N]
img_new = np.clip(img_new, 0, img_new.max())
img_new = np.uint8(img_new)

plt.figure(figsize=(15, 12))
plt.subplot(221), plt.imshow(img_ori, 'gray'), plt.title('With noise'), plt.xticks([]),plt.yticks([])
plt.subplot(222), plt.imshow(spectrum_log, 'gray'), plt.title('Spectrum'), plt.xticks([]),plt.yticks([])
plt.subplot(223), plt.imshow(spectrum_filter_log, 'gray'), plt.title('Spectrum With Filter'), plt.xticks([]),plt.yticks([])
plt.subplot(224), plt.imshow(img_new, 'gray'), plt.title('IDFT'), plt.xticks([]),plt.yticks([])
plt.tight_layout()
plt.show()
cv2.imwrite("img_haze1_denoised.png",img_new)