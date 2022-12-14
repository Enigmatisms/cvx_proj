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

    #==================?????????=============================
    mask = get_1_minus1(arr)
    dst = arr * mask

    return dst

def spectrum_fft(fft):
    """
    return FFT spectrum
    """
    return np.sqrt(np.power(fft.real, 2) + np.power(fft.imag, 2))

def median_filter(img, f ,uk, vk, radius1=10,radius2=10):
    """
    create butterworth notch resistant filter, equation 4.155
    param: img:    input, source image
    param: uk:     input, int, center of the height
    param: vk:     input, int, center of the width
    param: radius: input, int, the radius of circle of the band pass filter, default is 10
    param: w:      input, int, the width of the band of the filter, default is 5
    param: n:      input, int, order of the butter worth fuction, 
    return a [0, 1] value butterworth band resistant filter
    """   
    M, N = img.shape[1], img.shape[0]
    
    u = np.arange(M)
    v = np.arange(N)
    
    u, v = np.meshgrid(u, v)
    
    
    DK1 = np.sqrt((u - M//2 - uk)**2 + (v - N//2 - vk)**2)
    DK2 = np.sqrt((u - M//2 - uk)**2 + (v - N//2 + vk)**2)
    DK3 = np.sqrt((u - M//2 + uk)**2 + (v - N//2 - vk)**2)
    DK4 = np.sqrt((u - M//2 + uk)**2 + (v - N//2 + vk)**2)

    f [DK1<=radius1] = np.median(f[DK1<=radius2])
    f [DK2<=radius1] = np.median(f[DK2<=radius2])
    f [DK3<=radius1] = np.median(f[DK3<=radius2])
    f [DK4<=radius1] = np.median(f[DK4<=radius2])
    
    return f

def apply_median_filter(img_ori: np.ndarray):
    # ?????????????????????????????????
    img_ori = cv2.equalizeHist(img_ori)
    # img_ori = np.hstack((img_ori,equ))
    # ???????????????????????????
    # FFT
    img_fft = np.fft.fft2(img_ori.astype(np.float32))
    # ?????????
    fshift = np.fft.fftshift(img_fft)            # ?????????????????????????????????????????????
    # ?????????????????????
    spectrum_fshift = spectrum_fft(fshift)

    # ????????????????????????
    spectrum_log = np.log(1 + spectrum_fshift)

    r = 5
    uk = 48 # 1/8 width and height 
    sn = 9

    f1shift = fshift

    for i in range(2,sn):
        f1shift = median_filter(img_ori,f1shift, radius1=r,radius2=r*5, uk=i*uk, vk=i*uk)
        for j in range (1,i):
            f1shift = median_filter(img_ori,f1shift, radius1=r,radius2=r*5, uk=j*uk, vk=i*uk)
            f1shift = median_filter(img_ori,f1shift, radius1=r,radius2=r*5, uk=i*uk, vk=j*uk)
        f1shift = median_filter(img_ori,f1shift, radius1=3,radius2=r*3, uk=0, vk=i*uk) 
        f1shift = median_filter(img_ori,f1shift, radius1=3,radius2=r*3, uk=i*uk, vk=0) 

    # ??????????????????
    spectrum_filter = spectrum_fft(f1shift)
    spectrum_filter_log = np.log(1 + spectrum_filter)
    f2shift = np.fft.ifftshift(f1shift) #????????????????????????
    img_new = np.fft.ifft2(f2shift)
    img_new = np.abs(img_new)
    return img_new, spectrum_log, spectrum_filter_log, img_ori

def single_channel_main(image: np.ndarray):
    img_new, spectrum_log, spectrum_filter_log, img_hist = apply_median_filter(image[..., 0])
    plt.figure(figsize=(15, 15))
    plt.subplot(221), plt.imshow(img_hist, 'gray'), plt.title('With Sine noise'), plt.xticks([]),plt.yticks([])
    plt.subplot(222), plt.imshow(spectrum_log, 'gray'), plt.title('Spectrum'), plt.xticks([]),plt.yticks([])
    plt.subplot(223), plt.imshow(spectrum_filter_log, 'gray'), plt.title('Spectrum Filtered'), plt.xticks([]),plt.yticks([])
    plt.subplot(224), plt.imshow(img_new, 'gray'), plt.title('Denoised'), plt.xticks([]),plt.yticks([])
    plt.tight_layout()
    plt.show()
    cv2.imwrite("img_haze1_denoised.png",img_new)

def all_channel_main(image: np.ndarray):
    output_channels = []
    histed_channels = []
    for i in range(3):
        img_new, _, _, img_histed = apply_median_filter(image[..., i])
        output_channels.append(img_new)
        histed_channels.append(img_histed)
    img_output = np.stack(output_channels, axis = -1)
    print(img_output.dtype)
    img_histed = np.stack(histed_channels, axis = -1)
    img_output /= 255
    plt.figure(1)
    plt.subplot(121), plt.imshow(img_histed), plt.title('Original image'), plt.xticks([]),plt.yticks([])
    plt.subplot(122), plt.imshow(img_output), plt.title('Output filtered'), plt.xticks([]),plt.yticks([])
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    img_ori = cv2.imread('../diff_1/raw_data/case1/scat/img_haze1.png')
    all_channel_main(img_ori)
