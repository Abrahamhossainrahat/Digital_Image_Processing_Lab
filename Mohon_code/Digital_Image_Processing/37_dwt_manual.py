import numpy as np
import matplotlib.pyplot as plt
import pywt
import cv2


def main():
    img = cv2.imread("/home/mohon/4_1/lab/dip_lab/images/lena.png", cv2.IMREAD_GRAYSCALE).astype(float)
    LL_m, LH_m, HL_m, HH_m = haar_dwt_2d(img)
    LL, (LH, HL, HH) = pywt.dwt2(img, 'haar')  # For verification with PyWavelets

    img_set = [LL_m, LH_m, HL_m, HH_m, LL, LH, HL, HH]
    titles = ['DWT LL', 'DWT LH', 'DWT HL', 'DWT HH', 'DWT LL (PyWavelets)', 'DWT LH (PyWavelets)', 'DWT HL (PyWavelets)', 'DWT HH (PyWavelets)']
    display(img_set, titles)

def haar_dwt_1d(signal):
    """Manual 1D Haar DWT."""
    N = len(signal)
    approx = []
    detail = []
    for i in range(0, N, 2):
        a = (signal[i] + signal[i+1]) / np.sqrt(2)
        d = (signal[i] - signal[i+1]) / np.sqrt(2)
        approx.append(a)
        detail.append(d)
    return np.array(approx), np.array(detail)

def haar_dwt_2d(image):
    """Manual 2D Haar DWT (single level)."""
    # Row transform
    rows_approx = []
    rows_detail = []
    for row in image:
        a, d = haar_dwt_1d(row)
        rows_approx.append(a)
        rows_detail.append(d)
    rows_approx = np.array(rows_approx)
    rows_detail = np.array(rows_detail)

    # Column transform
    cols_approx = []
    cols_detail = []
    for col in rows_approx.T:
        a, d = haar_dwt_1d(col)
        cols_approx.append(a)
        cols_detail.append(d)
    LL = np.array(cols_approx).T
    LH = np.array(cols_detail).T

    cols_approx = []
    cols_detail = []
    for col in rows_detail.T:
        a, d = haar_dwt_1d(col)
        cols_approx.append(a)
        cols_detail.append(d)
    HL = np.array(cols_approx).T
    HH = np.array(cols_detail).T

    return LL, LH, HL, HH

def display(img_set, img_title):
    plt.figure(figsize=(15, 10))
    
    for i in range(len(img_set)):        
        plt.subplot(2, 4, i + 1)  
        
        if(img_set[i].ndim == 2):    # show gray-scale image 
            plt.imshow(img_set[i], cmap='gray')
            plt.title(img_title[i])  
            plt.axis('off') 
        else:   # show histogram (
            plt.bar(range(len(img_set[i])), img_set[i])
            plt.title(img_title[i])            
    
    plt.tight_layout() 
    plt.show()


if __name__ == "__main__":
    main()