'''
Problem Statement:
The task is to read an image in RGB format and convert it into grayscale format
manually using two different methods:
1. Using the formula:
    I_gray = 0.299 * R + 0.587 * G + 0.114 * B
2. Using the formula:
    I_gray = (R + G + B) / 3
The first method uses different weights for each color channel, while the second method
uses equal weights for all channels.   
'''

#======================= Necessary Imports =========================
import matplotlib.pyplot as plt
import cv2
import numpy as np 

def main():
    #======================== Read Images ===========================
    img_path = "/home/mohon/4_1/code/dip_code/images/rgb6.png"
    img_3D = cv2.imread(img_path, cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img_3D, cv2.COLOR_BGR2RGB)
    img_gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    #======================== Call Grayscale Functions ==============
    img_manual_gray = gray(img_rgb)
    img_manual_gray_same_weight = gray_same_weight(img_rgb)     


    #======================== Display Images =======================
    plt.subplot(2, 2, 1)
    plt.imshow(img_rgb)
    plt.title('Original RGB Image')
    

    plt.subplot(2, 2, 2)
    plt.imshow(img_gray, cmap='gray')
    plt.title('Grayscale Image')
   

    plt.subplot(2,2,3)
    plt.imshow(img_manual_gray, cmap='gray')
    plt.title('Manual Grayscale Image')

    plt.subplot(2,2,4)
    plt.imshow(img_manual_gray_same_weight, cmap='gray')
    plt.title('Manual Grayscale Image (Same Weight)')
    

    plt.tight_layout()
    plt.show()
    plt.close()

#======================== Grayscale Conversion Functions ==============
def gray(img_3D):
    rgb_red = img_3D[:, :, 0]
    rgb_green = img_3D[:, :, 1]
    rgb_blue = img_3D[:, :, 2]

    img_gray = 0.299 * rgb_red + 0.587 * rgb_green + 0.114 * rgb_blue
    img_gray = img_gray.astype(np.uint)

    return img_gray

#==================== Grayscale Conversion with Same Weight ============
def gray_same_weight(img_3D):
    rgb_red = img_3D[:, :, 0]
    rgb_green = img_3D[:, :, 1]
    rgb_blue = img_3D[:, :, 2]


    img_gray = (rgb_red + rgb_green + rgb_blue) / 3
    img_gray = img_gray.astype(np.uint)

    return img_gray




if __name__ == '__main__':
    main()

