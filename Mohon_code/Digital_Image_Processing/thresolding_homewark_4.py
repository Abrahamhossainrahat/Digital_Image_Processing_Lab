'''
Problem Statement:
You are given a grayscale image. Perform the following:
1. Plot the original grayscale image.
2. Apply three different thresholding methods:
   a) Thresholding 1: Binary threshold at limit1.
   b) Thresholding 2: Pixels in [limit1, limit2] set to 127.
   c) Thresholding 3: Pixels in [limit1, limit2] set to 127, pixels > limit2 set to 255.
3. Plot the histogram for each image (original + three thresholded images).
'''

#================= Importing necessary libraries ======================

import matplotlib.pyplot as plt
import cv2
import numpy as np 


#================= Threshold limits ====================================
limit1 = 127
limit2 = 200


#================= Main function =======================================
def main():   
    # Load the image in grayscale mode (0 = grayscale flag)
    img_gray = cv2.imread("/home/mohon/4_1/code/dip_code/images/rgb_flower1.png", 0)

    # Create a list of images: original + three thresholded versions
    img_set = [img_gray, thresolding1(img_gray), thresolding2(img_gray), thresolding3(img_gray)]

    # Compute histograms for each image in the set
    hist_set = [histogram(img_set[0]), histogram(img_set[1]), histogram(img_set[2]), histogram(img_set[3])]    

    display_img(img_set)
    display_hist(hist_set)



#================= Function to display images ===============================
def display_img(img_set):
    for i in range(len(img_set)):        
        plt.subplot(2,4, i+1)
        plt.imshow(img_set[i], cmap = 'gray')
        plt.axis('off')
        
    


#================= Function to display histograms ======================
def display_hist(hist):
    for i in range(len(hist)):
        plt.subplot(2, 4, i+5)
        plt.bar(range(256), hist[i], linewidth=2)
    
    plt.tight_layout()
    plt.show()


#================= Thresholding Method 1 ======================
def thresolding1(img_gray):
    img_tmp = img_gray.copy()
    
    for i in range(img_gray.shape[0]):
        for j in range(img_gray.shape[1]):
            if(img_tmp[i][j] <= limit1):
                img_tmp[i][j]    = 0
            else:
                img_tmp[i][j] = 255   
    
    return img_tmp

#================= Thresholding Method 2 ======================
def thresolding2(img_gray):
    img_tmp = img_gray.copy()
    
    for i in range(img_gray.shape[0]):
        for j in range(img_gray.shape[1]):
            if( limit1 <= img_tmp[i][j]  and img_tmp[i][j]<=limit2):
                img_tmp[i][j] = 127    
    
    return img_tmp

#================= Thresholding Method 3 ======================
def thresolding3(img_gray):
    img_tmp = img_gray.copy()
    
    for i in range(img_gray.shape[0]):
        for j in range(img_gray.shape[1]):
            if( limit1 <= img_tmp[i][j]  and img_tmp[i][j]<=limit2):
                img_tmp[i][j] = 127 
            elif (limit2 < img_tmp[i][j]):
                img_tmp[i][j] = 255    
    
    return img_tmp


#================= Function to calculate histogram ==================
def histogram(gray):
     
    hist = np.zeros((256,), dtype=np.uint32)

    for i in range(gray.shape[0]):
        for j in range(gray.shape[1]):
                hist[gray[i, j]] += 1
    
        
    return hist

            
#================= Run the program ======================
if __name__ == '__main__':
    main()

