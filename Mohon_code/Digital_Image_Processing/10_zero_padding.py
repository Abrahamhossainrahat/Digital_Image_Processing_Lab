'''
Problem Statement: Zero Padding
    Given two grayscale images, the task is to zero pad the smaller image so that
    it matches the size of the larger image. The zero-padded image should be centered
    within the larger image.
'''
#======================= Necessary Imports =========================
import matplotlib.pyplot as plt
import numpy as np
import cv2


def main():
    #======================== Read Images =========================
    # Read two grayscale images
    gray_1 = cv2.imread("/home/mohon/4_1/code/dip_code/images/gray_1.png")
    gray_2 = cv2.imread("/home/mohon/4_1/code/dip_code/images/gray_2.png")

    #========================= Get to Grayscale =========================
    gray_1 = gray_1[:,:, 0]
    gray_2 = gray_2[:,:, 0]    

    #========================= Show the shape of the images ==================
    print(gray_1.shape)
    print(gray_2.shape)    

    #========================= Zero Padding =========================
    # Calculate the number of rows and columns to pad
    row_left = (gray_1.shape[0] - gray_2.shape[0] + 1 ) // 2    
    column_top = (gray_1.shape[1] - gray_2.shape[1] + 1 ) // 2  

    #========================= Create a zero-padded image =========================
    # Create a new image with the same size as gray_1, initialized to zeros  
    row = gray_1.shape[0]
    column = gray_1.shape[1]
    gray_3 = np.zeros(( row, column) )
    #========================= Place gray_2 in the center of gray_3 (Way 1)===========
    #gray_3[row_left: row_left+gray_2.shape[0], column_top:column_top+gray_2.shape[1]] = gray_2

    #========================= Place gray_2 in the center of gray_3 (Way 2)===========
    for i in range(gray_2.shape[0]):
        for j in range(gray_2.shape[1]):
            gray_3[row_left + i, column_top + j] = gray_2[i, j]
    


    #========================= Original first and Second the images ===============
    plt.figure(figsize=(10, 10))
    plt.subplot(3,1,1)
    plt.imshow(gray_1, cmap = "gray")

    plt.subplot(3,1,2)
    plt.imshow(gray_2, cmap = "gray")

    #========================= Zero-padded image ==================================
    plt.subplot(3,1,3)
    plt.imshow(gray_3, cmap = 'gray')


    plt.tight_layout()
    plt.show()


#========================= Main Function =========================
if __name__ == '__main__':
    main()
