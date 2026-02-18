'''
Problem Statement:
Implement a non-linear transformation on a grayscale image using the formula:
    I_out = I_in^g
    or,
    S = C * R^g
where g is a constant (gamma) and C = 1, is a constant scaling factor.
The value of g can be varied to observe the effect of non-linear transformation
on the image.
'''

#======================= Necessary Imports =========================
import matplotlib.pyplot as plt
import numpy as np
import cv2


def main():
    #======================== Read Images ============================
    gray = cv2.imread("/home/mohon/4_1/code/dip_code/images/gray_1.png", cv2.IMREAD_GRAYSCALE)
      
    #======================== Define Constants =========================
    gammas = [0.1, 0.3, 0.7, 1, 2, 3]

    plt.subplot(4,2,1)
    plt.imshow(gray, cmap = 'gray')
    plt.title('Original Grayscale Image')
    
    for i in range(len(gammas)) :
        img_2D = gray
        gray_non_linear = non_linear(img_2D, gammas[i])       

        plt.subplot(4,2, i+2)
        plt.title(f'Non-linear Transformation with g = {gammas[i]}')
        plt.axis('off')
        plt.imshow(gray_non_linear, cmap = 'gray')  
    

    plt.tight_layout()
    plt.show()


#======================== Non-linear Transformation Function ============= 
def non_linear(img_2D, gamma):
    img_2D = img_2D.astype(np.float32)
    # Normalize the image to the range [0, 1]
    img_2D = img_2D / 255.0
    
    for i in range(img_2D.shape[0]):
        for j in range(img_2D.shape[1]):
           img_2D[i,j] = img_2D[i,j] ** gamma
    
    return img_2D * 255  # Scale back to [0, 255] range



#========================= Main Function =========================
if __name__ == '__main__':
    main()
