import matplotlib.pyplot as plt
import cv2
import numpy as np 

limit1 = 127
limit2 = 200


def main():
    
    
    img_gray1 = cv2.imread("/home/mohon/4_1/code/dip_code/images/rgb6.png", 0)
    img_set = [img_gray1, thresolding1(img_gray1), thresolding2(img_gray1), thresolding3(img_gray1)]

    img_gray1 = cv2.imread("/home/mohon/4_1/code/dip_code/images/rgb_flower1.png", 0)
    img_set += [img_gray1, thresolding1(img_gray1), thresolding2(img_gray1), thresolding3(img_gray1)]

    img_gray1 = cv2.imread("/home/mohon/4_1/code/dip_code/images/rgb_flower.png", 0)
    img_set += [img_gray1, thresolding1(img_gray1), thresolding2(img_gray1), thresolding3(img_gray1)]
    

    display(img_set)  

    


def display(img_set):

    for i in range(len(img_set)):
        
        plt.subplot(3,4, i+1)
        plt.imshow(img_set[i], cmap = 'gray')
        plt.axis('off')
        
    plt.tight_layout()
    plt.show()


def thresolding1(img_gray):
    img_tmp = img_gray.copy()
    
    for i in range(img_gray.shape[0]):
        for j in range(img_gray.shape[1]):
            if(img_tmp[i][j] <= limit1):
                img_tmp[i][j]    = 0
            else:
                img_tmp[i][j] = 255
    
    
    return img_tmp


def thresolding2(img_gray):
    img_tmp = img_gray.copy()
    
    for i in range(img_gray.shape[0]):
        for j in range(img_gray.shape[1]):
            if( limit1 <= img_tmp[i][j]  and img_tmp[i][j]<=limit2):
                img_tmp[i][j] = 127          
    
    
    return img_tmp

def thresolding3(img_gray):
    img_tmp = img_gray.copy()
    
    for i in range(img_gray.shape[0]):
        for j in range(img_gray.shape[1]):
            if( limit1 <= img_tmp[i][j]  and img_tmp[i][j]<=limit2):
                img_tmp[i][j] = 127 
            if (limit2 < img_tmp[i][j]):
                img_tmp[i][j] = 255
    
    
    return img_tmp

            

if __name__ == '__main__':
    main()

