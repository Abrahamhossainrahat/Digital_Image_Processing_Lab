import cv2
import numpy as np
import matplotlib.pyplot as plt

def main():
    #--------------------- Load image ---------------------
    img = cv2.imread("/home/mohon/4_1/lab/dip_lab/images/canny_sample.png", 0)

    #----------------- Apply Gaussian blur----------------
    img = cv2.GaussianBlur(img, (3, 3), 1.4)

    #---------------- Apply Canny Edge Detection----------
    edge1 = cv2.Canny(img, threshold1=50, threshold2=100)
    edge2 = cv2.Canny(img, threshold1=50, threshold2=200)
    edge3 = cv2.Canny(img, threshold1=100, threshold2=150)
    edge4 = cv2.Canny(img, threshold1=100, threshold2=200)
    

    hist_original = histogram(img)
    hist_edge = histogram(edge4)

    img_set = [img, edge4, hist_original, hist_edge]
    img_title = ['Original image', 'Canny Edge', 'Original histogram',
                 'Canny Edge Histogram']
    img_set_2 = [edge1, edge2, edge3, edge4]
    img_title_2 = ['Threshold: 50-100', 'Threshold: 50-200', 'Threshold: 100-150', 'Threshold: 100-200']

    display(img_set_2, img_title_2)
    display(img_set, img_title)

#--------------------- Histogram Calculation ---------------------
def histogram(img_2D):
    h, w = img_2D.shape
    hist = np.zeros(256, dtype=int)

    for i in range(h):
        for j in range(w):
            pixel_value = img_2D[i, j]
            hist[pixel_value] += 1

    return hist


#--------------------- Display image ---------------------
def display(img_set, img_title):
    for i in range(len(img_set)):
        
        plt.subplot(2, 2, i + 1)  
        if(img_set[i].ndim == 2):       
            plt.imshow(img_set[i], cmap = 'gray')
            plt.title(img_title[i])  
            plt.axis('off') 
        else:   
            plt.bar(range(256), img_set[i])
            plt.title(img_title[i])            
    
    plt.tight_layout() 
    plt.show()


if __name__ == '__main__':
    main()
