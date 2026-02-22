import matplotlib.pyplot as plt
import numpy as np
import cv2

def main():
    img = cv2.imread("/home/abraham/Desktop/Abraham/CSE4-1/DIP_Lab/Image/dog.jpg", 0)
    
    blurred = cv2.GaussianBlur(img, (5,5), 1.3)
    
    canny_edge = cv2.Canny(blurred, 50, 180)
    
    plt.figure(figsize = (12,8))
    plt.subplot(1,2,1)
    plt.imshow(img, cmap = 'gray')
    
    plt.subplot(1,2,2)
    plt.imshow(canny_edge, cmap = 'gray')
    
    plt.tight_layout()
    plt.show()
    
    
    
if __name__ == "__main__":
    main()