import matplotlib.pyplot as plt 
import numpy as np 
import cv2

def display(img):
    plt.figure(figsize=(12,8))
    for i in range(len(img)):
        plt.subplot(1,3,i+1)
        plt.imshow(img[i], cmap = 'gray')
        plt.axis('off')
    plt.tight_layout()
    plt.show()
    
def main():
    img = cv2.imread("/home/abraham/Desktop/Abraham/CSE4-1/DIP_Lab/Image/dog.jpg")
    
    blurred = cv2.GaussianBlur(img, (5,5), 1.3)
    canny_edge = cv2.Canny(blurred, 50, 200)
    
    img_set = [img, blurred, canny_edge]
    display(img_set)



if __name__ == "__main__":
    main()
