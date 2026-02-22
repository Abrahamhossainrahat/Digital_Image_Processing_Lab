import matplotlib.pyplot as plt 
import numpy as np 
import cv2

lower_limit = 127
upper_limit = 200

def thresold1(img):
    img_temp = img.copy()
    height, width = img.shape
    for i in range(height):
        for j in range(width):
            if img_temp[i][j] < lower_limit :
                img_temp[i][j] = 0
            else:
                img_temp[i][j] = 255
                
    return img_temp

def thresold2(img):
    img_temp = img.copy()
    height, width = img.shape
    for i in range(height):
        for j in range(width):
            if lower_limit <= img_temp[i][j] <= upper_limit :
                img_temp[i][j] = 127
                
    return img_temp

def thresold3(img):
    img_temp = img.copy()
    height, width = img.shape
    for i in range(height):
        for j in range(width):
            if lower_limit <= img_temp[i][j] <= upper_limit :
                img_temp[i][j] = 127
            else:
                img_temp[i][j] = 255
                
    return img_temp

def histogram(img):
    hist = np.zeros(256, dtype = int)
    height , width = img.shape
    for i in range(height):
        for j in range(width):
            hist[img[i][j]] += 1
            
    return hist
    
def display(img):
    
    plt.figure(figsize=(12,6))
    n = len(img)
    for i in range(n):
        plt.subplot(2,n,i+1)
        plt.imshow(img[i], cmap = 'gray')
        plt.axis('off')
    
    for i in range(n):
        plt.subplot(2,n,i+n+1)
        hist = histogram(img[i])
        plt.bar(range(256), hist, color = 'gray')
    plt.tight_layout()
    plt.show()
    

def main():
    img = cv2.imread("/home/abraham/Desktop/Abraham/CSE4-1/DIP_Lab/Image/img1.jpg", 0)
    
    thresold1_img = thresold1(img)
    thresold2_img = thresold2(img)
    thresold3_img = thresold3(img)
    
    img_set = [img, thresold1_img, thresold2_img, thresold3_img]
    titles = ["Original_Img", "thresold1_img", "thresold2_img", "thresold3_img"]
    
    display(img_set)
    
    
if __name__ == "__main__":
    main()