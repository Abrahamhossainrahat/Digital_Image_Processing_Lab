import matplotlib.pyplot as plt 
import numpy as np
import cv2

def display(img, titles):
    plt.figure(figsize = (12,8))
    for i in range(len(img)):
        plt.subplot(4,5,i+1)
        plt.imshow(img[i], cmap = 'gray')
        plt.axis()

def fft_img(img):
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    magnitude = np.log(np.abs(fshift)+1)
    return fshift, magnitude
    
def image_processing(file, index):
    img = cv2.imread(file, 0)
    
    fshift, magnitude = fft_img(img)

def main():
    img1 = "/home/abraham/Desktop/Abraham/CSE4-1/DIP_Lab/Image/dog.jpg"
    img2 = "/home/abraham/Desktop/Abraham/CSE4-1/DIP_Lab/Image/img1.jpg"
    img3 = "/home/abraham/Desktop/Abraham/CSE4-1/DIP_Lab/Image/rose.jpeg"
    img4 = "/home/abraham/Desktop/Abraham/CSE4-1/DIP_Lab/Image/canny.png"
    img5 = "/home/abraham/Desktop/Abraham/CSE4-1/DIP_Lab/Image/couple.png"
    
    files = [img1, img2, img3, img4, img5]
    for idx , file in enumerate(files, 1):
        print(f"{file} is running")
        image_processing(file, idx)

    
    
    
if __name__ == "__main__":
    main()