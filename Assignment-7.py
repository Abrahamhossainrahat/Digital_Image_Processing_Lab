import matplotlib.pyplot as plt 
import numpy as np 
import cv2

def display(img, titles):
    plt.figure(figsize=(12,8))
    
    for i in range(len(img)):
        plt.subplot(2,3,i+1)
        if i<2:
            plt.imshow(img[i], cmap='gray')
            plt.axis('off')
        else:
            plt.bar(range(256), img[i])
        plt.title(titles[i])
    plt.tight_layout()
    plt.show()


def histogram(img):
    h, w = img.shape
    hist = np.zeros(256, dtype= int)
    
    for i in range(h):
        for j in range(w):
            value = img[i,j]
            hist[value] += 1
    return hist
    
def h_pdf(hist):
    return hist/sum(hist)

def h_cdf(pdf):
    return np.cumsum(pdf)
    
def convolution(img, new_level):
    return new_level[img]
    
def main():
    img = cv2.imread("/home/abraham/Desktop/Abraham/CSE4-1/DIP_Lab/Image/img1.jpg", 0)
    
    hist = histogram(img)
    pdf = h_pdf(hist)
    cdf = h_cdf(pdf)
    new_level = np.round(np.array(cdf) * 255).astype(np.uint8)
    new_img = convolution(img, new_level)
    hist_e = histogram(new_img)
    
    img_set = [img, new_img, hist, pdf, cdf, hist_e]
    titles = ["Original_Image","New_Image", "Histogram", "PDF", "CDF","Histoghram_Equalization"]
    
    display(img_set, titles)
    


if __name__ == "__main__":
    main()
