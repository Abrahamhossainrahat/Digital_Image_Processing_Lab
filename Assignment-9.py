import matplotlib.pyplot as plt
import numpy as np
import cv2
from skimage import exposure

#Display function 
def display(img, titles):
    plt.figure(figsize = (12, 8))
    for i in range(len(img)):
        plt.subplot(2,5,i+1)
        if i<5:
            plt.imshow(img[i], cmap = 'gray')
            plt.axis('off')
        else:
            plt.bar(range(256), img[i])
        plt.title(titles[i])
    
    plt.tight_layout()
    plt.show()
    

# Histogram calculation
def histogram(img):
    h, w = img.shape
    hist = np.zeros(256, dtype = np.uint8)
    for i in range(h):
        for j in range(w):
            value = img[i, j]
            hist[value] += 1
    return hist

#PDF
def pdf(hist):
    return hist/sum(hist)
#CDF   
def cdf(pdf):
    return np.cumsum(pdf)
    
# Built-in histogram matching
def histogram_matching_built_in(source, reference):
    return exposure.match_histograms(source, reference, channel_axis=None).astype(np.uint8)

# Histogram matching 1
def histogram_matching_img1(cdf1, cdf2):
    mapping = np.zeros(256, dtype = np.uint8)
    for i in range(256):
        diff = np.abs(cdf1 - cdf2[i])
        mapping[i] = np.argmin(diff)
    return mapping

def histogram_matching_img2(cdf1, cdf2):
    return np.interp(cdf1, cdf2, np.arange(256)).astype(np.uint8)
    
def convolution(img, new):
    return new[img]

# operation work
def operation(src, ref):
    #Histogram
    hist1 = histogram(src)
    hist2 = histogram(ref)
    
    #PDF
    pdf1 = pdf(hist1)
    pdf2 = pdf(hist2)
    
    #CDF
    cdf1 = cdf(pdf1)
    cdf2 = cdf(pdf2)
    
    # Histogram matching built-in
    matching_img_buit_in = histogram_matching_built_in(src, ref)
    matching_hist_built_in = histogram(matching_img_buit_in)
    
    # Histogram matching way1
    mapping1 = histogram_matching_img1(cdf1, cdf2)
    matching_img1 = convolution(src, mapping1)
    matching_histogram1 = histogram(matching_img1)
    
    # Histogram matching way2 
    mapping2 = histogram_matching_img2(cdf1, cdf2)
    matching_img2 = convolution(src, mapping2)
    matching_histogram2 = histogram(matching_img2)
    
    img_set = [src, ref, matching_img_buit_in,matching_img1,matching_img2, hist1, hist2, matching_hist_built_in,matching_histogram1, matching_histogram2]
    titles = ["Source-Image", "Reference-Image","matching_img_buit_in","matching_img1","matching_img2", "Source-Histogram", "Reference-Histogram","Histogram_matching_built-in","Histogram-Matching1", "Histogram-Matching2"]
    
    display(img_set, titles)

def main():
    source = cv2.imread("/home/abraham/Desktop/Abraham/CSE4-1/DIP_Lab/Image/dog.jpg", 0)
    reference = cv2.imread("/home/abraham/Desktop/Abraham/CSE4-1/DIP_Lab/Image/img1.jpg", 0)
    third = cv2.imread("/home/abraham/Desktop/Abraham/CSE4-1/DIP_Lab/Image/rose.jpeg", 0)
    
    case = [
        ("Source vs reference", source, reference),
        ("Source vs third", source, third),
        ("Source vs Source", source, source),
        ]
    
    for title, src, ref in case :
        operation(src, ref)


if __name__ == "__main__":
    main()
