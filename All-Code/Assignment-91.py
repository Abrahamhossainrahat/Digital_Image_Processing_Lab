import matplotlib.pyplot as plt
import numpy as np
import cv2
from skimage import exposure

def display(img, titles, label):
    plt.figure(figsize = (12, 8))
    plt.title(label)
    for i in range(len(img)):
        plt.subplot(3,3,i+1)
        if i<3:
            plt.imshow(img[i], cmap = 'gray')
            plt.axis('off')
        else:
            plt.bar(range(256), img[i])
            
        plt.title(titles[i])
        
    plt.tight_layout()
    plt.show()

def histogram(img):
    h, w = img.shape
    
    hist = np.zeros(256, np.uint8)
    for i in range(h):
        for j in range(w):
            value = img[i, j]
            hist[value] += 1
    return hist

def pdf(hist):
    return hist/sum(hist)

def cdf(pdf):
    return np.cumsum(pdf)

def histogram_match_built_in(src_img, ref_img):
    return exposure.match_histograms(src_img, ref_img, channel_axis = None).astype(np.uint8)


def histogram_match1(src_cdf, ref_cdf):
    match = np.zeros(256, np.uint8)
    for index in range(256): 
        dif = np.abs(src_cdf - ref_cdf[index])
        match[index] = np.argmin(dif)
    return match

def histogram_mach2(src_cdf, ref_cdf):
    mapping = np.interp(src_cdf, ref_cdf, np.arange(256)).astype(np.uint8)
    return mapping
   
def convolution(img, new_level):
    return new_level[img]

def operation(src_img, ref_img, label):
    
    hist1 = histogram(src_img)
    hist2 = histogram(ref_img)
    
    pdf1 = pdf(hist1)
    pdf2 = pdf(hist2)
    
    cdf1 = cdf(pdf1)
    cdf2 = cdf(pdf2)
    
    maching = histogram_match1(cdf1, cdf2)
    maching_img = convolution(src_img, maching)
    hist_match = histogram(maching_img)
    
    img_mach_built_in = histogram_match_built_in(src_img, ref_img)
    hist_built_in = histogram(img_mach_built_in)
        
    maching2 = histogram_mach2(cdf1, cdf2)
    maching_img2 = convolution(src_img, maching2)
    hist_mach2 = histogram(maching_img2)
    
    # img_set = [src_img, ref_img, maching_img, hist1, hist2,pdf1, pdf2, cdf1, cdf2, maching, hist_match]
    # titles = ["Source_Img", "Ref_Img","Maching_Img", "Source_Hist", "Ref_Hist", "Src_PDF", "Ref_PDF", "Src_CDF", "Ref_CDF","Maching","Histogram_Maching"]
    img_set = [src_img, ref_img, maching_img, hist1, hist2, hist_match, hist_built_in, hist_mach2]
    titles = ["Source_Img", "Ref_Img","Maching_Img", "Source_Hist", "Ref_Hist", "Histogram_Maching", "Built-in_Histogram_Maching", "Histogram_Maching2"]
    
    display(img_set, titles, label)
    
    # plt.imshow(ref_img)
    # plt.show()

def main():
    src_img = cv2.imread("/home/abraham/Desktop/Abraham/CSE4-1/DIP_Lab/Image/dog.jpg", 0)
    ref_img = cv2.imread("/home/abraham/Desktop/Abraham/CSE4-1/DIP_Lab/Image/img1.jpg", 0)
    third_img = cv2.imread("/home/abraham/Desktop/Abraham/CSE4-1/DIP_Lab/Image/rose.jpeg", 0)
    
    case = [
        ("Source vs Reference", src_img, ref_img),
        ("Source vs Third", src_img, third_img),

    ]
    
    #print(third_img)
    for label, source, reference in case:
        operation(source, reference, label)
    
if __name__ == "__main__":
    main()