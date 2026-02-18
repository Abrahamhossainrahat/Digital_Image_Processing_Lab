'''
Problem Statement (Histogram Equalization):
Given a grayscale image, perform histogram equalization to enhance its contrast.
Also plot the original and equalized histograms.
'''

#================= Importing necessary libraries ======================
import matplotlib.pyplot as plt
import numpy as np
import cv2


#================= Execution workflow ==============
def main():
    img_path = "/home/mohon/4_1/lab/dip_lab/images/runway.tif"
    img_gray = cv2.imread(img_path, 0)

    #================= Histogram Equalization Pipeline =================
    hist = histogram(img_gray)                     # Original histogram
    pdf = pdf_f(hist)                              # Probability density function
    cdf = cdf_f(pdf)                               # Cumulative distribution function
    new_level = np.round(np.array(cdf) * 255).astype(np.uint8)  # Mapping new intensity levels
    new_img = img_conv(img_gray, new_level)        # Apply mapping to image
    hist_e = histogram(new_img)                    # Histogram of equalized image


    img_set = [img_gray, new_img, hist, pdf, cdf, hist_e]
    img_title = ["Original Image", "Equalized Image", "Original Histogram",
                 "PDF", "CDF", "Equalized Histogram"]
    display(img_set, img_title)


#================= Apply new intensity levels to image ================
def img_conv(img_gray, new_level):
    return new_level[img_gray]


#================= Function to calculate histogram ====================
def histogram(img_2D):
    h, w = img_2D.shape
    hist = np.zeros(256, dtype=int)

    for i in range(h):
        for j in range(w):
            pixel_value = img_2D[i, j]
            hist[pixel_value] += 1

    return hist

#================= Function to calculate PDF ==========================
def pdf_f(hist):
    return hist / hist.sum()

#================= Function to calculate CDF ==========================
def cdf_f(pdf):
    return np.cumsum(pdf)

#================= Display images and histograms ======================
def display(img_set, titles):
    plt.figure(figsize=(14, 10))  

    for i in range(len(img_set)):
        plt.subplot(3, 2, i + 1)
        if img_set[i].ndim == 2:
            plt.imshow(img_set[i], cmap="gray")    # Display image
            plt.axis('off')
        else:
            plt.bar(range(256), img_set[i])        # Display histogram or PDF/CDF
        plt.title(titles[i])

    plt.tight_layout()
    plt.show()
    plt.close()
    

#================= Main function to run the script ====================
if __name__ == "__main__":
    main()
