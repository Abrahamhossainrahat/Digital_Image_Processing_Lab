import matplotlib.pyplot as plt
import cv2
import numpy as np 

# Threshold limits
limit1 = 127
limit2 = 200

# Thresholding techniques
def thresolding1(img_gray):
    img_tmp = img_gray.copy()
    for i in range(img_gray.shape[0]):
        for j in range(img_gray.shape[1]):
            img_tmp[i][j] = 0 if img_tmp[i][j] <= limit1 else 255
    return img_tmp

def thresolding2(img_gray):
    img_tmp = img_gray.copy()
    for i in range(img_gray.shape[0]):
        for j in range(img_gray.shape[1]):
            if limit1 <= img_tmp[i][j] <= limit2:
                img_tmp[i][j] = 127
    return img_tmp

def thresolding3(img_gray):
    img_tmp = img_gray.copy()
    for i in range(img_gray.shape[0]):
        for j in range(img_gray.shape[1]):
            if limit1 <= img_tmp[i][j] <= limit2:
                img_tmp[i][j] = 127 
            elif img_tmp[i][j] > limit2:
                img_tmp[i][j] = 255
    return img_tmp

# Histogram calculation
def histogram(img_gray):
    hist = np.zeros(256, dtype=int)
    for i in range(img_gray.shape[0]):
        for j in range(img_gray.shape[1]):
            hist[img_gray[i][j]] += 1
    return hist

# Display images and histograms in single figure
def display_images_and_histograms(img_set, img_title):
    n = len(img_set)
    plt.figure(figsize=(10, 6))
    for i in range(n):
        # Show image
        plt.subplot(2, n, i + 1)
        plt.imshow(img_set[i], cmap='gray')
        plt.title(img_title[i])
        plt.axis('off')

        # Show histogram
        plt.subplot(2, n, n + i + 1)
        hist = histogram(img_set[i])
        plt.bar(range(256), hist, color='gray')
        plt.title(f"Histogram {img_title[i]}")
        plt.xlabel("Pixel Value")
        plt.ylabel("Frequency")

    plt.tight_layout()
    plt.show()

# Main function
def main():
    img_path = 'Image/poddo2.jpg'  # Change path accordingly
    img_gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    # Apply thresholding
    img_th1 = thresolding1(img_gray)
    img_th2 = thresolding2(img_gray)
    img_th3 = thresolding3(img_gray)

    # Store images and titles
    img_set = [img_gray, img_th1, img_th2, img_th3]
    img_title = ["Original", "Threshold 1", "Threshold 2", "Threshold 3"]

    # Display everything in single figure
    display_images_and_histograms(img_set, img_title)

if __name__ == '__main__':
    main()
