import matplotlib.pyplot as plt
import numpy as np
import cv2

def main():
    # Load source and reference images
    src_path = "/home/mohon/4_1/lab/dip_lab/images/runway.tif"
    ref_path = "/home/mohon/4_1/lab/dip_lab/images/rgb6.png"
    src_img = cv2.imread(src_path, 0)
    ref_img = cv2.imread(ref_path, 0)

    # histogram
    src_hist = histogram(src_img)
    ref_hist = histogram(ref_img)

    #pdf
    src_pdf = pdf_f(src_hist)
    ref_pdf = pdf_f(ref_hist)

    #cdf
    src_cdf = cdf_f(src_pdf)
    ref_cdf = cdf_f(ref_pdf)

    
    mapping = match_histogram(src_cdf, ref_cdf)

    # Apply mapping
    matched_img = img_conv(src_img, mapping)
    matched_hist = histogram(matched_img)

    # Display results
    img_set = [src_img, ref_img, matched_img, src_hist, ref_hist, src_cdf, ref_cdf, matched_hist]
    img_title = ["Source Image", "Reference Image", "Matched Image",
                 "Source Histogram", "Reference Histogram", "Source cdf", "Reference cdf", "Matched Histogram"]
    display(img_set, img_title)

#================= Histogram Matching Mapping =========================
def match_histogram(src_cdf, ref_cdf):
    mapping = np.zeros(256, dtype=np.uint8)
    for src_val in range(256):
        diff = np.abs(ref_cdf - src_cdf[src_val])
        mapping[src_val] = np.argmin(diff)
    return mapping

#================= Apply Mapping to Image =============================
def img_conv(img_gray, mapping):
    return mapping[img_gray]

#================= Histogram Calculation ==============================
def histogram(img_2D):
    h, w = img_2D.shape
    hist = np.zeros(256, dtype=int)
    for i in range(h):
        for j in range(w):
            hist[img_2D[i, j]] += 1
    return hist

#================= PDF and CDF ========================================
def pdf_f(hist):
    return hist / hist.sum()

def cdf_f(pdf):
    return np.cumsum(pdf)

#================= Display Function ===================================
def display(img_set, titles):
    plt.figure(figsize=(14, 10))
    for i in range(len(img_set)):
        plt.subplot(3, 3, i + 1)
        if img_set[i].ndim == 2:
            plt.imshow(img_set[i], cmap="gray")
            plt.axis('off')
        else:
            plt.bar(range(256), img_set[i])
        plt.title(titles[i])
    plt.tight_layout()
    plt.show()
    plt.close()

#================= Run Script =========================================
if __name__ == "__main__":
    main()
