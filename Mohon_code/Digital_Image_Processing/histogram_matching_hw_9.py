'''
Problem Statement:
    applying built-in Scikit-Learn Library's function for histogram matching
    applying your two ways implemented Python functions for histogram matching
    analyzing results of two ways implemented Python functions for histogram matching
    observing effect of histogram matching for low-contrast, high-contrast and normal contrast source image
    observing effect of histogram matching for low-contrast, high-contrast and normal contrast reference image
'''

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from skimage.exposure import match_histograms

# Global counter for saving outputs
img_counter = 1

def main():
    global img_counter
    os.makedirs("/home/mohon/4_1/lab/dip_lab/outputs", exist_ok=True)

    src_path = [': low-contrast', ': normal-contrast', ': high-contrast']
    ref_path = [': low-contrast', ': normal-contrast', ': high-contrast']

    src_imgs = [
        cv2.imread("/home/mohon/4_1/lab/dip_lab/results/source_low.png", 0),
        cv2.imread("/home/mohon/4_1/lab/dip_lab/results/source_normal.png", 0),
        cv2.imread("/home/mohon/4_1/lab/dip_lab/results/source_high.png", 0)
    ]
    ref_imgs = [
        cv2.imread("/home/mohon/4_1/lab/dip_lab/results/reference_low.png", 0),
        cv2.imread("/home/mohon/4_1/lab/dip_lab/results/reference_normal.png", 0),
        cv2.imread("/home/mohon/4_1/lab/dip_lab/results/reference_high.png", 0)
    ]

    # Built-in histogram matching
    for i, ref_img in enumerate(ref_imgs):
        for j, src_img in enumerate(src_imgs):
            matched_img = match_histograms(src_img, ref_img).astype(np.uint8)
            img_set = [src_img, ref_img, matched_img,
                       histogram(src_img), histogram(ref_img), histogram(matched_img)]
            img_title = [f"Source{src_path[j]}", f"Reference{ref_path[i]}", "Matched (built-in)",
                         "Source Hist", "Reference Hist", "Matched Hist"]
            show_and_save(img_set, img_title)

    # Own implementation 1
    for i, ref_img in enumerate(ref_imgs):
        for j, src_img in enumerate(src_imgs):
            matched_img = histogram_matching_1(src_img, ref_img).astype(np.uint8)
            img_set = [src_img, ref_img, matched_img,
                       histogram(src_img), histogram(ref_img), histogram(matched_img)]
            img_title = [f"Source{src_path[j]}", f"Reference{ref_path[i]}", "Matched (Direct mapping)",
                         "Source Hist", "Reference Hist", "Matched Hist"]
            show_and_save(img_set, img_title)

    # Own implementation 2
    for i, ref_img in enumerate(ref_imgs):
        for j, src_img in enumerate(src_imgs):
            matched_img = histogram_matching_2(src_img, ref_img).astype(np.uint8)
            img_set = [src_img, ref_img, matched_img,
                       histogram(src_img), histogram(ref_img), histogram(matched_img)]
            img_title = [f"Source{src_path[j]}", f"Reference{ref_path[i]}", "Matched (inverse mapping)",
                         "Source Hist", "Reference Hist", "Matched Hist"]
            show_and_save(img_set, img_title)

    print("All outputs saved in 'outputs/' folder")
    input("Press Enter to exit...")


#================= histogram matching using direct mapping =================
def histogram_matching_1(src, ref):
    src_hist = histogram(src)
    ref_hist = histogram(ref)
    src_pdf = pdf_f(src_hist)
    ref_pdf = pdf_f(ref_hist)
    src_cdf = cdf_f(src_pdf)
    ref_cdf = cdf_f(ref_pdf)
    mapping = np.zeros(256, dtype=np.uint8)

    for src_val in range(256):
        diff = np.abs(ref_cdf - src_cdf[src_val])
        mapping[src_val] = np.argmin(diff)

    matched = mapping[src]
    return matched

def pdf_f(hist):
    return hist / hist.sum()

def cdf_f(pdf):
    return np.cumsum(pdf)
#=========================== end of own implementation-1=========================

#================= histogram matching using equalization and inverse mapping =================
def histogram_matching_2(src, ref):
    T_r = compute_equalization_map(src)
    equalized_img = T_r[src]

    G_z = compute_reference_transform(ref)
    G_inv = invert_reference_transform(G_z)

    matched_img = G_inv[equalized_img]
    return matched_img

def compute_equalization_map(img):
    hist = histogram(img)
    pdf = hist / hist.sum()
    cdf = np.cumsum(pdf)
    return np.round(cdf * 255).astype(np.uint8)

def compute_reference_transform(img):
    hist = histogram(img)
    pdf = hist / hist.sum()
    cdf = np.cumsum(pdf)
    return np.round(cdf * 255).astype(np.uint8)

def invert_reference_transform(G_z):
    G_inv = np.zeros(256, dtype=np.uint8)
    for s_val in range(256):
        diff = np.abs(G_z - s_val)
        G_inv[s_val] = np.argmin(diff)
    return G_inv
#================= end of own implementation-2 =========================

def histogram(img):
    return np.bincount(img.ravel(), minlength=256)

def show_and_save(imgs, titles):
    global img_counter


    plt.figure(figsize=(12, 6))
    for i in range(len(imgs)):
      
        plt.subplot(2, 3, i + 1)
        if imgs[i].ndim == 2:
            plt.imshow(imgs[i], cmap='gray')
            plt.axis('off')
        else:
            plt.bar(range(256), imgs[i], width=1.0)
        plt.title(titles[i])
    plt.tight_layout()

    # # Save file
    # filename = f"/home/mohon/4_1/lab/dip_lab/outputs/{img_counter}.png"
    # plt.savefig(filename)
    # print(f"Saved {filename}")
    # img_counter += 1

    # Show image
    plt.show(block=False)
    plt.pause(1)
    


if __name__ == "__main__":
    main()
