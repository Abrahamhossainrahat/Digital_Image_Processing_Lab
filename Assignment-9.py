import matplotlib.pyplot as plt
import numpy as np
import cv2
from skimage import exposure

# Normalization
def to_float(img):
    return img.astype(np.float32) / 255.0

# Back to image and clipping
def to_uint8(img):
    return np.clip(img * 255, 0, 255).astype(np.uint8)

# Calculate the CDF
def cdf(src):
    hist, _ = np.histogram(src.flatten(), 256, [0, 256])
    pdf = hist / hist.sum() if hist.sum() else 1
    return np.cumsum(pdf)

# Built-in histogram matching (Scikit-Image)
def builtin_match(source, reference):
    channel_axis = None if source.ndim == 2 else -1
    return exposure.match_histograms(source, reference, channel_axis=channel_axis)

# User-defined matching function 1
def match_histogram(src_cdf, ref_cdf):
    mapping = np.zeros(256, dtype=np.uint8)
    for src_val in range(256):
        diff = np.abs(ref_cdf - src_cdf[src_val])
        mapping[src_val] = np.argmin(diff)
    return mapping

# User-defined matching function 2 (using interpolation)
def match_histogram2(src_cdf, ref_cdf):
    mapping = np.interp(src_cdf, ref_cdf, np.arange(256)).astype(np.uint8)
    return mapping

# Operation for each case
def operation(src_img, ref_img, case_label):
    src_f, ref_f = to_float(src_img), to_float(ref_img)
    built_in = builtin_match(src_f, ref_f)
    src_image, ref_image, built_in = to_uint8(src_f), to_uint8(ref_f), to_uint8(built_in)

    # User Define histogram Matching 1
    src_cdf = cdf(src_img)
    ref_cdf = cdf(ref_img)
    mapping = match_histogram(src_cdf, ref_cdf)
    user_image1 = mapping[src_img]

    # User Define histogram Matching 2
    mapping2 = match_histogram2(src_cdf, ref_cdf)
    user_image2 = mapping2[src_img]

    title_set = ['Source Image', 'Reference Image', 'Built-in Matching', 'User Define 1', 'User Define 2']
    images = [src_image, ref_image, built_in, user_image1, user_image2]

    plt.figure(figsize=(8,6))
    for i in range(len(images)):
        # Left side: image
        plt.subplot(5, 2, 2 * i + 1)
        plt.imshow(images[i], cmap="gray")
        plt.title(title_set[i])
        plt.axis("off")

        # Right side: histogram
        plt.subplot(5, 2, 2 * i + 2)
        plt.hist(images[i].ravel(), bins=256, range=(0, 256), color="black")
        plt.title(title_set[i] + " Histogram")

    plt.suptitle(case_label, fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.show()

def main():
    # Load source and reference images
    low_contrast = r"E:\BSC_In_CSE\4th_Year_1st _ememster\Digital_Image_Processing\Image\lowContrast.jpg"
    high_contrast = r"E:\BSC_In_CSE\4th_Year_1st _ememster\Digital_Image_Processing\Image\highContrast.jpg"
    normal_contrast = r"E:\BSC_In_CSE\4th_Year_1st _ememster\Digital_Image_Processing\Image\mediumContrsast.jpeg"

    cases = [
        ("low contrast src vs normal contrast ref", low_contrast, normal_contrast),
        ("low contrast src vs high contrast ref", low_contrast, high_contrast),
        ("low contrast src vs low contrast ref", low_contrast, low_contrast),

        ("normal contrast src vs normal contrast ref", normal_contrast, normal_contrast),
        ("normal contrast src vs high contrast ref", normal_contrast, high_contrast),
        ("normal contrast src vs low contrast ref", normal_contrast, low_contrast),

        ("high contrast src vs normal contrast ref", high_contrast, normal_contrast),
        ("high contrast src vs high contrast ref", high_contrast, high_contrast),
        ("high contrast src vs low contrast ref", high_contrast, low_contrast),
    ]

    for label, src, ref in cases:
        src_img = cv2.imread(src, 0)
        ref_img = cv2.imread(ref, 0)
        if src_img is None or ref_img is None:
            raise FileNotFoundError(f"Check image paths:\n  src: {src}\n  ref: {ref}")
        operation(src_img, ref_img, label)

if __name__ == "__main__":
    main()
