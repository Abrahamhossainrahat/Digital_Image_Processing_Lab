"""
Assignment:
    1. Transforming 5 images, having their low contrast, high-contrast and 
       normal contrast versions, from spatial domain to frequency domain by 
       Fast Discrete Fourier Transform (FDFT). Observe specific styles or 
       differences in frequency domain for different contrasts.
    2. Apply low-pass, high-pass, and band-pass filtering on the frequency 
       domain of low-, normal-, and high-contrast images.
"""

import matplotlib.pyplot as plt
import cv2
import numpy as np
import os

# PARAMETERS 
IMG_FOLDER = "./outputs"         # input folder
SAVE_FOLDER = "../dft_outputs" # output folder for saved plots
IMAGE_IDS = [1, 2, 3, 4, 5]   # process image sets

# MAIN FUNCTION 
def main():
    os.makedirs(SAVE_FOLDER, exist_ok=True)

    for idx in IMAGE_IDS:
        print(f"\nProcessing image set {idx} ...")

        versions = {
            "low": f"dog_{idx}_low_contrast.jpg",
            "normal": f"dog_{idx}_normal.jpg",
            "high": f"dog_{idx}_high_contrast.jpg"
        }

        for level, filename in versions.items():
            path = os.path.join(IMG_FOLDER, filename)
            if not os.path.exists(path):
                print(f"File not found: {path}")
                continue

            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            dft_analysis(img, level, idx)

# DFT ANALYSIS 
def dft_analysis(img_gray, level, idx):
    rows, cols = img_gray.shape
    min_dim = min(rows, cols)

    # Adaptive radii
    radius_low = int(min_dim * 0.25)
    radius_high = int(min_dim * 0.10)
    radius_low = max(10, radius_low)
    radius_high = max(5, radius_high)

    dft_shift, magnitude_dft_original, img_back_original = dft(img_gray)
    img_back_low, magnitude_low = low_pass_filtering(dft_shift, radius_low)
    img_back_high, magnitude_high = high_pass_filtering(dft_shift, radius_low)
    img_back_band, magnitude_band = band_pass_filtering(dft_shift, radius_low, radius_high)

    # Save results as an image
    display_and_save_results(
        img_gray, magnitude_dft_original, img_back_original,
        magnitude_low, img_back_low,
        magnitude_high, img_back_high,
        magnitude_band, img_back_band,
        idx, level
    )

# DFT 
def dft(img_gray):
    dft = np.fft.fft2(img_gray)
    dft_shift = np.fft.fftshift(dft)
    magnitude = np.log(np.abs(dft_shift) + 1)
    idft = np.fft.ifft2(np.fft.ifftshift(dft_shift))
    img_back = np.abs(idft)
    return dft_shift, magnitude, img_back

# FILTERING 
def low_pass_filtering(dft_shift, radius):
    mask = low_pass_mask(dft_shift.shape, radius)
    low_passed = dft_shift * mask
    magnitude = np.log(np.abs(low_passed) + 1)
    img_back = np.abs(np.fft.ifft2(np.fft.ifftshift(low_passed)))
    return img_back, magnitude

def high_pass_filtering(dft_shift, radius):
    mask = high_pass_mask(dft_shift.shape, radius)
    high_passed = dft_shift * mask
    magnitude = np.log(np.abs(high_passed) + 1)
    img_back = np.abs(np.fft.ifft2(np.fft.ifftshift(high_passed)))
    return img_back, magnitude

def band_pass_filtering(dft_shift, r_low, r_high):
    mask = band_pass_mask(dft_shift.shape, r_low, r_high)
    band_passed = dft_shift * mask
    magnitude = np.log(np.abs(band_passed) + 1)
    img_back = np.abs(np.fft.ifft2(np.fft.ifftshift(band_passed)))
    return img_back, magnitude

# MASKS 
def low_pass_mask(shape, radius):
    rows, cols = shape
    crow, ccol = rows // 2, cols // 2
    mask = np.zeros((rows, cols), np.uint8)
    cv2.circle(mask, (ccol, crow), radius, 1, thickness=-1)
    return mask

def high_pass_mask(shape, radius):
    return 1 - low_pass_mask(shape, radius)

def band_pass_mask(shape, r_low, r_high):
    return low_pass_mask(shape, r_high) - low_pass_mask(shape, r_low)

#DISPLAY & SAVE 
def display_and_save_results(img, mag_ori, img_ori,
                             mag_low, img_low,
                             mag_high, img_high,
                             mag_band, img_band,
                             idx, level):

    os.makedirs(SAVE_FOLDER, exist_ok=True)
    title = f"Image {idx} - {level.capitalize()} Contrast"

    plt.figure(figsize=(14, 10))
    plt.suptitle(title, fontsize=16, color='purple')

    images = [
        (img, "Original Image"),
        (mag_ori, "Magnitude Spectrum (Original)"),
        (img_ori, "Reconstructed (Original)"),
        (mag_low, "Magnitude Spectrum (Low-pass)"),
        (img_low, "Reconstructed (Low-pass)"),
        (mag_high, "Magnitude Spectrum (High-pass)"),
        (img_high, "Reconstructed (High-pass)"),
        (mag_band, "Magnitude Spectrum (Band-pass)"),
        (img_band, "Reconstructed (Band-pass)")
    ]

    for i, (im, label) in enumerate(images):
        plt.subplot(3, 3, i + 1)
        plt.imshow(im, cmap='gray')
        plt.title(label, fontsize=10)
        plt.axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    # Save the figure
    save_path = os.path.join(SAVE_FOLDER, f"image_{idx}_{level}_dft.png")
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    print(f"Saved figure: {save_path}")

    plt.close()
    # plt.show()
    
if __name__ == "__main__":
    main()