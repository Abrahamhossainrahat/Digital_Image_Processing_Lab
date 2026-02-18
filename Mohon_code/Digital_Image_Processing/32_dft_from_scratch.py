import cv2
import numpy as np
import matplotlib.pyplot as plt

img_size = 64

def main():
    img = cv2.imread('/home/mohon/4_1/lab/dip_lab/images/dft_sample_1.png', 0)
    img = cv2.resize(img, (img_size, img_size))  # Resize for faster computation
    dft_manual = dft_2d_manual(img)
    magnitude_spectrum_manual = np.abs(dft_manual)
    magnitude_log_manual = np.log(1 + magnitude_spectrum_manual)
    shifted_magnitude = manual_fftshift_manual(magnitude_log_manual)
    idft_manual = inverse_dft_2d_manual(dft_manual)

    dft_builtin = np.fft.fft2(img)
    magnitude_spectrum_builtin = np.abs(dft_builtin)
    magnitude_log_builtin = np.log(1 + magnitude_spectrum_builtin)
    shifted_magnitude_builtin = np.fft.fftshift(magnitude_log_builtin)
    idft_builtin = np.fft.ifft2(dft_builtin)

    display_image([img, shifted_magnitude, np.abs(idft_manual), img, shifted_magnitude_builtin, np.abs(idft_builtin)], title=['Original Image', 'Magnitude Spectrum (Manual)', 'Reconstructed Image (Manual)', 'Original Image', 'Magnitude Spectrum (Built-in)', 'Reconstructed Image (Built-in)'])

    input_img = [
        cv2.imread('/home/mohon/4_1/lab/dip_lab/images/dft_sample_2.png', 0),
        cv2.imread('/home/mohon/4_1/lab/dip_lab/images/dft_sample_3.png', 0),
        cv2.imread('/home/mohon/4_1/lab/dip_lab/images/dft_sample_4.png', 0),
        cv2.imread('/home/mohon/4_1/lab/dip_lab/images/dft_sample_5.png', 0)
    ]
    dft_magnitude = []

    for idx, image in enumerate(input_img):
        image = cv2.resize(image, (img_size, img_size))
        dft_result = dft_2d_manual(image)
        magnitude = np.abs(dft_result)
        magnitude_log = np.log(1 + magnitude)
        magnitude_shifted = manual_fftshift_manual(magnitude_log)
        dft_magnitude.append(magnitude_shifted)
    
    display_function2(input_img, dft_magnitude, cols=4)
        


def dft_1d_manual(img):
    N = img.shape[0]
    dft_result = np.zeros(N, dtype=complex)
    for k in range(N):
        sum_val = 0
        for n in range(N):
            angle = -2j * np.pi * k * n / N
            sum_val += img[n] * np.exp(angle)
        dft_result[k] = sum_val
    return dft_result

def dft_2d_manual(image):
    # Apply 1D DFT on rows
    temp = np.array([dft_1d_manual(row) for row in image], dtype=complex)
    # Apply 1D DFT on columns
    dft_result = np.array([dft_1d_manual(col) for col in temp.T], dtype=complex).T
    return dft_result

def manual_fftshift_manual(magnitude):
    rows, cols = magnitude.shape
    crow, ccol = rows // 2, cols // 2
    shifted = np.zeros_like(magnitude)
    # Swap quadrants
    shifted[:crow, :ccol] = magnitude[crow:, ccol:]
    shifted[crow:, ccol:] = magnitude[:crow, :ccol]
    shifted[:crow, ccol:] = magnitude[crow:, :ccol]
    shifted[crow:, :ccol] = magnitude[:crow, ccol:]
    return shifted

def inverse_dft_1d_manual(dft_coeffs):
    N = dft_coeffs.shape[0]
    idft_result = np.zeros(N, dtype=complex)
    for n in range(N):
        sum_val = 0
        for k in range(N):
            angle = 2j * np.pi * k * n / N
            sum_val += dft_coeffs[k] * np.exp(angle)
        idft_result[n] = sum_val / N
    return idft_result

def inverse_dft_2d_manual(dft_image):
    # Apply 1D inverse DFT on columns
    temp = np.array([inverse_dft_1d_manual(col) for col in dft_image.T], dtype=complex).T
    # Apply 1D inverse DFT on rows
    idft_result = np.array([inverse_dft_1d_manual(row) for row in temp], dtype=complex)
    return idft_result

def display_image(img, title='Image', cols = 3):
    rows = (len(img) + cols - 1) // cols
    plt.figure(figsize=(10, 5))

    for i in range(len(img)):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(img[i], cmap='gray')
        plt.title(title[i])
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()

def display_function2(input_images, dft_magnitudes, title=[], cols=3):
    total_images = len(input_images) + len(dft_magnitudes)
    rows = (total_images + cols - 1) // cols
    plt.figure(figsize=(15, 5 * rows))

    for i in range(len(input_images)):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(input_images[i], cmap='gray')
        plt.title(f'Input Image {i}')
        plt.axis('off')

    for j in range(len(dft_magnitudes)):
        plt.subplot(rows, cols, len(input_images) + j + 1)
        plt.imshow(dft_magnitudes[j], cmap='gray')
        plt.title(title[j] if title else f'DFT Magnitude {j}')
        plt.axis('off')

    plt.tight_layout()
    plt.show()
    


if __name__ == "__main__":
    main()