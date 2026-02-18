import numpy as np
import cv2
import matplotlib.pyplot as plt

img_size = 64

def main():
    # Load image in grayscale
    img_path = '/home/mohon/4_1/lab/dip_lab/images/dft_sample_1.png'
    image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    # Resize for faster computation
    image = cv2.resize(image, (img_size, img_size))

    # Compute 2D FFT
    fft_result = fft_2d(image)

    # Compute magnitude spectrum
    magnitude = np.abs(fft_result)
    magnitude_log = np.log(1 + magnitude)

    # Shift zero frequency to center
    magnitude_shifted = manual_fftshift(magnitude_log)

    img_set = [image, magnitude_shifted]
    titles = ['Original Image', 'Magnitude Spectrum (Manual FFT)']

    display(img_set, titles)

    input_images = [
        cv2.imread('/home/mohon/4_1/lab/dip_lab/images/dft_sample_2.png', cv2.IMREAD_GRAYSCALE),
        cv2.imread('/home/mohon/4_1/lab/dip_lab/images/dft_sample_3.png', cv2.IMREAD_GRAYSCALE),
        cv2.imread('/home/mohon/4_1/lab/dip_lab/images/dft_sample_4.png', cv2.IMREAD_GRAYSCALE),
        cv2.imread('/home/mohon/4_1/lab/dip_lab/images/dft_sample_5.png', cv2.IMREAD_GRAYSCALE)
    ]
    dft_magnitudes = [] 

    for img in input_images:
        img_resized = cv2.resize(img, (img_size, img_size))
        fft_res = fft_2d(img_resized)
        mag = np.abs(fft_res)
        mag_log = np.log(1 + mag)
        mag_shifted = manual_fftshift(mag_log)
        dft_magnitudes.append(mag_shifted)

    display_2(input_images, dft_magnitudes)
   


# ---------- FFT IMPLEMENTATION ----------

def fft_1d(x):
    """
    Recursive Cooley-Tukey FFT for 1D array.
    Input length must be a power of 2.
    """
    x = np.asarray(x, dtype=complex)
    N = x.shape[0]
    if N <= 1:
        return x
    if N % 2 != 0:
        raise ValueError("Length of input must be a power of 2")

    # Divide: even and odd parts
    X_even = fft_1d(x[::2])
    X_odd = fft_1d(x[1::2])

    # Combine
    factor = np.exp(-2j * np.pi * np.arange(N) / N)
    return np.concatenate([
        X_even + factor[:N // 2] * X_odd,
        X_even + factor[N // 2:] * X_odd
    ])


def fft_2d(image):
    """
    Compute 2D FFT by applying 1D FFT on rows, then on columns.
    """
    image = np.asarray(image, dtype=float)
    M, N = image.shape

    # Pad to next power of 2 for efficiency
    M2 = 1 << (M - 1).bit_length()
    N2 = 1 << (N - 1).bit_length()
    padded = np.zeros((M2, N2))
    padded[:M, :N] = image

    # FFT along rows
    fft_rows = np.array([fft_1d(row) for row in padded])
    # FFT along columns
    fft_full = np.array([fft_1d(col) for col in fft_rows.T]).T
    return fft_full


def manual_fftshift(matrix):
    """
    Manually shift the zero-frequency component to the center of the spectrum.
    Equivalent to np.fft.fftshift().
    """
    rows, cols = matrix.shape
    row_half = rows // 2
    col_half = cols // 2

    top_left = matrix[:row_half, :col_half]
    top_right = matrix[:row_half, col_half:]
    bottom_left = matrix[row_half:, :col_half]
    bottom_right = matrix[row_half:, col_half:]

    top = np.hstack((bottom_right, bottom_left))
    bottom = np.hstack((top_right, top_left))
    shifted = np.vstack((top, bottom))

    return shifted

def display(img_set, titles, cols = 2):
    rows = (len(img_set) + cols - 1) // cols

    for i in range(len(img_set)):        
        plt.subplot(rows, cols, i + 1)  
        if(img_set[i].ndim == 2):    # show gray-scale image 
            plt.imshow(img_set[i], cmap = 'gray')
            plt.title(titles[i])  
            plt.axis('off') 
        else:   # show histogram
            plt.bar(range(256), img_set[i])
            plt.title(titles[i])
    
    plt.tight_layout()
    plt.show()

def display_2(input_images,dft_magnitudes, cols = 4):
    total = len(input_images) + len(dft_magnitudes)
    rows = (total + cols - 1) // cols
    plt.figure(figsize=(12, 6))

    for i in range(len(input_images)):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(input_images[i], cmap='gray')
        plt.title(f'Input Image {i+1}')
        plt.axis('off')

    for i in range(len(dft_magnitudes)):
        plt.subplot(rows, cols, i + 1 + len(input_images))
        plt.imshow(dft_magnitudes[i], cmap='gray')
        plt.title(f'DFT Magnitude {i+1}')
        plt.axis('off')

    plt.tight_layout()
    plt.show()

    

    

# ---------- Run main ----------
if __name__ == "__main__":
    main()
