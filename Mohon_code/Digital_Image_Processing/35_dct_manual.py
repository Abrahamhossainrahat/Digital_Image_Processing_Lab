import numpy as np
import cv2
import matplotlib.pyplot as plt


def main():
    img = cv2.imread("/home/mohon/4_1/lab/dip_lab/images/lena.png", cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError("Image not found.")
    img = cv2.resize(img, (128, 128))  # Resize for manageable computation
    dct_coeffs = dct2_separable(img)    
    dct_magnitudes = np.log(np.abs(dct_coeffs) + 1)

    img_set = [img, dct_magnitudes]
    titles = ['Original Image', 'DCT magnitude']
    display(img_set, titles)   
    

    input_images = [
        cv2.imread('/home/mohon/4_1/lab/dip_lab/images/dft_sample_2.png', cv2.IMREAD_GRAYSCALE),
        cv2.imread('/home/mohon/4_1/lab/dip_lab/images/dft_sample_3.png', cv2.IMREAD_GRAYSCALE),
        cv2.imread('/home/mohon/4_1/lab/dip_lab/images/dft_sample_4.png', cv2.IMREAD_GRAYSCALE),    
        cv2.imread('/home/mohon/4_1/lab/dip_lab/images/dft_sample_5.png', cv2.IMREAD_GRAYSCALE)
    ]
    dct_magnitudes = []
    for img in input_images:
        img_resized = cv2.resize(img, (128, 128))
        dct_res = dct2_separable(img_resized)
        dct_magnitudes.append(np.log(np.abs(dct_res) + 1))
    
    display(input_images + dct_magnitudes, 
            ['Input Image 1', 'Input Image 2', 'Input Image 3', 'Input Image 4',
             'DCT magnitude 1', 'DCT magnitude 2', 'DCT magnitude 3', 'DCT magnitude 4'],
            cols=4)



def display(img_set, titles, cols=4):
    rows = (len(img_set) + cols - 1) // cols
    plt.figure(figsize=(15, 4 * rows))

    for i, img in enumerate(img_set):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(img, cmap='gray')
        plt.title(titles[i])
        plt.axis('off')
        
    plt.tight_layout()
    plt.show()

def dct_1d(x):
    """Manual 1D DCT-II."""
    N = len(x)
    X = np.zeros(N)
    for k in range(N):
        alpha = np.sqrt(1/N) if k == 0 else np.sqrt(2/N)
        sum_val = 0.0
        for n in range(N):
            sum_val += x[n] * np.cos(np.pi * (n + 0.5) * k / N)
        X[k] = alpha * sum_val
    return X

def dct2_separable(img):
    """2D DCT via row then column transforms."""
    # Apply 1D DCT on rows
    dct_rows = np.array([dct_1d(row) for row in img])
    # Apply 1D DCT on columns
    dct_cols = np.array([dct_1d(col) for col in dct_rows.T]).T
    return dct_cols




if __name__ == "__main__":
    main()
