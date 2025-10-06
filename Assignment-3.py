import numpy as np
import matplotlib.pyplot as plt
import cv2

def bit_plane_slicing(img):
    bit_planes = []
    for i in range(8):
        shifted_img = img >> i
        lsb_bit = shifted_img & 1
        sliced_image = lsb_bit * np.power(2, i)
        bit_planes.append(sliced_image)
    return bit_planes

def display(img_set, title_set, color_set):
    row = 2
    col = (len(img_set) + 1) // row  # ceil division for columns
    k = 0
    for i in range(row):
        for j in range(col):
            if k >= len(img_set):
                break
            img = img_set[k]
            plt.subplot(row, col, k + 1)
            plt.title(title_set[k])
            if len(img.shape) == 3:
                plt.imshow(img)
            else:
                plt.imshow(img, cmap=color_set[k])
            plt.axis('off')
            k += 1
    plt.tight_layout()
    plt.show()

def main():
    img_path = 'Image/poddo.jpg' 
    img = cv2.imread(img_path, 0)  # grayscale
    bit_planes = bit_plane_slicing(img)

    # Combine planes for visualization
    bit_planes12 = bit_planes[0] + bit_planes[1]
    bit_planes34 = bit_planes[2] + bit_planes[3]
    bit_planes56 = bit_planes[4] + bit_planes[5]
    bit_planes78 = bit_planes[6] + bit_planes[7]

    # Proper reconstructed image
    reconstructed_image = sum(bit_planes)

    img_set = [img, bit_planes12, bit_planes34, bit_planes56, bit_planes78, reconstructed_image]
    title_set = ['Original Image', 'Plane 1+2', 'Plane 3+4', 'Plane 5+6', 'Plane 7+8', 'Reconstructed Image']
    color_set = ['gray'] * len(img_set)

    display(img_set, title_set, color_set)

if __name__ == '__main__':
    main()
