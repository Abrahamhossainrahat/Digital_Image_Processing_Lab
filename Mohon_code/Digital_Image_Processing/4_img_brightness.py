'''
Problem Statement:
Write a python script that loads an RGB image, performs the following transformations:
g(x,y,c) = f(x,y,c) + constant
constant = 10

'''

import matplotlib.pyplot as plt
import cv2


def main():
    #===================== Load the image =========================
    img_path = "/home/mohon/4_1/cse4161/images/rgb_flower1.png"
    img_3D = cv2.imread(img_path)    
    img_rgb = cv2.cvtColor(img_3D, cv2.COLOR_BGR2RGB)
    constant = 50

    #===================== Apply transformation =========================  
    img_rgb_transformed = cv2.add(img_rgb, constant)
    

    #===================== Display the images =========================
    plt.figure(figsize=(10, 10))
    plt.subplot(2, 2, 1)
    plt.imshow(img_rgb)
    plt.title('Original RGB Image') 

    plt.subplot(2, 2, 2)
    plt.imshow(img_rgb_transformed)
    plt.title('Transformed RGB Image')
    
    plt.tight_layout()
    plt.show()
    plt.close()


if __name__ == "__main__":
    main()
