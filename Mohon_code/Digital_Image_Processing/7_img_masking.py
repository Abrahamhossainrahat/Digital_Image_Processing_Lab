''''
Problem Statement:
Write a Python program to perform image masking on a 3D image. That is to say, set a specific area of the image to zero.   
'''

import matplotlib.pyplot as plt
import cv2


def main():
    img_3D = cv2.imread("/home/mohon/4_1/cse4161/images/rgb_flower.png")

    masking_area = 1000
    img_3D[:masking_area, :masking_area] = 0    

    img_3D = cv2.cvtColor(img_3D, cv2.COLOR_BGR2RGB)
    plt.imshow(img_3D)
    plt.title('Modified Image')
    plt.show()




if __name__ == "__main__":
    main()
