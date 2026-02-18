'''
Problem Statement:
The task is to read an image in BGR format and convert it to RGB format manually.
'''
import matplotlib.pyplot as plt
import cv2
import numpy as np 

#======================== RGB manually Conversion =========================

def main():
    img_path = "/home/mohon/4_1/code/dip_code/images/rgb_flower1.png"
    img_3D = cv2.imread(img_path)    

    rgb_red = img_3D[:, :, 2]
    rgb_green = img_3D[:, :, 1]
    rgb_blue = img_3D[:, :, 0] 

    #===============  Way 1 =====================================
    img_rgb = np.zeros_like(img_3D)
    img_rgb[:, :, 0] = rgb_red
    img_rgb[:, :, 1] = rgb_green
    img_rgb[:, :, 2] = rgb_blue
    
    #===============  Way 2 =====================================
    #img_rgb_1 = img_3D[:, :, ::-1]


    #===============  Displaying Images =========================
    plt.figure(figsize=(10, 10))
    plt.subplot(2,2,1)
    plt.imshow(img_3D) 
    plt.title('BGR Image')   

    plt.subplot(2, 2, 2)
    plt.imshow(img_rgb)
    plt.title('RGB Image')



    plt.tight_layout()
    plt.show()
    plt.close()



if __name__ == '__main__':
    main()

