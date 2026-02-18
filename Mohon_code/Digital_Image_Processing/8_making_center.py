'''
Problem statement: 
You are given a 3D image and you need to masking the center area of 
the image i.e. set a specific area of the image to zero.
'''
#================== Import necessary libraries =========================
import matplotlib.pyplot as plt
import cv2

def main():
    #====================  read the image using cv2 =========================
    img_path = "/home/mohon/4_1/code/dip_code/images/rgb_flower1.png"
    img_3D = cv2.imread(img_path)

    #====================  convert the image to RGB =========================
    img_3D = cv2.cvtColor(img_3D, cv2.COLOR_BGR2RGB)
    #====================  set the size of maksing length ===================
    mask_size = 100
    
    #====================  set the center area of the image to zero =========
    # Resize the image to ensure the masking area is within bounds
    img_3D = cv2.resize(img_3D, (1000 + mask_size, 1000+mask_size))
    img_3D[500:501+mask_size, 500:501+mask_size] = 0

    #====================  display the modified image ======================
    plt.figure(figsize=(8, 8))
    plt.title('Modified Image with Masking')
    plt.imshow(img_3D)
    plt.show()
    plt.close()



if __name__ == "__main__":
    main()