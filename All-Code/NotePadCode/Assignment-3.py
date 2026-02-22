import matplotlib.pyplot as plt
import numpy as np
import cv2

def bit_plan_slice(img):
    bit_planes = []
    for i in range(8): 
        shifted_img = img>>i
        lsb_img = shifted_img & 1
        slice_img = lsb_img * np.power(2,i)
        bit_planes.append(slice_img)
    return bit_planes

def display(img_set, titles):
    plt.figure(figsize= (12,8))
    
    for i in range(len(img_set)):
        plt.subplot(2,3,i+1)
        plt.imshow(img_set[i], cmap='gray')
        plt.title(titles[i])
        plt.axis('off')
        
    plt.tight_layout()
    plt.show()


def main():
    img = cv2.imread("/home/abraham/Desktop/Abraham/CSE4-1/DIP_Lab/Image/img1.jpg", 0)
    
    bit_planes = bit_plan_slice(img)
    restored_img = sum(bit_planes)
    
    img_01 = bit_planes[0] + bit_planes[1]
    img_23 = bit_planes[2] + bit_planes[3]
    img_45 = bit_planes[4] + bit_planes[5]
    img_67 = bit_planes[6] + bit_planes[7]
    
    img_set = [img, img_01, img_23, img_45, img_67, restored_img]
    titles = ["img", "img_01", "img_23", "img_45", "img_67", "restored_img"]
    display(img_set, titles)
    
if __name__ == "__main__":
    main()
