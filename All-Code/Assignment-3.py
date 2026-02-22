import matplotlib.pyplot as plt
import numpy as np
import cv2

def bit_plan_slicing(img):
    bit_plans = []
    
    for i in range(8):    
        shifted_img = img >> i
        lsb_bit = shifted_img & 1
        slicing_img = lsb_bit * np.power(2,i)
        bit_plans.append(slicing_img)
    
    return bit_plans

def display(img_set, titles):
    plt.figure(figsize=(12,8))
    
    for i in range(len(img_set)):  
        plt.subplot(2,3,i+1)
        plt.imshow(img_set[i], cmap= 'gray')
        plt.title(titles[i])
        plt.axis('off')
        
    plt.show()
    

def main():
    img = cv2.imread("/home/abraham/Desktop/Abraham/CSE4-1/DIP_Lab/Image/img1.jpg", 0)
    
    bit_plans = bit_plan_slicing(img)
    
    plane_01 = bit_plans[0] + bit_plans[1]
    plane_23 = bit_plans[2] + bit_plans[3]
    plane_45 = bit_plans[4] + bit_plans[5]
    plane_67 = bit_plans[6] + bit_plans[7]
    
    restored_img = sum(bit_plans)
    
    img_set = [img, plane_01, plane_23, plane_45, plane_67, restored_img]
    titles = ["Original_Img", "plane_01", "plane_23", "plane_45", "plane_67", "restored_img"]
    
    display(img_set, titles)
    
    
if __name__ == '__main__':
    main()