import matplotlib.pyplot as plt
import numpy as np
import cv2

def shed(height, width, value, x):
    if(x == 0):
        img = np.ones((height, width, 3), dtype = np.uint8)*value
        return img
        
    elif(x==4):
        img = np.ones((height, width, 3), dtype = np.uint8)
        img[:, :, 1] = value
        img[:, :, 2] = value
        return img
        
    elif(x==5):
        img = np.ones((height, width, 3), dtype = np.uint8)
        img[:, :, 0] = value
        img[:, :, 2] = value
        return img
    
    elif(x==6):
        img = np.ones((height, width, 3), dtype = np.uint8)
        img[:, :, 0] = value
        img[:, :, 1] = value
        return img
        
    img = np.ones((height, width, 3), dtype = np.uint8)
    img[:, :, x-1] = value
    return img

img = cv2.imread("/home/abraham/Desktop/Abraham/CSE4-1/DIP_Lab/Image/img1.jpg")
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

height, width, channel = img_rgb.shape
values = [0,50,127,200,255]

plt.figure(figsize = (12,8))
pos = 1
for i in range(7):
    for j in range(5):
        plt.subplot(7,5,pos)
        plt.axis('off')
        plt.imshow(shed(height, width, values[j],i))
        pos+=1
        
plt.show()
