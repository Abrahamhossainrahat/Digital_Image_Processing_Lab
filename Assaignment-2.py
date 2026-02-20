import matplotlib.pyplot as plt
import numpy as np 
import cv2

# Calculation for all Image Shade 
def shade(height, width, value, x):
    if x == 0:   # White Image
        white_img = np.ones((height, width, 3), dtype=np.uint8) * value
        return white_img
    elif x == 4:  # Cyan Image
        img = np.ones((height, width, 3), dtype=np.uint8) 
        img[:, :, 1] = value 
        img[:, :, 2] = value
        return img
    elif x == 5 or x == 6:    # Magenta(5)--(0,2) Image and Yellow(6)--(0,1)
        img = np.ones((height, width, 3), dtype=np.uint8) 
        img[:, :, 0] = value 
        img[:, :, 7-x] = value
        return img
    # Pure Red , Green and Blue Image 
    img = np.ones((height, width, 3), dtype=np.uint8) 
    img[:, :, x-1] = value 
    return img

#Image Read
img = cv2.imread('Image/picture.jpeg')
# Convert BGR to RGB
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)   

height, width, channels = img_rgb.shape
values = [0, 50, 127, 180, 255]        # Shade Value 

#Image Title 
img_title = ['white_img', 'gray1_img', 'gray2_img','gray3_img', 'black_img', 'red_img', 'red_img1','red_img2', 'red_img3','red_img4', 'green_img','green_img1','green_img2','green_img3', 'green_img4', 'blue_img','blue_img1','blue_img2', 'blue_img3', 'blue_img4','cyan_img','cyan_img1', 'cyan_img2', 'cyan_img3', 'cyan_img4', 'magenta_img','magenta_img1','magenta_img2','magenta_img3','magenta_img4', 'yellow_img','yellow_img1','yellow_img2','yellow_img3','yellow_img4']

#Plot Image 
plt.figure(figsize=(14,8))
for color in range(7):
    for i in range(5):
        plt.subplot(7, 5, color*5 + i+1)
        plt.imshow(shade(height, width,values[i],color))
        plt.title(img_title[color*5 + i])
        plt.axis('off')
        
plt.tight_layout()
plt.show()