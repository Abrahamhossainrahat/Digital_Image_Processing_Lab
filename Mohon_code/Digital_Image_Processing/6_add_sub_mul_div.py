'''
Problem Statement:
Write a Python program to perform addition, subtraction, multiplication, and division on two images.
'''

import cv2
import matplotlib.pyplot as plt

#================ Load the images ================================================================
img1 = cv2.cvtColor(cv2.imread("/home/mohon/4_1/cse4161/images/rgb_flower.png"), cv2.COLOR_BGR2RGB)
img2 = cv2.cvtColor(cv2.imread("/home/mohon/4_1/cse4161/images/rgb_flower1.png"), cv2.COLOR_BGR2RGB)

#================ Resize the second image to match the first image dimensions ====================
img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
#cv.resize(img_source, (new_width, new_height))

#================ Perform arithmetic operations on the images ==================================
added      = cv2.add(img1, img2)
subtracted = cv2.subtract(img1, img2)
multiplied = cv2.multiply(img1, img2)
divided    = cv2.divide(img1, img2)

#================ Display the original and resulting images ====================================
titles = ['Image 1', 'Image 2', 'Addition', 'Subtraction', 'Multiplication', 'Division']
images = [img1, img2, added, subtracted, multiplied, divided]

plt.figure(figsize=(12, 8))
for i in range(6):
    plt.subplot(2, 3, i+1)
    plt.imshow(images[i])
    plt.title(titles[i])
   


plt.tight_layout()
plt.show()
