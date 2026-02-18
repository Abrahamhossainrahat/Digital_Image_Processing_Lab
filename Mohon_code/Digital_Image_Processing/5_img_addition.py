''' 
Problem statement: 
Write a Python script that loads a rgb image and performs the following transformations:
1. Load 2 images.
2. Add the two images together.
3. Display the original images and the resulting image.
'''
import cv2
import matplotlib.pyplot as plt

#================= Load the images =====================================
img1 = cv2.cvtColor(cv2.imread("path/to/image1.png"), cv2.COLOR_BGR2RGB)
img2 = cv2.cvtColor(cv2.imread("path/to/image2.png"), cv2.COLOR_BGR2RGB)


#================= Resize the second image to match the first ==========
img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))


#================= Add the images together ============================
result = cv2.add(img1, img2)

#================= Display the images =================================
plt.figure(figsize=(12, 6)) 
titles = ['Image 1', 'Image 2', 'Added Image']
images = [img1, img2, result]

for i in range(3):
    plt.subplot(1, 3, i+1)
    plt.imshow(images[i])
    plt.title(titles[i])
    plt.axis('off')

plt.tight_layout()
plt.show()
