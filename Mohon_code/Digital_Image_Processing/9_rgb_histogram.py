'''
Problem Statemet:
You are given a rgb image. Do the following:
1. Plot the original image.
2. Plot the red channel of the image.
3. Plot the green channel of the image.
4. Plot the blue channel of the image.
5. Plot the histogram of each channel.
'''
#================= Importing necessary libraries ======================
import matplotlib.pyplot as plt
import numpy as np
import cv2


def main():
    #================= Load the image =================================
    # Change the path to your image file
    img_path = "/home/mohon/4_1/code/dip_code/images/rgb_flower1.png"
    img_3D = cv2.imread(img_path)
    # Convert the image from BGR to RGB format
    img_rgb = cv2.cvtColor(img_3D, cv2.COLOR_BGR2RGB)
    
    #================= separate the channels =========================
    # Extract the red, green, and blue channels
    img_red = img_rgb[:, :, 0]
    img_green = img_rgb[:, :, 1]
    img_blue = img_rgb[:, :, 2]


    #================= Plotting the images ============================
    plt.figure(figsize=(25,22))
    plt.subplot(4,2,1)
    plt.imshow(img_rgb)  

    plt.subplot(4,2,2)
    plt.imshow(img_red, cmap = "Reds")
    
    plt.subplot(4,2,3)
    plt.imshow(img_green, cmap = 'Greens')
    
    plt.subplot(4,2,4)
    plt.imshow(img_blue, cmap = 'Blues')
    
    
    #================= Calculate histograms =========================
    red_histogram = histogram(img_red)
    green_histogram = histogram(img_green)
    blue_histogram = histogram(img_blue)

    #================= Plotting the histograms ======================
    plt.subplot(4,2,5)
    plt.plot(range(256),red_histogram, color='red')
    
    plt.subplot(4,2,6)
    plt.plot(range(256), green_histogram, color = 'green')
    
    plt.subplot(4,2,7)
    plt.plot(range(256), blue_histogram, color = 'blue')
    

    plt.tight_layout()
    plt.show()
    plt.close()

    

#================= Function to calculate histogram ==================
def histogram(img_2D):
    h, w = img_2D.shape
    hist = np.zeros(256, dtype = int)  


    for i in range(h):
        for j in range(w):
            pixel_value = img_2D[i,j]
            hist[pixel_value] += 1

    print(hist)
    return hist


#================= Main function to run the script =================
if __name__ == "__main__":
    main()