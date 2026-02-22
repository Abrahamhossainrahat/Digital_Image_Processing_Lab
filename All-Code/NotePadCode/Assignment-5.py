""" Write a report based on your observations after performing spatial filtering using OpenCV’s
built-in function cv2.filter2D() with:
●​ A smoothing or average kernel (e.g., a kernel with all elements equal to 1)​
●​ A Sobel kernel in the x-direction and a Sobel kernel in the y-direction​
●​ A Prewitt kernel in the x-direction and a Prewitt kernel in the y-direction​
●​ A Laplace kernel​ """


import matplotlib.pyplot as plt
import numpy as np
import cv2


def main():
    img = cv2.imread("/home/abraham/Desktop/Abraham/CSE4-1/DIP_Lab/Image/img1.jpg", 0)
    
    # kernel 
    # Average Kernel 
    avg_kernel = (1/9) * np.array([[1,1,1],
                            [1,1,1],
                            [1,1,1]], dtype=np.float32)
    # Sobel Kernel
    sobel_x = np.array([[-1,0,1],
                        [-2,0,2],
                        [-1,0,1]], dtype=np.float32)
                        
    sobel_y = np.array([[1,2,1],
                        [0,0,0],
                        [-1,-2,-1]], dtype=np.float32)
                        
    # Prewitt kerbel
    prewitt_x = np.array([[-1,0,1],
                        [-1,0,1],
                        [-1,0,1]], dtype=np.float32)
                        
    prewitt_y = np.array([[1,1,1],
                        [0,0,0],
                        [-1,-1,-1]], dtype=np.float32)
                        
    # Laplasian kernel 
    laplasian_kernel = np.array([[0,-1,0],
                        [-1,4,-1],
                        [0,-1,0]], dtype=np.float32)
               
    # Filterring 
    sobel_x_img = cv2.filter2D(img, -1, sobel_x)
    sobel_y_img = cv2.filter2D(img, -1, sobel_y)
    prewitt_x_img = cv2.filter2D(img, -1, prewitt_x)
    prewitt_y_img = cv2.filter2D(img, -1, prewitt_y)
    laplasian_img = cv2.filter2D(img, -1, laplasian_kernel)
    smooth_img = cv2.filter2D(img, -1, avg_kernel)
    
    
    img_set = [img, sobel_x_img, sobel_y_img, prewitt_x_img, prewitt_y_img, laplasian_img, smooth_img ]
    titles = ["Original_Img", "sobel_x", "sobel_y", "prewitt_x", "prewitt_y", "laplasian_img", "smooth_img" ]
    
  
    plt.figure(figsize= (12,8))
    for i in range(len(img_set)):
        plt.subplot(3,3,i+1)
        plt.imshow(img_set[i], cmap = 'gray')
        plt.title(titles[i])
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    




if __name__ == "__main__":
    main()
