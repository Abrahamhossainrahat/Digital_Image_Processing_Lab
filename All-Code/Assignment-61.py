import matplotlib.pyplot as plt
import numpy as np
import cv2

def display(img, titles):
    plt.figure(figsize=(12,8))
    for i in range(len(img)):
        plt.subplot(4,4,i+1)
        plt.imshow(img[i], cmap = 'gray')
        plt.title(titles[i])
        plt.axis('off')
        
    plt.show()

def convolution_valid(img, kernel):
    temp_img = img.astype(np.float32)
    height, width = img.shape
    k_height, k_width = kernel.shape
    
    output_height = height - k_height + 1
    output_width = width - k_width + 1
    
    output_img = np.zeros((output_height, output_width), dtype = np.float32)
    
    for h in range(output_height):
        for w in range(output_width):
            roi = temp_img[h:h+k_height, w:w+k_width]
            output_img[h, w] = np.sum(roi * kernel)
            
    return np.clip(output_img, 0, 255).astype(np.uint8)
        
def convolution_same(img, kernel):
    temp_img = img.astype(np.float32)
    height, width = img.shape
    k_height, k_width = kernel.shape
    output_height = height
    output_width = width
    output_img = np.zeros((output_height, output_width), dtype= np.float32)
    
    pad_h = k_height // 2
    pad_w = k_width // 2
    padded_img = np.pad(temp_img, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant', constant_values=0)
    for h in range(output_height):
        for w in range(output_width):
            roi = padded_img[h:h+k_height, w:w+k_width]
            output_img[h, w] = np.sum(roi * kernel)
            
    return np.clip(output_img, 0, 255).astype(np.uint8)

def main():
    img = cv2.imread("/home/abraham/Desktop/Abraham/CSE4-1/DIP_Lab/Image/img1.jpg", 0)
    
    # gausian noise 
    row, col = img.shape
    mean, var = 0, 0.01
    sigma = np.sqrt(var)
    
    gauss = np.random.normal(mean, sigma, (row, col))
    noisy_img = img + gauss * 255
    noisy_img = np.clip(noisy_img, 0, 255).astype(np.uint8)
    
    
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
               
    
    filters = {
        "Avg": avg_kernel,
        "sobel_x" : sobel_x,
        "sobel_y" : sobel_y,
        "prewitt_x" : prewitt_x,
        "prewitt_y" : prewitt_y,
        "lapasian" : laplasian_kernel,
    }
    
    img_set = [img, noisy_img]
    titles = ["originalImg", "Noisy_Img"]
    
    for name, kernel in filters.items():
        valid_img = convolution_valid(img, kernel)
        same_img = convolution_same(img, kernel)
        img_set.extend([valid_img, same_img])
        titles.extend([f'{name} valid', f'{name} same'])
    
    
    display(img_set, titles)
    
    
    
    
if __name__ == "__main__":
    main()