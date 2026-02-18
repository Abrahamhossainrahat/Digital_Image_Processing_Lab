import matplotlib.pyplot as plt
import numpy as np
import cv2

def main():
    img = cv2.imread("/home/mohon/4_1/lab/dip_lab/images/canny.png", 0)

    #----------- step-1: noise reduction-------------------------------
    img_filtered = noise_reduction(img)

    #----------- step-2 : gradient calculation -------------------------
    (Gx, Gy, G, theta) = gradient_estimation(img_filtered)

    #----------- step-3: non-maximum supression -----------------------
    nms = non_maximum_supression(G, theta)

    #------------- step-4: double thresholding -----------------------
    thresh, weak, strong = threshold(nms, low=20, high=40)

    #-------------- step-5: hysteresis -------------------------------
    edges = hysteresis(thresh, weak, strong)
    

    





    img_set = [img, img_filtered, Gx, Gy, G, nms, edges]
    title = ['original', 'img_filtered', 'Gx', 'Gy', 'G', 'NMS', 'edges']
    display(img_set, title)



#------------  step-1 ----------------------------------------
def noise_reduction(img, kernel_size=3, sigma=0):

    #--- option-1 
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), sigma)


    #----- option-2
    avg_filter = np.array([[1, 1, 1],
                          [1, 1, 1],
                          [1, 1, 1]], dtype=np.float32) / 9

    img_filtered = filter(img, avg_filter)
    return img_filtered


def filter(input_img, kernel):
  tmp_img = input_img.astype(np.float32)
  input_h, input_w = input_img.shape
  kernel_h, kernel_w = kernel.shape
  output_h = input_h - kernel_h + 1
  output_w = input_w - kernel_w + 1

  output_img = np.zeros((output_h, output_w), dtype = np.float32)
  for h in range(output_h):
    for w in range(output_w):
      roi = tmp_img[h : h + kernel_h, w : w + kernel_w]
      output_img[h, w] = int(np.sum(roi * kernel))

  output_img = np.clip(output_img, 0, 255).astype(np.uint8)

  return output_img



#--------------- step-2 -------------------------------------
def gradient_estimation(img):
    
    sobel_x = np.array([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]], dtype=np.float32)

    sobel_y = np.array([[-1, -2, -1],
                        [0,  0,  0],
                        [1,  2,  1]], dtype=np.float32)

    
    Gx = filter(img, sobel_x)
    Gy = filter(img, sobel_y)
    G = np.hypot(Gx , Gy)
    G = ((G / G.max()) * 255).astype(np.uint8)
    theta = np.arctan2(Gy, Gx)

    return (Gx, Gy, G, theta)

#-------------  step-3 ---------------------------------------
def non_maximum_supression(G, theta):
    h, w = G.shape
    Z = np.zeros((h,w), dtype=np.uint8)
    angle = theta * 180.0 / np.pi 
    angle[angle < 0] += 180


    for i in range(1, h-1):
        for j in range(1, w-1):
            q = 255
            r = 255
            a = angle[i,j]

            if (22.5 > a) or (a >= 157.5):
                q = G[i, j+1]
                r = G[i, j-1]
            elif (22.5 <= a) and (a < 67.5):
                q = G[i-1, j+1]
                r = G[i+1, j-1]
            elif (67.5 <= a) and (a < 112.5):
                q = G[i-1, j]
                r = G[i+1, j]
            elif (112.5 <= a) and (a < 157.5):
                q = G[i-1, j-1]
                r = G[i+1, j+1]
            
            if(G[i, j] >= q) and (G[i, j] >= r):
                Z[i, j] = G[i, j]
            else:
                Z[i, j] = 0
    
    return Z


#----------------- setp-4 : implementation -------------------------
def threshold(img, low, high):
    res = np.zeros_like(img)
    strong = 255
    weak = 75

    strong_i, strong_j = np.where(img >= high)
    weak_i, weak_j = np.where((img >= low) & (img < high))

    res[strong_i, strong_j] = strong
    res[weak_i, weak_j] = weak
    return res, weak, strong

#------------------ step-5: implementation ----------------------------
def hysteresis(img, weak, strong=255):
    M, N = img.shape
    for i in range(1, M-1):
        for j in range(1, N-1):
            if img[i,j] == weak:
                if ((img[i+1, j-1] == strong) or (img[i+1, j] == strong) or (img[i+1, j+1] == strong)
                    or (img[i, j-1] == strong) or (img[i, j+1] == strong)
                    or (img[i-1, j-1] == strong) or (img[i-1, j] == strong) or (img[i-1, j+1] == strong)):
                    img[i,j] = strong
                else:
                    img[i,j] = 0
    return img







def display(img_set, img_title):

    # plt.figure(figsize=(16,12))

    for i in range(len(img_set)):
        
        plt.subplot(2, 4, i + 1)  
        if(img_set[i].ndim == 2):       
            plt.imshow(img_set[i], cmap = 'gray')
            plt.title(img_title[i])  
            plt.axis('off') 
        else:   
            plt.bar(range(256), img_set[i])
            plt.title(img_title[i])            
    
    plt.legend()
    plt.tight_layout() 
    plt.show()


if __name__ == '__main__':
    main()