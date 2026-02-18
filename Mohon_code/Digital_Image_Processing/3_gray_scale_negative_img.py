'''
Problem Statement:
Write a Python script that loads a grayscale image and performs the following transformations:
1. Load the image.
2. Print the shape of the image.
3. Display the original image.
4. Create a negative of the image using the transformation:
    g(x, y) = max(f(x, y)) - f(x, y),
5. Display the negative image. 

'''
import matplotlib.pyplot as plt



def main():
    #================ Load the image ========================================
    img_path = "/home/mohon/4_1/cse4161/images/gray_4.png"
    img_2D = plt.imread(img_path)
    

    #=============== Print the shape ============================
    print("Image shape using matplotlib:", img_2D.shape)
    


    #============== Display the original image ==============================
    plt.figure(figsize=(10, 10))
    plt.subplot(2, 1, 1)
    plt.imshow(img_2D[:, :, 0], cmap='gray')
    plt.title('Original Grayscale Image')
   

    #============== Perform transformation (prepare negative image) ==========
    img_negative = img_2D.max() - img_2D

    #============== Display the transformed image ============================
    plt.subplot(2, 1, 2)
    plt.imshow(img_negative[:, :, 0], cmap='gray')
    plt.title('Negative Image')
    

    plt.tight_layout()
    plt.show()
    plt.close()


if __name__ == '__main__':
    main()
