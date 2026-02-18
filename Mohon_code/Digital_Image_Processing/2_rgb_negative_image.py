'''  
Given an image, this script load the image, convert it from 4D to 3D, perform the given transformations:
g(x, y, c) = max(f(x, y, c)) - f(x, y, c) 

here, f(x, y, c) is the pixel value at position (x, y) for channel c = 3.
'''
import matplotlib.pyplot as plt

def main():
    #================  Load the image ========================================
    img_path = "/home/mohon/4_1/cse4161/images/rgb2.png"
    img_4D = plt.imread(img_path)


    #===============  Convert 3D from 4D =====================================
    img_3D = img_4D[:, :, :3]


    #============== perform transformation(prepare negative image)  ==========
    img_negative = img_3D.max() - img_3D


    #==============  display the original image ==============================
    plt.figure(figsize=(20, 20))
    plt.subplot(3, 2, 1)
    plt.imshow(img_3D) 

    #============= display red channel image ================================
    plt.subplot(3,2,2)
    plt.imshow(img_3D[:, :, 0], cmap = 'Reds')
    

    #============  display green channel image =============================
    plt.subplot(3, 2, 3)
    plt.imshow(img_3D[:, :, 1], cmap = 'Greens')

    #=============  dispaly the blue channel image ========================
    plt.subplot(3,2,4)
    plt.imshow(img_3D[:, :, 2], cmap = 'Blues')

    #============== display the transformed image =========================
    plt.subplot(3, 2, 5)
    plt.imshow(img_negative)


    plt.show()
    plt.close()



    
if __name__ == '__main__': 
    main()







