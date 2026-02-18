''' 
Problem Statement: 
Given an image, this script load the image, convert it from 4D to 3D, print some pixel values
and display the image using matplotlib.
'''

import matplotlib.pyplot as plt


def main():
    #=================   Load the image ======================
    img_path = '/home/mohon/4_1/cse4161/images/rgb.png'
    img_4D = plt.imread(img_path)
    print(img_4D.shape)

    #================= Convert to 3D from 4D ======================
    img_3D = img_4D[:, :, :3]
    print(img_3D.shape)

    #================= Print some values of images ======================
    print(img_3D[:5, :5, 0])  # Print first 5x5 pixels of the red channel
    print(img_3D[:5, :5, 1])  # Print first 5x5 pixels of the green channel
    print(img_3D[:5, :5, 2])  # Print first 5x5 pixels of the blue channel
    print(img_3D.max(), img_3D.min())  # Print max and min values of the image

    #================= display the image ======================
    plt.imshow(img_3D)
    plt.show()
    plt.close()



if __name__ == '__main__':
    main()
