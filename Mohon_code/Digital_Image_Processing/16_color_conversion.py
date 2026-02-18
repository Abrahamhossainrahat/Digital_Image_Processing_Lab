'''
Problem Statement:
Manually generate image of black, red, green, blue, cyan, magenta and yellow. Then show each color
into 4 stage shade.
'''

import matplotlib.pyplot as plt
import cv2
import numpy as np 

img_height = 200
img_width = 200


def main():

    shade_start = 0
    shade_increment = 63
    shade_stage = 5
    

    black = color_shading(0, 0, 0, shade_increment, shade_increment, shade_increment , shade_stage)
    red = color_shading(shade_start, 0, 0, shade_increment, 0, 0, shade_stage)
    green = color_shading(0, shade_start, 0, 0, shade_increment, 0, shade_stage)
    blue = color_shading(0, 0, shade_start, 0, 0, shade_increment, shade_stage)
    cyan = color_shading(0, shade_start, shade_start, 0, shade_increment, shade_increment, shade_stage)
    magenta = color_shading(shade_start, 0, shade_start, shade_increment, 0, shade_increment, shade_stage)
    yellow = color_shading(shade_start, shade_start, 0, shade_increment, shade_increment, 0, shade_stage)
    
    colors_name = ['black', 'red', 'green', 'blue', 'cyan', 'magenta', 'yellow']
    all_colors = black + red + green + blue + cyan + magenta + yellow 
    display(all_colors, colors_name, shade_stage)



def color_shading(c1=0, c2=0, c3=0, increment_channel1 = 0, increment_channel2 =0, increment_channel3 = 0, shade_stage = 3):
    img = []

    for i in range(shade_stage):
        tmp = np.zeros((img_height, img_width, 3), dtype=np.uint8)
        tmp[:] = [c1, c2, c3]
        if i+2 == shade_stage:
            if(increment_channel1 > 0):
                increment_channel1 += 3
            if(increment_channel2 > 0):
                increment_channel2 += 3
            if(increment_channel3 > 0):
                increment_channel3 += 3           
        c1 += increment_channel1
        c2 += increment_channel2
        c3 += increment_channel3
        img.append(tmp)
    
    return img



def display(img, colors_name, shade_stage):
    k = 0
    for i in range(len(img)):
        plt.subplot(len(colors_name), shade_stage, i+1)
        plt.imshow(img[i])
        #plt.title(colors_name[k])
        plt.axis('off')

        if (i+1) % shade_stage == 0:
            k += 1
      
    
    plt.tight_layout()
    plt.show()
    


if __name__ == '__main__':
    main()