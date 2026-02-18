import matplotlib.pyplot as plt
import cv2
import numpy as np

def main():
	h,w = 200,300
	shades = [0,50,84,168,255]
    
	image = np.ones((h,w,3),np.uint8)
	red,green,blue,yellow,cyan,magenta = image.copy(),image.copy(),image.copy(),image.copy(),image.copy(),image.copy()
	n = 7
    
	for i, shade in enumerate(shades):
		image =  np.ones((h,w,3),np.uint8)*shade
		plt.subplot(n,len(shades),i+1)
		plt.title(f"White Shade {shade/255.0:.2f}\n")
		plt.imshow(image)
		
		red[:, :, 0] = (shade) 
		green[:, :,1 ] = (shade)
		blue[:,:,2] = (shade)
	
		plt.subplot(n, len(shades),i+1+(len(shades)))
		plt.title(f" Red Shade {shade/255.0:.2f}")
		plt.imshow(red)
		plt.axis('off')
		
		plt.subplot(n,len(shades),i+1+(2*len(shades)))
		plt.title(f"Green Shade {shade/255.0:.2f}")
		plt.imshow(green)
		plt.axis('off')

		plt.subplot(n,len(shades),i+1+(3*len(shades)))
		plt.title(f"Blue Shade {shade/255.0:.2f}")
		plt.imshow(blue)
		plt.axis('off')
		
		yellow[:, :, 0] = shade
		yellow [:, :, 1] = shade
		plt.subplot(n,len(shades),i+1+(4*len(shades)))
		plt.title(f"Yellow Shade {shade/255.0:.2f}")
		plt.imshow(yellow)
		plt.axis('off')		
		
		cyan[:, :, 1] = shade
		cyan[:, : , 2] =shade
		plt.subplot(n,len(shades),i+1+(5*len(shades)))
		plt.title(f"Cyan Shade {shade/255.0:.2f}")
		plt.imshow(cyan)
		plt.axis('off')
		
		magenta[:, :, 0] = shade
		magenta[:, : , 2] =shade
		plt.subplot(n,len(shades),i+1+(6*len(shades)))
		plt.title(f"Magenta Shade {shade/255.0:.2f}")
		plt.imshow(magenta)
		plt.axis('off')
	
	plt.tight_layout()
	plt.subplots_adjust(hspace=0.8, wspace = 0.2)
	plt.show()
if __name__ == '__main__':
    main()
