import cv2
import matplotlib.pyplot as plt

# Load image in grayscale
img_1_1 = cv2.imread('/home/mohon/4_1/lab/dip_lab/images/canny.png', 0)

# Apply Canny edge detection
edges = cv2.Canny(img_1, threshold1=100, threshold2=200)

# Apply CLAHE
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
enhanced = clahe.apply(img_1)


edges_clahe = cv2.Canny(enhanced, threshold1=30, threshold2=100)

# Display original and edge-detected images
plt.subplot(2, 3, 1)
plt.imshow(img_1, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(2, 3, 2)
plt.imshow(edges, cmap='gray')
plt.title('Canny Edges')
plt.axis('off')

plt.subplot(2, 3, 3)
plt.imshow(edges_clahe, cmap='gray')
plt.title('CLAHE Edges')
plt.axis('off')

plt.tight_layout()
plt.show()
