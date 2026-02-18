import cv2
import numpy as np
from heapq import heappush, heappop
import matplotlib.pyplot as plt

# ---------- Main ----------
def main():
    # Load grayscale image
    img = cv2.imread("/home/mohon/4_1/lab/dip_lab/images/piper.png", 0)
    
    print("Original image shape:", img.shape)

    # Step 1: Count frequency of each pixel
    u, c = np.unique(img, return_counts=True)
    frequency = dict(zip(u, c))

    # Step 2: Build Huffman Tree
    root = build_huffman_tree(frequency)

    # Step 3: Generate Huffman Codes
    codebook = generate_codes(root)
    # print("\n--- Huffman Codes ---")
    # for k, v in sorted(codebook.items()):
    #     print(f"Pixel {k}: Code {v}")

    # Step 4: Encode Image
    encoded = encode_image(img, codebook)
    print("\nEncoded bit length:", len(encoded))

    # Step 5: Decode back to verify
    decoded_img = decode_image(encoded, root, img.shape)

    # Step 6: Verify correctness
    same = np.array_equal(img, decoded_img)
    print("\nDecoded correctly:", same)    

    # Compression Ratio
    original_bits = img.size * 8
    compressed_bits = len(encoded)
    ratio = original_bits / compressed_bits
    print(f"\nCompression Ratio: {ratio:.2f} : 1")

    image_set = [img, decoded_img]
    title_set = ['Original Image', 'Decoded Image']
    display(image_set, title_set)


# ---------- Huffman Node Class ----------
class Node:
    def __init__(self, symbol, freq):
        self.symbol = symbol
        self.freq = freq
        self.left = None
        self.right = None

    # define comparison for heap
    def __lt__(self, other):
        return self.freq < other.freq


# ---------- Build Huffman Tree ----------
def build_huffman_tree(frequency):
    heap = []
    for symbol, freq in frequency.items():
        heappush(heap, Node(symbol, freq))

    while len(heap) > 1:
        left = heappop(heap)
        right = heappop(heap)
        merged = Node(None, left.freq + right.freq)
        merged.left = left
        merged.right = right
        heappush(heap, merged)

    return heap[0]


# ---------- Generate Huffman Codes ----------
def generate_codes(node, code="", codebook={}):
    if node is None:
        return

    if node.symbol is not None:
        codebook[node.symbol] = code
        return

    generate_codes(node.left, code + "0", codebook)
    generate_codes(node.right, code + "1", codebook)
    return codebook


# ---------- Encode Image ----------
def encode_image(image, codebook):
    encoded_str = ""
    for pixel in image.flatten():
        encoded_str += codebook[pixel]
    return encoded_str


# ---------- Decode Encoded String ----------
def decode_image(encoded_str, root, shape):
    decoded_pixels = []
    node = root
    for bit in encoded_str:
        if bit == '0':
            node = node.left
        else:
            node = node.right

        if node.symbol is not None:
            decoded_pixels.append(node.symbol)
            node = root

    decoded_array = np.array(decoded_pixels, dtype=np.uint8).reshape(shape)
    return decoded_array


def display(img_set, img_title, op=None):

    for i in range(len(img_set)):        
        plt.subplot(1, 2, i + 1)  
        if(img_set[i].ndim == 2):    # show gray-scale image 
            plt.imshow(img_set[i], cmap = 'gray')
            plt.title(img_title[i])  
            plt.axis('off') 
        else:   # show histogram
            plt.bar(range(256), img_set[i])
            plt.title(img_title[i])            
    
    plt.tight_layout() 
    plt.show()

if __name__ == "__main__":
    main()
