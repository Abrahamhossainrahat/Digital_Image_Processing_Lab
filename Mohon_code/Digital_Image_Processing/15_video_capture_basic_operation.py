import cv2
import numpy as np


def main():
    cam = cv2.VideoCapture(0)

    while True:
        _, frame = cam.read()       

        
        negative_img = negative(frame)
        gray = grayscale(frame)
        img_gamma = non_linear(gray)

        # Build final display rows
        row1 = np.hstack((frame, negative_img, cv2.merge([img_gamma] * 3)))
        row2 = np.hstack((draw_histogram(frame), draw_histogram(negative_img), draw_histogram(cv2.merge([img_gamma] * 3))))


        grid = np.vstack((row1, row2))
        cv2.imshow("Original | Negative | Gamma ", grid)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()


def grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def negative(img):
    return 255 - img

def non_linear(img, gamma=2):
    normalized = img / 255.0
    img = np.power(normalized, gamma)
    return np.uint8(img * 255)

def draw_histogram(img):
    gray = grayscale(img)    
    hist = np.zeros((256,), dtype=np.float32)

    for i in range(gray.shape[0]):
        for j in range(gray.shape[1]):
            hist[gray[i, j]] += 1
    
    hist = hist / np.max(hist) * 200
    hist_img = np.full((200, 256), 255, dtype=np.uint8)
    height = hist_img.shape[0]
    
    for x, y in enumerate(hist):
        cv2.line(hist_img, (x, height - 1), (x, height - 1 - int(y.item())), 0)


    return cv2.resize(cv2.merge([hist_img] * 3), (img.shape[1], img.shape[0]))




if __name__ == "__main__":
    main()
