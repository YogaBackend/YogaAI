import cv2
import numpy as np

def process(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.GaussianBlur(img_gray, (5, 5), 25)
    img_canny = cv2.Canny(img_blur, 5, 50)
    kernel = np.ones((3, 3))
    img_dilate = cv2.dilate(img_canny, kernel, iterations=4)
    img_erode = cv2.erode(img_dilate, kernel, iterations=1)
    return img_erode

def get_contours(img, img_original):
    img_contours = img_original.copy()
    contours, hierarchies = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    cv2.drawContours(img_contours, contours, -1, (0, 255, 0), -1) 
    return img_contours

# Use webcam (camera index 0)
cap = cv2.VideoCapture(0)

success, img1 = cap.read()
if not success:
    print("Error: Could not read the first frame from the webcam.")
    exit()

success, img2 = cap.read()
heat_map = np.zeros(img1.shape[:-1])

while success:
    diff = cv2.absdiff(img1, img2)
    img_contours = get_contours(process(diff), img1)

    heat_map[np.all(img_contours == [0, 255, 0], 2)] += 3
    heat_map[np.any(img_contours != [0, 255, 0], 2)] -= 3
    heat_map[heat_map < 0] = 0
    heat_map[heat_map > 255] = 255

    img_mapped = cv2.applyColorMap(heat_map.astype('uint8'), cv2.COLORMAP_JET)

    cv2.imshow("Original", img1)
    cv2.imshow("Heat Map", img_mapped)
    
    img1 = img2
    success, img2 = cap.read()
    
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()