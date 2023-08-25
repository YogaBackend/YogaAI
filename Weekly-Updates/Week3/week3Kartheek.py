import cv2
import numpy as np
import mediapipe as mp
import pandas as pd
from openpyxl import Workbook

mpPose = mp.solutions.pose
pose = mpPose.Pose()
mpDraw = mp.solutions.drawing_utils

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
    contours, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(img_contours, contours, -1, (0, 255, 0), -1)
    return img_contours

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

success, img1 = cap.read()
if not success:
    print("Error: Could not read the first frame from the webcam.")
    exit()

heat_map = np.zeros(img1.shape[:-1])
bone_data = []

while cap.isOpened():
    success, img2 = cap.read()
    if not success:
        print("Error: Could not read a frame from the webcam.")
        break

    diff = cv2.absdiff(img1, img2)
    img_contours = get_contours(process(diff), img1)

    heat_map[np.all(img_contours == [0, 255, 0], 2)] += 3
    heat_map[np.any(img_contours != [0, 255, 0], 2)] -= 3
    heat_map[heat_map < 0] = 0
    heat_map[heat_map > 255] = 255

    non_zero_pixels = np.argwhere(heat_map > 0)
    if len(non_zero_pixels) > 0:
        x, y, w, h = cv2.boundingRect(non_zero_pixels)
        
        if w > 0 and h > 0:
            cropped_heatmap = heat_map[y:y+h, x:x+w]
            cropped_frame = img2[y:y+h, x:x+w]

            if cropped_frame.shape[0] > 0 and cropped_frame.shape[1] > 0:  # Check for valid dimensions
                img_mapped = cv2.applyColorMap(cropped_heatmap.astype('uint8'), cv2.COLORMAP_JET)

                imgRGB = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2RGB)
                results = pose.process(imgRGB)

                if results.pose_landmarks:
                    landmarks = []
                    for lm in results.pose_landmarks.landmark:
                        landmarks.extend([lm.x, lm.y, lm.z])
                    bone_data.append(landmarks)

                    mpDraw.draw_landmarks(cropped_frame, results.pose_landmarks, mpPose.POSE_CONNECTIONS)

                cv2.imshow("Heat Map", img_mapped)
                cv2.imshow("Skeleton Mesh", cropped_frame)

    img1 = img2

    key = cv2.waitKey(1)
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
