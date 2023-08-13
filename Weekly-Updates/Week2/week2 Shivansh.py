import cv2
import mediapipe as mp
import pandas as pd
from openpyxl import Workbook

mpPose = mp.solutions.pose
pose = mpPose.Pose()
mpDraw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)  # Use webcam (camera index 0)

# Create an empty list to store bone data
bone_data = []

while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = pose.process(imgRGB)

    if results.pose_landmarks:
        landmarks = []
        for lm in results.pose_landmarks.landmark:
            landmarks.extend([lm.x, lm.y, lm.z])
        bone_data.append(landmarks)

        mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)

    cv2.imshow("Image", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Create a DataFrame from the collected bone data
columns = [f'LM_{i}_X' for i in range(33)] + [f'LM_{i}_Y' for i in range(33)] + [f'LM_{i}_Z' for i in range(33)]
bone_df = pd.DataFrame(bone_data, columns=columns)

# Save the DataFrame to an Excel file
excel_filename = 'your location.xlsx'

# Create an Excel writer using openpyxl
with pd.ExcelWriter(excel_filename, engine='openpyxl') as writer:
    bone_df.to_excel(writer, sheet_name='Bone Data', index=False)

print("Excel file saved successfully.")

cap.release()
cv2.destroyAllWindows()
