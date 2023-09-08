import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from openpyxl import Workbook

def calculate_angle(a, b, c):
    a = np.array(a)  # First
    b = np.array(b)  # Mid
    c = np.array(c)  # End
    
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi) 
    
    if angle > 180.0:
        angle = 360 - angle
        
    return angle

# Set up Mediapipe drawing utilities and pose model
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose


# Replace this line with the path to your video file
video_path = 'C:/Users/Pravin/Desktop/heatmap generation/dance women.mp4'

# Initialize video capture from the file
cap = cv2.VideoCapture(video_path)

# Start capturing video from the webcam
#cap = cv2.VideoCapture(0)

# Initialize the pose instance with confidence thresholds
with mp_pose.Pose(min_detection_confidence=0.8, min_tracking_confidence=0.5) as pose:
    bone_data = []  # Store bone data
    angle_data = []  # Store angle data
    
    while cap.isOpened():
        ret, frame = cap.read()
        
        if not ret:
            break
        
        # Recolor image to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        
        # Make detection
        results = pose.process(image)
        
        # Recolor back to BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                
        # Inside the while loop
        try:
            landmarks = results.pose_landmarks.landmark

            # Get coordinates of specific joints
            left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

            right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
            right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
            right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]

            left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
            left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
            left_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]

            right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
            right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
            right_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]

            # Calculate angles for all joints
            left_elbow_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
            right_elbow_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)
            left_knee_angle = calculate_angle(left_hip, left_knee, left_ankle)
            right_knee_angle = calculate_angle(right_hip, right_knee, right_ankle)

            # Append angle data to the list
            angle_data.append([left_elbow_angle, right_elbow_angle, left_knee_angle, right_knee_angle])

            # Visualize angles on the image
            cv2.putText(image, f'Left Elbow: {left_elbow_angle:.2f}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            cv2.putText(image, f'Right Elbow: {right_elbow_angle:.2f}', (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            cv2.putText(image, f'Left Knee: {left_knee_angle:.2f}', (50, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            cv2.putText(image, f'Right Knee: {right_knee_angle:.2f}', (50, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            # Collect bone data
            landmarks = []
            for lm in results.pose_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])
            bone_data.append(landmarks)

            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        except:
            # Handle the case when landmarks are not visible
            left_elbow_angle = 'N/A'
            right_elbow_angle = 'N/A'
            left_knee_angle = 'N/A'
            right_knee_angle = 'N/A'

            # Visualize "N/A" on the image
            cv2.putText(image, 'Left Elbow: N/A', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            cv2.putText(image, 'Right Elbow: N/A', (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            cv2.putText(image, 'Left Knee: N/A', (50, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            cv2.putText(image, 'Right Knee: N/A', (50, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # Display the image
        cv2.imshow('Mediapipe Feed', image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Create DataFrames from collected data
num_landmarks = 33
dimensions = ['X', 'Y', 'Z']
columns_bone = [f'{i}_{dim}' for i in range(num_landmarks) for dim in dimensions]
columns_angle = ['Left_Elbow_Angle', 'Right_Elbow_Angle', 'Left_Knee_Angle', 'Right_Knee_Angle']

bone_df = pd.DataFrame(bone_data, columns=columns_bone)
angle_df = pd.DataFrame(angle_data, columns=columns_angle)

# Merge bone data and angle data
merged_df = pd.concat([bone_df, angle_df], axis=1)

# Release the capture
cap.release()
cv2.destroyAllWindows()

# Save merged data to an Excel file
merged_df.to_excel('C:/Users/Pravin/Desktop/heatmap generation/merged_data.xlsx', index=False)
