import cv2
import mediapipe as mp
import numpy as np
import pandas as pd

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

# Start capturing video from the webcam
cap = cv2.VideoCapture(0)

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

        # Render pose landmarks on the image
        mp_drawing.draw_landmarks(
            image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
        )

        # Display the image
        cv2.imshow('Mediapipe Feed', image)


        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

# Release the capture
cap.release()
cv2.destroyAllWindows()

# Create DataFrames from collected data
columns_bone = [f'LM_{i}_X' for i in range(33)] + [f'LM_{i}_Y' for i in range(33)] + [f'LM_{i}_Z' for i in range(33)]
columns_angle = ['Left_Elbow_Angle', 'Right_Elbow_Angle', 'Left_Knee_Angle', 'Right_Knee_Angle']

bone_df = pd.DataFrame(bone_data, columns=columns_bone)
angle_df = pd.DataFrame(angle_data, columns=columns_angle)

cap.release()
cv2.destroyAllWindows()
