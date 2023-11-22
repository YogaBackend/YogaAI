import cv2
import mediapipe as mp
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import multiprocessing

# Function to smooth data using a moving average
def smooth_data(data, window_size=5):
    smoothed_data = np.convolve(data, np.ones(window_size)/window_size, mode='same')
    return smoothed_data

def webcam_feed(pose_status_queue, num_bones, bone_df):
    mpPose = mp.solutions.pose
    pose = mpPose.Pose()
    mpDraw = mp.solutions.drawing_utils

    def check_pose(pose_landmarks, i, ax):
        threshold = 0.3
        pose_status = []

        # Calculate the frame index to access within the valid range
        i %= len(bone_df)

        for bone_idx in range(num_bones):
            x_values = smooth_data([pose_landmarks.landmark[bone_idx].x])
            y_values = smooth_data([pose_landmarks.landmark[bone_idx].y])
            z_values = smooth_data([pose_landmarks.landmark[bone_idx].z])

            # Extract the corresponding bone data from the Excel sheet
            excel_x = bone_df.iloc[i][f'{bone_idx}_X']
            excel_y = bone_df.iloc[i][f'{bone_idx}_Y']
            excel_z = bone_df.iloc[i][f'{bone_idx}_Z']

            # Calculate the Euclidean distance between the current and Excel bone positions
            distance = np.sqrt((x_values - excel_x)**2 + (y_values - excel_y)**2 + (z_values - excel_z)**2)

            # Check if the pose is within the threshold
            if np.any(distance > threshold):
                ax.plot(x_values, y_values, z_values, linewidth=1, color='red')
                pose_status.append("Bad Pose")
            else:
                pose_status.append("Good Pose")

        return pose_status

    cap = cv2.VideoCapture(0)
    i = 0  # Initialize frame counter

    # Create an animated skeleton meshq
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Perform pose estimation on the frame
        with mpPose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(frame_rgb)

            # Draw the pose on the frame
            if results.pose_landmarks:
                mpDraw.draw_landmarks(frame, results.pose_landmarks, mpPose.POSE_CONNECTIONS)

                # Check pose status for each bone
                pose_status = check_pose(results.pose_landmarks, i, ax)

                # Overlay the pose status on the frame
                cv2.putText(frame, f'Pose Status: {", ".join(pose_status)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (0, 255, 0), 2)

                # Send pose status to the skeleton animation process
                pose_status_queue.put(pose_status)

                i += 1  # Increment frame counter

            # Display the frame with pose estimation
            cv2.imshow('Webcam Feed with Pose Correction', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Function to handle 3D skeleton animation
def skeleton_animation(pose_status_queue, num_bones, bone_df):
    # Determine the number of frames in your data
    num_frames = len(bone_df)

    # Create an animated skeleton mesh
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    def animate(i):
        ax.clear()
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        
        ax.set_xlim(-0.2, 0.2)
        ax.set_ylim(-0.2, 0.2)
        ax.set_zlim(0, 0.1)
        
        ax.set_title(f'Skeleton Mesh Animation (Frame {i})')
        
        # Get pose status from the webcam feed process
        if not pose_status_queue.empty():
            pose_status = pose_status_queue.get()
            print(f'Frame {i} Pose Status: {", ".join(pose_status)}')

        for bone_idx in range(num_bones):
            x_values = smooth_data(bone_df.iloc[i][f'{bone_idx}_X'])
            y_values = smooth_data(bone_df.iloc[i][f'{bone_idx}_Y'])
            z_values = smooth_data(bone_df.iloc[i][f'{bone_idx}_Z'])
            
            ax.scatter(x_values, y_values, z_values, marker='o', label=f'Bone {bone_idx}', s=20, c=f'C{bone_idx}')
            
            if bone_idx in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]:
                ax.plot(x_values, y_values, z_values, linewidth=1, color=f'C{bone_idx}')

        ax.view_init(elev=90, azim=90)

    ani = FuncAnimation(fig, animate, frames=num_frames, interval=0)
    plt.show()

if __name__ == '__main__':
    # Define the number of bones (32 in your case)
    num_bones = 33  # 0 to 32 bones

    # Read data from the Excel sheet
    excel_filename = 'C:/Users/Pravin/Desktop/heatmap generation/merged_data.xlsx'
    bone_df = pd.read_excel(excel_filename)

    # Create a queue to communicate pose status between processes
    pose_status_queue = multiprocessing.Queue()

    # Create separate processes for webcam feed and skeleton animation
    webcam_process = multiprocessing.Process(target=webcam_feed, args=(pose_status_queue, num_bones, bone_df))
    animation_process = multiprocessing.Process(target=skeleton_animation, args=(pose_status_queue, num_bones, bone_df))

    # Start both processes
    webcam_process.start()
    animation_process.start()