import cv2
import mediapipe as mp
import pandas as pd
from openpyxl import Workbook
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

mpPose = mp.solutions.pose
pose = mpPose.Pose()
mpDraw = mp.solutions.drawing_utils

# Read data from the Excel sheet
excel_filename = 'C:/Users/Pravin/Desktop/heatmap generation/merged_data.xlsx'
bone_df = pd.read_excel(excel_filename)

# Determine the number of frames in your data
num_frames = len(bone_df)

# Define the number of bones (32 in your case)
num_bones = 33  # 0 to 32 bones

# Create an animated skeleton mesh
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Function to smooth data using a moving average
def smooth_data(data, window_size=5):
    smoothed_data = np.convolve(data, np.ones(window_size)/window_size, mode='same')
    return smoothed_data

def animate(i):
    ax.clear()
    ax.set_xlabel('X')  # X becomes X
    ax.set_ylabel('Y')  # Y becomes Y
    ax.set_zlabel('Z')  # Z becomes Z
    
    # Adjust the limits to zoom in on the plot
    ax.set_xlim(-0.2, 0.2)  # Adjust these values for tighter zoom
    ax.set_ylim(-0.2, 0.2)
    ax.set_zlim(0, 0.1)  # Set Z axis to represent height
    
    ax.set_title(f'Skeleton Mesh Animation (Frame {i})')
    
    for bone_idx in range(num_bones):
        # Extract the x, y, and z values for the bones for frame i and smooth them
        x_values = smooth_data(bone_df.iloc[i][f'{bone_idx}_X'])  # X becomes X
        y_values = smooth_data(bone_df.iloc[i][f'{bone_idx}_Y'])  # Y becomes Y
        z_values = smooth_data(bone_df.iloc[i][f'{bone_idx}_Z'])  # Z becomes Z
        
        # Increase marker size for better visibility    
        ax.scatter(x_values, y_values, z_values, marker='o', label=f'Bone {bone_idx}', s=20, c=f'C{bone_idx}')  # Assign different colors
        
        # Connect the bone points with lines
        if bone_idx in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]:
            ax.plot(x_values, y_values, z_values, linewidth=1, color=f'C{bone_idx}')

    # Adjust camera perspective to align with the axes
    ax.view_init(elev=90, azim=90)  # Rotate the view to align with the axes

# Create an animation with a slower speed (100 milliseconds between frames)
ani = FuncAnimation(fig, animate, frames=num_frames, interval=50)
plt.show()
