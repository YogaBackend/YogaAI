The following project backend is being done by Shivansh and Kartheek.
The project aims to track the full body of the person in realtime and guide them if they are making any mistake.


for windows you will need to install these library first.

Windows+R -> type "cmd" -> paste the below given commands one line at a time

pip install opencv-python
pip install mediapipe
pip install pandas
pip install openpyxl
pip install matplotlib

OpenCV (cv2):
OpenCV, or Open Source Computer Vision Library, is a powerful library for computer vision and image processing tasks. It provides tools for loading, manipulating, and analyzing images and videos. In your code, it is used for reading and processing images or videos if needed.

Mediapipe:
Mediapipe is a library developed by Google that simplifies the process of building various types of media processing pipelines, including tasks like pose estimation, hand tracking, and facial recognition. In your code, you are using it for pose estimation to track the positions of different body parts in a series of frames.

Pandas:
Pandas is a popular data manipulation library that provides data structures and functions for efficiently working with structured data, such as spreadsheets or databases. In your code, Pandas is used to read and manage data from an Excel spreadsheet, making it easier to work with tabular data.

Openpyxl:
Openpyxl is a Python library for reading and writing Excel files. It allows you to create, modify, and extract data from Excel spreadsheets. In your code, Openpyxl is used to read data from an Excel file ('merged_data.xlsx').

Matplotlib:
Matplotlib is a versatile plotting library for creating static, animated, or interactive visualizations in Python. In your code, you use Matplotlib to create a 3D plot of the pose data and animate it over a series of frames. It helps you visualize the movement of different body parts in a 3D space.
