# import cv2
# import numpy as np

# # Define the file paths of the original and processed video datasets

# original_dataset_path = '/home/mihir/Minor/Minor_project/raw_files/video/test.mp4'
# processed_dataset_path = '/home/mihir/Minor/Minor_project/processed_files/video/test.mp4'

# # Create video capture objects for the original and processed videos
# original_cap = cv2.VideoCapture(original_dataset_path)
# processed_cap = cv2.VideoCapture(processed_dataset_path)

# # Calculate the PSNR of each frame and the mean PSNR of the video
# psnr_values = []
# while original_cap.isOpened() and processed_cap.isOpened():
#     # Read a frame from each video capture object
#     original_frame = original_cap.read()[1]
#     processed_frame = processed_cap.read()[1]
    
#     if original_frame is not None and processed_frame is not None:
#         # Calculate the PSNR of the two frames
#         mse = np.mean((original_frame - processed_frame) ** 2)
#         psnr = 10 * np.log10(255 ** 2 / mse)
        
#         # Append the PSNR to the list of PSNR values
#         psnr_values.append(psnr)
#     else:
#         # One of the video capture objects has reached the end of the video
#         break

# original_cap.release()
# processed_cap.release()

# # Calculate the mean PSNR of the video
# mean_psnr = np.mean(psnr_values)

# # Print the mean PSNR
# print(f'Mean PSNR: {mean_psnr:.2f}')

# # Calculate the information loss
# original_psnr = 40
# information_loss = (1 - (mean_psnr / original_psnr)) * 100

# # Print the information loss as a percentage
# print(f'Information loss: {information_loss:.2f}%')


import cv2
import numpy as np
import os

# Define the directory paths of the original and processed video datasets
original_dataset_dir = '/home/mihir/Minor/Minor_project/raw_files/video/'
processed_dataset_dir = '/home/mihir/Minor/Minor_project/processed_files/video/'

# Get a list of file names in the original dataset directory
original_files = os.listdir(original_dataset_dir)

# Calculate the PSNR of each video in the dataset
psnr_values = []
for file_name in original_files:
    # Define the file paths of the original and processed videos
    original_video_path = os.path.join(original_dataset_dir, file_name)
    processed_video_path = os.path.join(processed_dataset_dir, file_name)
    
    # Create video capture objects for the original and processed videos
    original_cap = cv2.VideoCapture(original_video_path)
    processed_cap = cv2.VideoCapture(processed_video_path)
    
    # Calculate the PSNR of each frame and the mean PSNR of the video
    video_psnr_values = []
    while original_cap.isOpened() and processed_cap.isOpened():
        # Read a frame from each video capture object
        original_frame = original_cap.read()[1]
        processed_frame = processed_cap.read()[1]

        if original_frame is not None and processed_frame is not None:
            # Calculate the PSNR of the two frames
            mse = np.mean((original_frame - processed_frame) ** 2)
            psnr = 10 * np.log10(255 ** 2 / mse)

            # Append the PSNR to the list of PSNR values for this video
            video_psnr_values.append(psnr)
        else:
            # One of the video capture objects has reached the end of the video
            break

    original_cap.release()
    processed_cap.release()

    # Calculate the mean PSNR of the video
    mean_video_psnr = np.mean(video_psnr_values)
    psnr_values.append(mean_video_psnr)

# Calculate the mean PSNR of the entire dataset
mean_psnr = np.mean(psnr_values)

# Print the mean PSNR
print(f'Mean PSNR: {mean_psnr:.2f}')

# Calculate the information loss
information_loss = 100 - (mean_psnr / 50) * 100

# Print the information loss as a percentage
print(f'Information loss: {information_loss:.2f}%')
