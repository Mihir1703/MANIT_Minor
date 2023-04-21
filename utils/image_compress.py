import os
import subprocess

def compress_image(input_dir,output_dir):
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Loop through all files in the input directory and compress them
    for filename in os.listdir(input_dir):
        input_file = os.path.join(input_dir, filename)
        output_file = os.path.join(output_dir, filename)
        if filename.endswith('.jpg') or filename.endswith('.jpeg'):
            # Use WebP format to compress the image file
            command = f'cwebp -q 80 {input_file} -o {output_file}.webp'
            subprocess.call(command, shell=True)
        elif filename.endswith('.mp4'):
            # Use H.264/AVC codec to compress the video file
            command = f'ffmpeg -i {input_file}  {output_file}'
            subprocess.call(command, shell=True)
        else:
            # Copy other files directly to the output directory
            os.system(f'cp {input_file} {output_file}')

    print('Compression completed!')
    return output_file

def decompress_image(input_dir,output_dir):
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Loop through all files in the input directory and decompress them
    for filename in os.listdir(input_dir):
        input_file = os.path.join(input_dir, filename)
        output_file = os.path.join(output_dir, os.path.splitext(filename)[0])
        if filename.endswith('.webp'):
            # Use WebP format to decompress the image file
            command = f'dwebp {input_file} -o {output_file}.png'
            subprocess.call(command, shell=True)
        elif filename.endswith('.mp4'):
            # Use H.264/AVC codec to decompress the video file
            command = f'ffmpeg -i {input_file} {output_file}.avi'
            subprocess.call(command, shell=True)
        else:
            # Copy other files directly to the output directory
            os.system(f'cp {input_file} {output_file}')

    print('Decompression completed!')
    return output_file


import time
start_time = time.time()
print('Compressing images...', compress_image("/home/mihir/Minor/Minor_project/raw_files/Lungs_Cancer_Dataset/lung_colon_image_set/colon_image_sets/colon_aca", "/home/mihir/Minor/Minor_project/processed_files/Lungs_image_compressed"))
print("--- %s seconds ---" % (time.time() - start_time))
# if __name__ == '__main__':
#     print('Compressing images...', compress_image("/home/mihir/coding/Minor/raw_files/video", "/home/mihir/coding/Minor/processed_files/video"))
#     print('Compressing images...', decompress_image("/home/mihir/coding/Minor/processed_files/video", "/home/mihir/coding/Minor/processed_files/vedio"))