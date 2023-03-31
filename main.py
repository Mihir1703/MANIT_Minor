import os
from utils.image_compress import *
from utils.text_compress import *
from fileUpload import  *
from utils.sparse_matrix import *

filename = '/home/mihir/coding/Minor/raw_files/index.txt'
filetype = 'text'

def compress(filename,filetype):
    out = ""
    if filetype == 'text':
        out = compress_txt(filename)
    elif filetype == 'image' or filetype == 'video':
        out = compress_image(filename,"/home/mihir/coding/Minor/processed_files/compressed")
    elif filetype == 'csv' or filetype == '.h5':
        run_sparse()
    return out

def decompress(filename,filetype):
    out = ""
    if filetype == 'text':
        out = decompress_txt(filename)
    elif filetype == 'image' or filetype == 'video':
        out = decompress_image("/home/mihir/coding/Minor/processed_files","/home/mihir/coding/Minor/processed_files/decompress/")
    return out

file_path = compress("/home/mihir/coding/Minor/raw_files/image_dataset/Photos_Dataset/Photos_Dataset","image")
print(upload_to_cloud('/home/mihir/coding/Minor/processed_files','minor'))
