import gzip
import os

def compress_txt(input_file):
    out = '/home/mihir/coding/Minor/processed_files/' + input_file.split('/')[-1] + '.gz'
    with open(input_file, 'rb') as f_in, gzip.open(out, 'wb') as f_out:
        # Read the input file in chunks of 4096 bytes
        chunk_size = 4096
        while True:
            chunk = f_in.read(chunk_size)
            if not chunk:
                break
            # Write the compressed data to the output file
            f_out.write(chunk)
    return f'{input_file}.gz'


def decompress_txt(input_file):
    out = './processed_files/' + input_file.split('/')[-1] + '.txt'
    with gzip.open(input_file, 'rb') as f_in, open(out, 'wb') as f_out:
        # Read the input file in chunks of 4096 bytes
        chunk_size = 4096
        while True:
            chunk = f_in.read(chunk_size)
            if not chunk:
                break
            # Write the compressed data to the output file
            f_out.write(chunk)
    os.rename(out, input_file.split('/')[-1][:-3] + '.txt')
    return input_file.split('/')[-1][:-3] + '.txt'


import time

if __name__ == '__main__':
    print(os.listdir())
    
    start_time = time.time()
    print(compress_txt('/home/mihir/coding/Minor/raw_files/index.txt'))
    print("--- %s seconds ---" % (time.time() - start_time))
    