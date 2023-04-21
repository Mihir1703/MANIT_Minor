import os
import gzip
import pyarrow.parquet as pq

# Set the file paths
csv_path = '/home/mihir/Minor/Minor_project/raw_files/sparse_matrix_dataset/cell_dataset/metadata.csv'
parquet_path = '/home/mihir/Minor/Minor_project/processed_files/spare_compressed/metadata.parquet'

# Get the size of the CSV file
csv_size = os.path.getsize(csv_path)

# Compress the CSV file using gzip
with open(csv_path, 'rb') as f_in, gzip.open(csv_path + '.gz', 'wb') as f_out:
    f_out.writelines(f_in)

# Get the size of the compressed CSV file
csv_gz_size = os.path.getsize(csv_path + '.gz')

# Get the size of the Parquet file
parquet_size = os.path.getsize(parquet_path)

# Compress the Parquet file using gzip
parquet_table = pq.read_table(parquet_path)
pq.write_table(parquet_table, parquet_path + '.gz', compression='gzip')

# Get the size of the compressed Parquet file
parquet_gz_size = os.path.getsize(parquet_path + '.gz')

# Calculate the compression ratios
csv_ratio = csv_size / csv_gz_size
parquet_ratio = parquet_size / parquet_gz_size

# Print the results
print('CSV size: {} bytes'.format(csv_size))
print('Compressed CSV size: {} bytes'.format(csv_gz_size))
print('CSV compression ratio: {:.2f}'.format(csv_ratio))
print('Parquet size: {} bytes'.format(parquet_size))
print('Compressed Parquet size: {} bytes'.format(parquet_gz_size))
print('Parquet compression ratio: {:.2f}'.format(parquet_ratio))
