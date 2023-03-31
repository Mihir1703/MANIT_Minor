import pandas as pd
import numpy as np
import scipy.sparse
import os
import pyarrow

def convert_to_parquet(filename, out_filename):
    df = pd.read_csv(filename)
    df.to_parquet(out_filename + ".parquet")


def convert_h5_to_sparse_csr(filename, out_filename, chunksize=2500):
    start = 0
    total_rows = 0

    sparse_chunks_data_list = []
    chunks_index_list = []
    columns_name = None
    while True:
        df_chunk = pd.read_hdf(filename, start=start, stop=start+chunksize)
        if len(df_chunk) == 0:
            break
        chunk_data_as_sparse = scipy.sparse.csr_matrix(df_chunk.to_numpy())
        sparse_chunks_data_list.append(chunk_data_as_sparse)
        chunks_index_list.append(df_chunk.index.to_numpy())

        if columns_name is None:
            columns_name = df_chunk.columns.to_numpy()
        else:
            assert np.all(columns_name == df_chunk.columns.to_numpy())

        total_rows += len(df_chunk)
        print(total_rows)
        if len(df_chunk) < chunksize: 
            del df_chunk
            break
        del df_chunk
        start += chunksize
        
    all_data_sparse = scipy.sparse.vstack(sparse_chunks_data_list)
    del sparse_chunks_data_list
    
    all_indices = np.hstack(chunks_index_list)
    
    scipy.sparse.save_npz(out_filename+"_values.sparse", all_data_sparse)
    np.savez(out_filename+"_idxcol.npz", index=all_indices, columns =columns_name)  



directory = '/home/mihir/coding/Minor/raw_files/sparse_matrix_dataset/cell_dataset'

for filename in os.listdir(directory):
    f = os.path.join(directory, filename)
    if os.path.isfile(f):
        split_tup = os.path.splitext(f)
        if(split_tup[1]=='.h5'):
            split_tup[0].replace('/home/mihir/coding/Minor/raw_files/sparse_matrix_dataset/cell_dataset/', ' ')
            print(split_tup[0])
            convert_h5_to_sparse_csr(f,f"New/{split_tup[0]}")
            
        if(split_tup[1]=='.csv'):
            split_tup[0].replace('/home/mihir/coding/Minor/raw_files/sparse_matrix_dataset/cell_dataset/', ' ')
            print(split_tup[0])
            convert_to_parquet(f, f"New/{split_tup[0]}")


def run_sparse():
    print("train_multi_targets.h5")
    convert_h5_to_sparse_csr("Dataset/train_multi_targets.h5", "New/Dataset/train_multi_targets")
    print("train_multi_inputs.h5")
    convert_h5_to_sparse_csr("Dataset/train_multi_inputs.h5", "New/Dataset/train_multi_inputs")
    print("train_cite_targets.h5")
    convert_h5_to_sparse_csr("Dataset/train_cite_targets.h5", "New/Dataset/train_cite_targets")
    print("train_cite_inputs.h5")
    convert_h5_to_sparse_csr("Dataset/train_cite_inputs.h5", "New/Dataset/train_cite_inputs")
    print("test_multi_inputs.h5")
    convert_h5_to_sparse_csr("Dataset/test_multi_inputs.h5", "New/Dataset/test_multi_inputs")