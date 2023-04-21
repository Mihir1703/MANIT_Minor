# Read the CSV file into memory as a two-dimensional array, where each row of the CSV file corresponds to a row in the array and each column of the CSV file corresponds to a column in the array.
# For each column in the array, determine its data type(s) (e.g., integer, float, string).
# For each data type in the column, calculate the range of values (i.e., minimum and maximum values).
# Replace each value in the array with a tuple that contains the value's data type and its value.
# Group the tuples in the array by data type.
# For each group of tuples that share a data type, replace each value in the group with its corresponding index in a dictionary that maps each unique value in the group to an integer index.
# Write the compressed data to a new CSV file, where each value in the array is represented as a tuple containing its data type and integer index.
# Write the dictionaries used for compression to a separate file.
# To decompress the data, read the dictionaries from the file and use them to reverse the compression process, replacing each tuple containing a data type and integer index with its corresponding value in the original column.


import csv

def File_cleaning(file):
  with open(file, 'r') as f:
    reader = csv.reader(f)
    rows = [row for row in reader if all(col != '' for col in row)]

  with open(file + '_cleaned.csv', 'w') as f:
    writer = csv.writer(f)
    for row in rows:
        writer.writerow(row)


def compress_csv(csv_file):
    File_cleaning(csv_file)
    # read CSV file into memory as a list of rows
    with open(csv_file + '_cleaned.csv', 'r') as f:
        reader = csv.reader(f)
        rows = [row for row in reader]

    # determine data types for each column
    num_cols = len(rows[0])
    col_data_types = []
    for col in range(num_cols):
        data_types = set()
        for row in rows[1:]:
            try:
                float(row[col])
                data_types.add(float)
            except ValueError:
                data_types.add(str)
        col_data_types.append(data_types)

    # initialize dictionaries for each data type
    type_to_index = {}
    index_to_value = {}

    # compress data
    compressed_rows = []
    for row in rows:
        compressed_row = []
        for col, data_type in zip(row, col_data_types):
            if float in data_type:
                value = float(col)
                if 'float' not in type_to_index:
                    type_to_index['float'] = {}
                    index_to_value['float'] = {}
                if value not in type_to_index['float']:
                    index = len(type_to_index['float'])
                    type_to_index['float'][value] = index
                    index_to_value['float'][index] = value
                compressed_row.append(('float', type_to_index['float'][value]))
            elif str in data_type:
                if 'str' not in type_to_index:
                    type_to_index['str'] = {}
                    index_to_value['str'] = {}
                if col not in type_to_index['str']:
                    index = len(type_to_index['str'])
                    type_to_index['str'][col] = index
                    index_to_value['str'][index] = col
                compressed_row.append(('str', type_to_index['str'][col]))
        compressed_rows.append(compressed_row)

    # write compressed data and dictionaries to new CSV files
    with open(csv_file + '.compressed', 'w') as f:
        writer = csv.writer(f)
        for row in compressed_rows:
            writer.writerow(row)

    with open(csv_file + '.dicts', 'w') as f:
        writer = csv.writer(f)
        for data_type in type_to_index:
            for index, value in index_to_value[data_type].items():
                writer.writerow([data_type, index, value])



def decompress_csv(csv_file):
    # read compressed CSV file and dictionaries into memorY
        
    with open(csv_file, 'r') as f:
        reader = csv.reader(f)
        compressed_rows = [row for row in reader]

    type_to_index = {}
    index_to_value = {}
    with open(csv_file + '.dicts', 'r') as f:
        reader = csv.reader(f)
        for data_type, index, value in reader:
            index = int(index)
            if data_type not in type_to_index:
                type_to_index[data_type] = {}
                index_to_value[data_type] = {}
            type_to_index[data_type][index] = value
            index_to_value[data_type][value] = index

    # decompress data
    decompressed_rows = []
    for row in compressed_rows:
        decompressed_row = []
        for data_type, index in row:
            value = index_to_value[data_type][index]
            decompressed_row.append(value)
        decompressed_rows.append(decompressed_row)

    # write decompressed data to new CSV file
    with open(csv_file + '.decompressed', 'w') as f:
        writer = csv.writer(f)


compress_csv('/home/mihir/Minor/Minor_project/raw_files/basket_analysis.csv')