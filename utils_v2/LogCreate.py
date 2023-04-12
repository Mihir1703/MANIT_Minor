import os

directory_path = "/home/mihir/Minor/Minor_project/raw_files"
directory_info = []

def get_directory_info(path):
    directory_data = {}
    for item in os.listdir(path):
        item_path = os.path.join(path, item)
        if os.path.isfile(item_path):
            file_ext = os.path.splitext(item)[1]
            file_size = os.path.getsize(item_path)
            if file_ext not in directory_data:
                directory_data[file_ext] = (1, file_size)
            else:
                count, size = directory_data[file_ext]
                directory_data[file_ext] = (count + 1, size + file_size)
        else:
            subdirectory_data = get_directory_info(item_path)

            for file_ext, (count, size) in subdirectory_data.items():
                if file_ext not in directory_data:
                    directory_data[file_ext] = (count, size)
                else:
                    current_count, current_size = directory_data[file_ext]
                    directory_data[file_ext] = (current_count + count, current_size + size)
    directory_info.append((path, directory_data))
    return directory_data


get_directory_info(directory_path)

with open("directory_info.log", "w") as log_file:
    log_file.write("Directory Path\tFile Type\tCount\tSize (Bytes)\n")
    for path, data in directory_info:
        for file_type, (count, size) in data.items():
            log_file.write("{}\t{}\t{}\t{}\n".format(path, file_type, count, size))
