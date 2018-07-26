from zipfile import ZipFile
import os
import sys

directory = sys.argv[1]

# directory = 'machineLearning'

print(directory)


def get_all_file_paths(directory):
    # initializing empty file paths list
    file_paths = []

    # crawling through directory and subdirectories
    for root, directories, files in os.walk(directory):
        for filename in files:
            # join the two strings in order to form the full filepath.
            filepath = os.path.join(root, filename)
            file_paths.append(filepath)

    # returning all file paths
    return file_paths


def main():
    # path to folder which needs to be zipped
    file_paths = get_all_file_paths(directory)

    # printing the list of all files to be zipped
    print('Following files will be zipped:')
    for file_name in file_paths:
        print(file_name)

    # writing files to a zipfile
    # with ZipFile('C:\\Users\\vberlia\\Desktop\\us_auotmation\\modules.zip', 'w') as zip:
    file_name = directory + '.zip'
    with ZipFile(file_name, 'w') as zip:
        # writing each file one by one
        for file in file_paths:
            # print(file)
            zip.write(file)

main()
