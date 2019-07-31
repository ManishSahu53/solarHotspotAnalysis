import json
import os
import argparse
import sys


# Generating list
def get_list(path, shapes):
    length = len(shapes)
    bbox = []

    for i in range(length):
        if shapes[i]['shape_type'] == 'rectangle':
            bbox.append(shapes[i]['points'][0][0])
            bbox.append(',')
            bbox.append(shapes[i]['points'][0][1])
            bbox.append(',')
            bbox.append(shapes[i]['points'][1][0])
            bbox.append(',')
            bbox.append(shapes[i]['points'][1][1])
            bbox.append(',')
            bbox.append('0')
            bbox.append(' ')
    data = (path + ' ' + ''.join([str(a) for a in bbox]))
    return data[:-1]

def getListOfFiles(path):
    # create a list of file and sub directories 
    # names in the given directory 
    listOfFile = os.listdir(path)
    allFiles = list()
    # Iterate over all the entries
    for entry in listOfFile:
        # Create full path
        fullPath = os.path.join(path, entry)
        # If entry is a directory then get the list of files in this directory 
        if os.path.isdir(fullPath):
            allFiles = allFiles + getListOfFiles(fullPath)
        else:
            allFiles.append(fullPath)
                
    return allFiles   

# Main functions
def main():
    files = getListOfFiles(path_input)
    list_file = open(path_output, 'w')
    for file in files:
        if file.endswith('.json'):
            path_image = os.path.splitext(file)[0] + '.jpg'
            json_data = json.load(open(file))
            shapes = json_data['shapes']
            data = get_list(path_image, shapes)
        else:
            continue
        list_file.write(data)
        list_file.write('\n')
    list_file.close()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Annotations to yolo bbox converter')
    parser.add_argument('-d', '--data',
                        help='Input directory containing image and json',
                        required=True)

    parser.add_argument('-o', '--output',
                        help='output txt file. [Default] - Annotation.txt',
                        default='annotation.txt',
                        required=False)

    # Parsing arguments
    args = parser.parse_args()
    path_input = args.data
    path_output = args.output

    main()
    print('successfully completed')
    sys.exit()
