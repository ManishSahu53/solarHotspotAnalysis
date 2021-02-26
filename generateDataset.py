import json
import os
import argparse
import sys
import base64
from PIL import Image
import io


# Saving image from binary format to jpg
def getImg(string, path_output):
    image = base64.b64decode(string)       
    img = Image.open(io.BytesIO(image))
    img.save(path_output)
    return True


# Converting list to map
def list2dict(List):
    map = {}
    for i in range(len(List)):
        map[List[i]] = i
        map['F1'] = 0

    return map


# Creating directory
def checkDir(path):
    if not os.path.exists(path):
        os.makedirs(path)


# Saving vocab to txt file
def list2txt(List, txt_path):
    """Writes one token per line, 0-based line id corresponds to
    the id f the token.
    Args:
        vocab: (iterable object) yields token
        txt_path: (string) path to vocab file
    """
    with open(txt_path, 'w') as f:
        for token in List:
            f.write(str(token) + '\n')


# Loading txt as list
def load_txt(path_txt):
    with open(path_txt, encoding='utf-8') as f:
        vocab = f.read().splitlines()
    return vocab


# Generating list
# label_idx x_center y_center width height
def get_list(file):
    json_data = json.load(open(file))
    shapes = json_data['shapes']
    length = len(shapes)
    print('Number of shapes found : {} in {}'.format(length, file))

    data = []
    imageWidth = float(json_data['imageWidth'])
    imageHeight = float(json_data['imageHeight'])

    for i in range(length):
        maxCol = 0  # X
        maxRow = 0  # Y
        minCol = 9999  # X
        minRow = 9999  # Y

        label = str(mapLabel[shapes[i]['label']])
        points = shapes[i]['points']
        for j in range(len(points)):
            col = int(points[j][0])  # x
            row = int(points[j][1])  # Y
            # print('col', col, 'row', row)
            if col > maxCol:
                maxCol = col
            if col < minCol:
                minCol = col

            if row > maxRow:
                maxRow = row
            if row < minRow:
                minRow = row
        # print('maxRow', maxRow, 'maxCol', maxCol, 'minRow', minRow, 'minCol', minCol)
        width = maxCol-minCol
        height = maxRow-minRow
        centreX = (maxCol+minCol)/2
        centreY = (maxRow+minRow)/2

        bbox = [label, centreX/imageWidth, centreY /
                imageHeight, width/imageWidth, height/imageHeight]
        # print('bbox: {}'.format(bbox))
        data.append(' '.join([str(a) for a in bbox]))

    return data


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
    trainData = []
    for i, file in enumerate(files):
        if file.endswith('.json'):
            temp_path_image = os.path.splitext(file)[0] + '.jpg'
            path_output_image =  os.path.join(path_image, str(i) + '.jpg')

            if not os.path.isfile(temp_path_image):
                print('File {} not found. Creating from base64'.format(temp_path_image))

                json_data = json.load(open(file))
                img = json_data['imageData']

                # create a writable image and write the decoding result 
                getImg(img, path_output_image)

            trainData.append(path_output_image)
            data = get_list(file)
            list2txt(data, os.path.join(path_label, str(i) +'.txt'))

        else:
            pass

    list2txt(trainData, os.path.join(path_output, 'train.txt'))


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

    path_labelMap = os.path.join(path_output, 'classes.names')
    label = load_txt(path_labelMap)
    mapLabel = list2dict(label)

    path_image = os.path.join(path_output, 'images')
    path_label = os.path.join(path_output, 'labels')
    checkDir(path_image)
    checkDir(path_label)

    main()
    print('successfully completed')
    sys.exit()
