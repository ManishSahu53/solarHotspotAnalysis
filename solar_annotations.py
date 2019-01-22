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


# Main functions
def main():
    for root, dirs, files in os.walk(path_input):
        list_file = open(path_output, 'w')
        if len(files) < 1:
            print('No file found. Exiting..')
            sys.exit()
        else:
            print('Found %s images' % (len(files)))
            for file in files:
                if file.endswith('.json'):
                    path_image = os.path.splitext(file)[0] + '.jpg'
                    json_data = json.load(open(os.path.join(root, file)))
                    shapes = json_data['shapes']
                    data = get_list(os.path.join(root, path_image), shapes)
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
