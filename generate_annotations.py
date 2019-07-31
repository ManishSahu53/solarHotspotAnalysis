import json
import os
import argparse
import sys
from facenet_pytorch import MTCNN, InceptionResnetV1
import config
from PIL import Image


mtcnn = MTCNN(image_size=config.IMAGE_SIZE, margin=config.IMAGE_SIZE/10)
mapping = config.MAPPING


# Generating list
def get_list(path, bbox, label):
    for bb in bbox:
        data = (path + ' ' + ','.join(str(int(a)) for a in bb))
        data = data + ',' + str(label)
    return data


# Main functions
def main():
    list_file = open(path_output, 'w')

    for root, dirs, files in os.walk(path_input):
        if len(files) < 1:
            print('No file found. Exiting..')
            sys.exit()
        else:
            print('Found %s images in directory %s' % (len(files), root))
            counter = 1
            for file in files:
                print('Processing %d image out of %d' % (counter, len(files)))
                if file.endswith(tuple(['.jpg', '.png', 'jpeg'])):
                    path_image = os.path.join(root, file)
                    print('path_image:', path_image)
                    label = mapping[os.path.basename(root)]
                    
                    try:
                        img = Image.open(path_image).convert('RGB')
                    except:
                        print('Could not open %s' % (path_image))
                        counter = counter + 1
                        continue

                    bbox, prob = mtcnn.detect(img)
                    data = get_list(path_image, bbox, label)
                    counter = counter + 1
                    
                else:
                    print('Skipping %s' %(root + '/' +file))
                    counter = counter + 1
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
