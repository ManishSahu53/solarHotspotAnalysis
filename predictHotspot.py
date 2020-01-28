import sys
import os
import argparse
from yolo import YOLO, detect_video
from PIL import Image


# Check and create directory
def checkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


# To detect image
def detect_img(yolo):
    img = options.image
    print('Input directory is %s' % (img))
    if options.output:
        path_output = options.output
    else:
        path_output = os.path.join(img, 'predict')

    checkdir(path_output)
    print('Output directory is %s ' % (path_output))

    if os.path.isdir(img):
        for path, subdirs, files in os.walk(img):
            for title in files:
                fileExt = os.path.splitext(title)[-1]
                name = os.path.splitext(title)[0]
                if fileExt.lower() == '.jpg':
                    print('Processing %s image' % (title))
                    try:
                        image = Image.open(os.path.join(path, title))
                    except:
                        print('Open Error! Try again!')
                    else:
                        r_image = yolo.detect_image(image)
                        try:
                            r_image.save(os.path.join(
                                path_output, name + '_predict.png'), 'PNG')
                        except:
                            print('Error saving to %s' % (os.path.join(
                                path_output, name + '_predict.png')))
        yolo.close_session()

    else:
        print('Input %s is not a valid directory' % (img))


options = None

if __name__ == '__main__':
    # class YOLO defines the default value, so suppress any default here
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    '''
    Command line options
    '''
    parser.add_argument(
        '--model', type=str,
        help='path to model weight file, default ' +
        YOLO.get_defaults("model_path")
    )

    parser.add_argument(
        '--anchors', type=str,
        help='path to anchor definitions, default ' +
        YOLO.get_defaults("anchors_path")
    )

    parser.add_argument(
        '--classes', type=str,
        help='path to class definitions, default ' +
        YOLO.get_defaults("classes_path")
    )

    parser.add_argument(
        '--gpu_num', type=int,
        help='Number of GPU to use, default ' +
        str(YOLO.get_defaults("gpu_num"))
    )

    parser.add_argument(
        '--image', default=False, type=str,
        help='Image detection mode, will ignore all positional arguments'
    )
    '''
    Command line positional arguments -- for video detection mode
    '''
    parser.add_argument(
        "--input", nargs='?', type=str, required=False, default='./path2your_video',
        help="Video input path"
    )

    parser.add_argument(
        "--output", nargs='?', type=str, default="",
        help="[Optional] Video output path"
    )

    options = parser.parse_args()

    if options.image:
        """
        Image detection mode, disregard any remaining command line arguments
        """
        print("Image detection mode")
        if "input" in options:
            print(" Ignoring remaining command line arguments: " +
                  options.input + "," + options.output)
        detect_img(YOLO(**vars(options)))
    elif "input" in options:
        detect_video(YOLO(**vars(options)), options.input, options.output)
    else:
        print("Must specify at least video_input_path.  See usage with --help.")
