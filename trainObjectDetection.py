"""
Retrain the YOLO model for your own dataset.
"""

import argparse
import numpy as np
import keras.backend as K
from keras.layers import Input, Lambda
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

from objectDetection.yolo3.model import preprocess_true_boxes, yolo_body, tiny_yolo_body, yolo_loss
from objectDetection.yolo3.utils import get_random_data


def get_classes(path_classes):
    '''loads the classes'''
    with open(path_classes) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names


def get_anchors(path_anchors):
    '''loads the anchors from a file'''
    with open(path_anchors) as f:
        anchors = f.readline()
    anchors = [float(x) for x in anchors.split(',')]
    return np.array(anchors).reshape(-1, 2)


def create_model(input_shape, anchors, num_classes, load_pretrained=True, freeze_body=2,
                 weights_path='model_data/yolo_weights.h5'):
    '''create the training model'''
    K.clear_session()  # get a new session
    image_input = Input(shape=(None, None, 3))
    h, w = input_shape
    num_anchors = len(anchors)
    print(num_anchors, input_shape)
    y_true = [Input(shape=(h//{0: 32, 1: 16, 2: 8}[l], w//{0: 32, 1: 16, 2: 8}[l],
                           num_anchors//3, num_classes+5)) for l in range(3)]

    model_body = yolo_body(image_input, num_anchors//3, num_classes)
    print('Create YOLOv3 model with {} anchors and {} classes.'.format(
        num_anchors, num_classes))

    if load_pretrained:
        model_body.load_weights(weights_path, by_name=True, skip_mismatch=True)
        print('Load weights {}.'.format(weights_path))
        if freeze_body in [1, 2]:
            # Freeze darknet53 body or freeze all but 3 output layers.
            num = (185, len(model_body.layers)-3)[freeze_body-1]
            for i in range(num):
                model_body.layers[i].trainable = False
            print('Freeze the first {} layers of total {} layers.'.format(
                num, len(model_body.layers)))

    model_loss = Lambda(yolo_loss, output_shape=(1,), name='yolo_loss',
                        arguments={'anchors': anchors, 'num_classes': num_classes, 'ignore_thresh': 0.5})(
        [*model_body.output, *y_true])
    model = Model([model_body.input, *y_true], model_loss)

    return model


def create_tiny_model(input_shape, anchors, num_classes, load_pretrained=True, freeze_body=2,
                      weights_path='model_data/tiny_yolo_weights.h5'):
    '''create the training model, for Tiny YOLOv3'''
    K.clear_session()  # get a new session
    image_input = Input(shape=(None, None, 3))
    h, w = input_shape
    num_anchors = len(anchors)

    y_true = [Input(shape=(h//{0: 32, 1: 16}[l], w//{0: 32, 1: 16}[l],
                           num_anchors//2, num_classes+5)) for l in range(2)]

    model_body = tiny_yolo_body(image_input, num_anchors//2, num_classes)
    print('Create Tiny YOLOv3 model with {} anchors and {} classes.'.format(
        num_anchors, num_classes))

    if load_pretrained:
        model_body.load_weights(weights_path, by_name=True, skip_mismatch=True)
        print('Load weights {}.'.format(weights_path))
        if freeze_body in [1, 2]:
            # Freeze the darknet body or freeze all but 2 output layers.
            num = (20, len(model_body.layers)-2)[freeze_body-1]
            for i in range(num):
                model_body.layers[i].trainable = False
            print('Freeze the first {} layers of total {} layers.'.format(
                num, len(model_body.layers)))

    model_loss = Lambda(yolo_loss, output_shape=(1,), name='yolo_loss',
                        arguments={'anchors': anchors, 'num_classes': num_classes, 'ignore_thresh': 0.7})(
        [*model_body.output, *y_true])
    model = Model([model_body.input, *y_true], model_loss)

    return model


def data_generator(annotation_lines, batch_size, input_shape, anchors, num_classes):
    '''data generator for fit_generator'''
    n = len(annotation_lines)
    i = 0
    while True:
        image_data = []
        box_data = []
        for b in range(batch_size):
            if i == 0:
                np.random.shuffle(annotation_lines)
            image, box = get_random_data(
                annotation_lines[i], input_shape, random=True)
            image_data.append(image)
            box_data.append(box)
            i = (i+1) % n
        image_data = np.array(image_data)
        box_data = np.array(box_data)
        y_true = preprocess_true_boxes(
            box_data, input_shape, anchors, num_classes)
        yield [image_data, *y_true], np.zeros(batch_size)


def data_generator_wrapper(annotation_lines, batch_size, input_shape, anchors, num_classes):
    n = len(annotation_lines)
    if n == 0 or batch_size <= 0:
        return None
    return data_generator(annotation_lines, batch_size, input_shape, anchors, num_classes)


def _main():
    parser = argparse.ArgumentParser(
        description='Annotations to yolo bbox converter')
    parser.add_argument('-d', '--data',
                        help='Input file containing annotation path and image path',
                        required=True)

    parser.add_argument('-c', '--classes',
                        help='Input file containing classes',
                        required=True)

    parser.add_argument('-a', '--anchor',
                        help='Input file containing anchors',
                        required=True)

    parser.add_argument('-s', '--size', type=int,
                        help='Input file size. [Default] - 416x416',
                        default=416,
                        required=False)

    parser.add_argument('-e', '--epoch', type=int,
                        help='Enter number of epochs for training. [Default] 10',
                        default=10,
                        required=False)

    parser.add_argument('-b', '--batch', type=int,
                        help='Enter number of batch size for training. [Default] 4',
                        default=4,
                        required=False)

    parser.add_argument('-w', '--weight', type=str,
                        help='Path of pretrained weight file of yolov3.',
                        required=True)
            
    # Parsing arguments
    args = parser.parse_args()

    path_annotation = args.data
    log_dir = 'logs/000/'
    path_classes = args.classes
    path_weight = args.weight
    path_anchors = args.anchor
    size = args.size
    class_names = get_classes(path_classes)
    num_classes = len(class_names)
    anchors = get_anchors(path_anchors)
    epoch = args.epoch
    batch_size = args.batch

    print('path_annotation : %s' % (path_annotation))
    print('path_classes : %s' % (path_classes))
    print('path_anchors : %s' % (path_anchors))
    print('size : %d' % (size))
    print('class_names : %s' % (class_names))
    print('anchors : %s' % (anchors))
    print('log_dir : %s' % (log_dir))
    print('epoch : %s' % (epoch))

    input_shape = (size, size)  # multiple of 32, hw

    is_tiny_version = len(anchors) == 6  # default setting
    if is_tiny_version:
        model = create_tiny_model(input_shape, anchors, num_classes,
                                  freeze_body=2, weights_path=path_weight)
    else:
        model = create_model(input_shape, anchors, num_classes,
                             freeze_body=2, weights_path=path_weight)  # make sure you know what you freeze

    logging = TensorBoard(log_dir=log_dir)
    checkpoint = ModelCheckpoint(log_dir + 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
                                 monitor='val_loss', save_weights_only=True, save_best_only=True, period=3)
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss', factor=0.1, patience=3, verbose=1)
    early_stopping = EarlyStopping(
        monitor='val_loss', min_delta=0, patience=10, verbose=1)

    val_split = 0.1
    with open(path_annotation) as f:
        lines = f.readlines()
    np.random.seed(10101)
    np.random.shuffle(lines)
    np.random.seed(None)
    num_val = int(len(lines)*val_split)
    num_train = len(lines) - num_val

    # Train with frozen layers first, to get a stable loss.
    # Adjust num epochs to your dataset. This step is enough to obtain a not bad model.
    if True:
        model.compile(optimizer=Adam(lr=1e-3), loss={
            # use custom yolo_loss Lambda layer.
            'yolo_loss': lambda y_true, y_pred: y_pred})

        tune_batch_size = batch_size*8
        print('Train on {} samples, val on {} samples, with batch size {}.'.format(
            num_train, num_val, tune_batch_size))
        model.fit_generator(data_generator_wrapper(lines[:num_train], tune_batch_size, input_shape, anchors, num_classes),
                            steps_per_epoch=max(1, num_train//tune_batch_size),
                            validation_data=data_generator_wrapper(
                                lines[num_train:], tune_batch_size, input_shape, anchors, num_classes),
                            validation_steps=max(1, num_val//tune_batch_size),
                            epochs=epoch,
                            initial_epoch=0,
                            callbacks=[logging, checkpoint])
        model.save_weights(log_dir + 'trained_weights_stage_1.h5')

    # Unfreeze and continue training, to fine-tune.
    # Train longer if the result is not good.voc_classes
    if True:
        for i in range(len(model.layers)):
            model.layers[i].trainable = True
        # recompile to apply the change
        # recompile to apply the change
        model.compile(optimizer=Adam(lr=1e-4),
                      loss={'yolo_loss': lambda y_true, y_pred: y_pred})
        print('Unfreeze all of the layers.')

        # batch_size = 32  # note that more GPUvoc_classes required after unfreezing the body
        print('Train on {} samples, val on {} samples, with batch size {}.'.format(
            num_train, num_val, batch_size))
        model.fit_generator(data_generator_wrapper(lines[:num_train], batch_size, input_shape, anchors, num_classes),
                            steps_per_epoch=max(1, num_train//batch_size),
                            validation_data=data_generator_wrapper(
                                lines[num_train:], batch_size, input_shape, anchors, num_classes),
                            validation_steps=max(1, num_val//batch_size),
                            epochs=epoch*2,
                            initial_epoch=epoch,
                            callbacks=[logging, checkpoint, reduce_lr, early_stopping])
        model.save_weights(log_dir + 'trained_weights_final.h5')

    # Further training if needed.


if __name__ == '__main__':
    _main()
