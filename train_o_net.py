#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# File: imagenet-resnet.py
# Author: Yuxin Wu <ppwwyyxx@gmail.com>

import cv2
import sys
import argparse
import numpy as np
import os
import multiprocessing
import pdb
import tensorflow as tf
from tensorflow.contrib.layers import variance_scaling_initializer
from tensorpack import *
from tensorpack.utils.stats import RatioCounter
from tensorpack.tfutils.symbolic_functions import *
from tensorpack.tfutils.summary import *
from tensorpack.tfutils import summary

from reader_o_net import *
from cfgs.config import cfg

class Model(ModelDesc):
    
    def _get_inputs(self):
        return [InputDesc(tf.float32, [None, cfg.img_size_48, cfg.img_size_48, 3], 'input'),
                InputDesc(tf.int32, [None], 'label'),
                InputDesc(tf.float32, [None, 4], 'bbox'),
                InputDesc(tf.float32, [None, 10], 'landmark')]
    def _build_graph(self, inputs):
       # with tf.device('/gpu:1'):
        image, label, bbox, landmark = inputs
        tf.summary.image('input_img', image)
        image = (image - 127.5) / 128
        
        with argscope(Conv2D, kernel_shape=3, nl=tf.identity, stride=1, padding="VALID"):
            r_net_conv = (LinearWrap(image)
            .Conv2D("conv1", 32)
            .PReLU(name="relu1")
            .MaxPooling('pool1', shape=3, padding='SAME', stride=2)
            .Conv2D("conv2", 64)
            .PReLU(name="relu2")
            .MaxPooling(name='pool2', shape=3, stride=2)
            .Conv2D("conv3", 64)
            .PReLU(name="relu3")
            .MaxPooling(name='pool3', shape=2, padding='SAME', stride=2)
            .Conv2D('conv4', 128, kernel_shape=2)
            .PReLU(name="relu4")
            .FullyConnected('fc0', out_dim=256, nl=tf.identity)
            # .Dropout('dropout', 0.25)
            .PReLU(name="relu5")())

            classification_conv = (LinearWrap(r_net_conv)
                .FullyConnected('fc1', out_dim=2, nl=tf.identity)())

            regression_conv = (LinearWrap(r_net_conv)
                .FullyConnected('fc2', out_dim=4, nl=tf.identity)())

            landmark_conv = (LinearWrap(r_net_conv)
                .FullyConnected('fc3', out_dim=10, nl=tf.identity)())

        # classification
        classification_indicator = tf.not_equal(label, -1)
        classification_label = tf.boolean_mask(tensor = label, mask = classification_indicator)

        prob_cls = tf.nn.softmax(classification_conv)
        prob_cls = tf.identity(prob_cls, name="LABELS")
            
        result_label = tf.boolean_mask(tensor=classification_conv, mask=classification_indicator)
        result_label = tf.reshape(result_label, (-1, 2), name="labels")
        classification_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=result_label, labels=classification_label)
        classification_loss = tf.reduce_sum(classification_loss, name="classification_loss")

        wrong = symbolic_functions.prediction_incorrect(result_label, classification_label, name='incorrect')
        train_error = tf.reduce_mean(wrong, name='train_error')
        summary.add_moving_summary(train_error)

        # regression
        regression_indicator = tf.not_equal(label, 0)
        regression_label = tf.boolean_mask(tensor = bbox, mask = regression_indicator)
    
        prob_reg = tf.identity(regression_conv, name="BBOXS")

        result_bbox = tf.boolean_mask(tensor=regression_conv, mask=regression_indicator)
        result_bbox = tf.reshape(result_bbox, (-1, 4))
     
        regression_loss = tf.square(tf.subtract(result_bbox, regression_label)) * 0.5
        regression_loss = tf.reduce_sum(regression_loss, name='regression_loss')

        # landmark
        prob_landmark = tf.identity(landmark_conv, name="LANDMARK")

        landmark_max_indicator = tf.reduce_max(landmark, 1)
        # landmark_indicator = tf.greater(landmark_max_indicator, 0)
        landmark_indicator = tf.not_equal(landmark_max_indicator, 0)

        landmark_result = tf.boolean_mask(tensor=landmark_conv, mask=landmark_indicator)
        lanmdark_label = tf.boolean_mask(tensor=landmark, mask=landmark_indicator)


        landmark_loss = tf.square(tf.subtract(landmark_result, lanmdark_label))
        landmark_loss = tf.reduce_sum(landmark_loss, name='landmark_loss')

        # loss = tf.add(classification_loss, regression_loss, name="loss")
        loss = tf.add_n([classification_loss, regression_loss, landmark_loss], name="loss")

        if cfg.weight_decay > 0:
            wd_cost = regularize_cost('.*/W', l2_regularizer(cfg.weight_decay), name='l2_regularize_loss')
            add_moving_summary(loss, wd_cost)

            add_moving_summary(classification_loss, regression_loss, landmark_loss)
            # add_moving_summary(classification_loss, regression_loss)
            self.cost = tf.add_n([loss, wd_cost], name='cost')
        else:
            add_moving_summary(classification_loss, regression_loss, landmark_loss)
            self.cost = tf.identity(loss, name='cost')


    def _get_optimizer(self):
        lr = get_scalar_var('learning_rate', 0.1, summary=True)
        return tf.train.MomentumOptimizer(lr, 0.9, use_nesterov=True)

def get_data(train_or_test):
    isTrain = train_or_test == 'train'

    filename_list = cfg.train_list if isTrain else cfg.test_list
    ds = Data(filename_list)
    if isTrain:
        augmentors = [
            # imgaug.RandomCrop(crop_shape=448),
            imgaug.RandomOrderAug(
                [imgaug.Brightness(30, clip=False),
                 imgaug.Contrast((0.8, 1.2), clip=False),
                 imgaug.Saturation(0.4),
                 # rgb-bgr conversion
                 imgaug.Lighting(0.1,
                                 eigval=[0.2175, 0.0188, 0.0045][::-1],
                                 eigvec=np.array(
                                     [[-0.5675, 0.7192, 0.4009],
                                      [-0.5808, -0.0045, -0.8140],
                                      [-0.5836, -0.6948, 0.4203]],
                                     dtype='float32')[::-1, ::-1]
                                 )]),
            imgaug.Clip(),
            # imgaug.Flip(horiz=True),
            imgaug.ToUint8()
        ]
    else:
        augmentors = []
    ds = AugmentImageComponent(ds, augmentors)
  
    if isTrain:
        ds = PrefetchDataZMQ(ds, min(6, multiprocessing.cpu_count()))
    ds = BatchData(ds, BATCH_SIZE, remainder=not isTrain)
    return ds
def get_config(args):
    # pdb.set_trace()
    dataset_train = get_data('train')
    dataset_val = get_data('test')

    return TrainConfig(
        dataflow=dataset_train,
        callbacks=[
            ModelSaver(),
            # MinSaver('cost')
            PeriodicTrigger(InferenceRunner(dataset_val, [
                ScalarStats('regression_loss'),
                ScalarStats('classification_loss'),
                ScalarStats('landmark_loss'),
                ClassificationError('incorrect')]),
            	every_k_epochs=3),
                # HyperParamSetterWithFunc('learning_rate',
                #                      lambda e, x: 1e-4 * (4.0 / 5) ** (e * 11.0 / 32) ),
           ScheduledHyperParamSetter('learning_rate',
                                      # orginal learning_rate
                                      # [(0, 1e-2), (30, 3e-3), (60, 1e-3), (85, 1e-4), (95, 1e-5)],
                                      # new learning_rate
                                     [(0, 1e-4)]),
            HumanHyperParamSetter('learning_rate'),
        ],
        model=Model(),
        steps_per_epoch=7200,
        # max_epoch=29,
        max_epoch=1000000,
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', help='comma separated list of GPU(s) to use.', default='0')
    parser.add_argument('--batch_size', default=64)
    parser.add_argument('--load', help='load model')
    args = parser.parse_args()

    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    
    assert args.gpu is not None, "Need to specify a list of gpu for training!"
    NR_GPU = len(args.gpu.split(','))
    BATCH_SIZE = int(args.batch_size) // NR_GPU
    
    logger.auto_set_dir()
    config = get_config(args)
    if args.load:
        config.session_init = get_model_loader(args.load)
    config.nr_tower = NR_GPU
    SyncMultiGPUTrainer(config).train()
