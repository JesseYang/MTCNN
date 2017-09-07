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

from reader import *
from cfgs.config import cfg

class Model(ModelDesc):
    def __init__(self, net_format='P-Net'):
        super(Model, self).__init__()
        self.net_format = net_format
        if self.net_format == 'P-Net':
            self.img_size = cfg.img_size_12
        elif self.net_format == 'R-Net':
            self.img_size = cfg.img_size_24
        else: self.img_size = cfg.img_size_48


    def _get_inputs(self):
        return [InputDesc(tf.float32, [None, self.img_size, self.img_size, 3], 'input'),
                InputDesc(tf.int32, [None], 'label'),
                InputDesc(tf.float32, [None, 4], 'bbox')]
    def _build_graph(self, inputs):
        image, label, bbox = inputs

        tf.summary.image('input_img', image)
        image = image * (1.0 / 255)
        # Wrong mean/std are used for compatibility with pre-trained models.
        # Should actually add a RGB-BGR conversion here.
        image_mean = tf.constant([0.485, 0.456, 0.406], dtype=tf.float32)
        image_std = tf.constant([0.229, 0.224, 0.225], dtype=tf.float32)
        image = (image - image_mean) / image_std
      
        if self.net_format == 'P-Net':

            classification_indicator = tf.not_equal(label, -1)
            classification_label = tf.boolean_mask(tensor = label, mask = classification_indicator)

            p_net_result = self._p_net_conv(image, cfg.channels_12, cfg.kernel_size_12)
            conv = Conv2D('conv_class', p_net_result, 2, (1, 1), 'VALID')

            prob_cls = tf.nn.softmax(conv)
            prob_cls = tf.identity(prob_cls, name="LABELS")
            

            result_label = tf.boolean_mask(tensor=conv, mask=classification_indicator)
            result_label = tf.reshape(result_label, (-1, 1 * 1 * 2), name="labels")
            classification_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=result_label, labels=classification_label)
            classification_loss = tf.reduce_sum(classification_loss, name="classification_loss")

            wrong = symbolic_functions.prediction_incorrect(result_label, classification_label, name='incorrect')
            train_error = tf.reduce_mean(wrong, name='train_error')
            summary.add_moving_summary(train_error)



            # regression
            regression_indicator = tf.not_equal(label, 0)
            regression_label = tf.boolean_mask(tensor = bbox, mask = regression_indicator)
    
            conv = Conv2D('conv_regress',p_net_result, 4, (1, 1), 'VALID')
            prob_reg = tf.identity(conv, name="BBOXS")

            result_bbox = tf.boolean_mask(tensor=conv, mask=regression_indicator)
            result_bbox = tf.reshape(result_bbox, (-1, 1 * 1 * 4))
     
            regression_loss = tf.square(tf.subtract(result_bbox, regression_label)) * 0.5
            regression_loss = tf.reduce_sum(regression_loss, name='regression_loss')


            loss = tf.add(classification_loss, regression_loss, name="loss")

        elif self.net_format == 'R-Net':
            # classification
            classification_indicator = tf.not_equal(label, -1)
            classification_image = tf.boolean_mask(tensor = image, mask = classification_indicator)
            classification_label = tf.boolean_mask(tensor = label, mask = classification_indicator)
            classification_image = self._r_net_conv(classification_image, cfg.channels_24, cfg.kernel_size_24)
            classification_image = tf.reshape(classification_image, [-1,2])
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=classification_image, labels=classification_label)
            loss = tf.reduce_mean(loss, name='crocss_entropy_loss')


           
        else:### O-Net
            ##classification
            classification_indicator = tf.not_equal(label, -1)
            classification_image = tf.boolean_mask(tensor = image, mask = classification_indicator)
            classification_label = tf.boolean_mask(tensor = label, mask = classification_indicator)
            classification_image = self._o_net_conv(classification_image, cfg.channels_24, cfg.kernel_size_24)
            classification_image = tf.reshape(classification_image, [-1,2])
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=classification_image, labels=classification_label)
            loss = tf.reduce_mean(loss, name='crocss_entropy_loss')

        if cfg.weight_decay > 0:
            wd_cost = regularize_cost('.*/W', l2_regularizer(cfg.weight_decay), name='l2_regularize_loss')
            # add_moving_summary(loss, wd_cost)
            self.cost = tf.add_n([loss, wd_cost], name='cost')
        else:
            add_moving_summary(classification_loss, regression_loss)
            self.cost = tf.identity(loss, name='cost')


    def _get_optimizer(self):
        lr = get_scalar_var('learning_rate', 0.1, summary=True)
        return tf.train.MomentumOptimizer(lr, 0.9, use_nesterov=True)

    def _p_net_conv(self, img, channels, kernel_size):
        with tf.variable_scope("face_classification") as scope:
            for layer_idx, conv in enumerate(channels):
                img = Conv2D('conv.{}'.format(layer_idx),
                    img,
                    channels[layer_idx],
                    (kernel_size[layer_idx],kernel_size[layer_idx]),
                    'VALID')
                img = tf.nn.relu(img)
                if layer_idx == 0:
                    img = tf.nn.max_pool(img,[1, kernel_size[layer_idx], kernel_size[layer_idx], 1], [1, 2, 2, 1], 'SAME')
        return img

    def _r_net_conv(self, img, channels, kernel_size):
        with tf.variable_scope("face_classification") as scope:
            for layer_idx, conv in enumerate(channels):
                # layer_input = tf.identity(img)
                img = Conv2D('conv.{}'.format(layer_idx),
                    img,
                    channels[layer_idx],
                    (kernel_size[layer_idx],kernel_size[layer_idx]),
                    'VALID')
                img = tf.nn.relu(img)
                if layer_idx != 2:
                    img = tf.nn.max_pool(img,[1, 3, 3, 1],[1, 2, 2, 1], 'SAME')
        
        img = tf.reshape(img, [-1, 3*3*64])
        w = tf.truncated_normal([3*3*64, 128], stddev = 0.1)
        b = tf.constant(0.1, shape = [128], stddev = 0.1)
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
            imgaug.Flip(horiz=True),
            imgaug.ToUint8()
        ]
    else:
        augmentors = []
    # ds = AugmentImageComponent(ds, augmentors)
  
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
            InferenceRunner(dataset_val, [
                ScalarStats('regression_loss'),
                ScalarStats('classification_loss'),
                ClassificationError('incorrect')]),
            ScheduledHyperParamSetter('learning_rate',
                                      #orginal learning_rate
                                      #[(0, 1e-2), (30, 3e-3), (60, 1e-3), (85, 1e-4), (95, 1e-5)]),
                                      #new learning_rate
                                      [(0, 1e-4)]),
            HumanHyperParamSetter('learning_rate'),
        ],
        model=Model(args.net_format),
        max_epoch=100000,
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', help='comma separated list of GPU(s) to use.', default='0')
    parser.add_argument('--batch_size', required=True)
    parser.add_argument('--net_format', choices=['P-Net', 'R-Net', 'Q-Net'], default='P-Net')
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
