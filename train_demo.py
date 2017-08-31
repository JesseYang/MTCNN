#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import numpy as np
import tensorflow as tf
import argparse
import os
import math
import json
import pdb

from tensorpack import *
from tensorpack.tfutils.symbolic_functions import *
from tensorpack.tfutils.summary import *

from reader import Data
from cfgs.config import cfg

BATCH_SIZE = 8

class Model(ModelDesc):
    def __init__(self):
        self.channels = cfg.channels
        self.channels.append(cfg.class_num)

    def _get_inputs(self):
        return [InputDesc(tf.float32, [None, None, None, 3], 'input'),
                InputDesc(tf.int32, [None, None], 'label')
               ]

    def _build_graph(self, input_vars):
        image, label = input_vars
        image = tf.identity(image, name="NETWORK_INPUT")
        tf.summary.image('input-image', image, max_outputs=5)
        l = image / 255.0 * 2 - 1

        with tf.variable_scope('segmentation') as scope:
            for layer_idx, dilation in enumerate(cfg.dilations):
                layer_input = tf.identity(l)
                if dilation == 1:
                    l = Conv2D('conv.{}'.format(layer_idx),
                               l,
                               self.channels[layer_idx],
                               (cfg.kernel_size[layer_idx], cfg.kernel_size[layer_idx]),
                               'SAME',
                               use_bias=not cfg.with_bn)
                else:
                    l = AtrousConv2D('atrous_conv.{}'.format(layer_idx),
                                     l,
                                     dilation,
                                     self.channels[layer_idx],
                                     (cfg.kernel_size[layer_idx], cfg.kernel_size[layer_idx]),
                                     'SAME',
                                     use_bias=not cfg.with_bn,
                                     mannual_atrous=False)

                if cfg.with_bn == True:
                    l = BatchNorm('bn.{}'.format(layer_idx), l)

                if layer_idx == len(cfg.dilations) - 1:
                    l = l
                else:
                    l = tf.nn.relu(l)

        output = tf.identity(l, name="NETWORK_OUTPUT")
        softmax_output = tf.nn.softmax(output, name="softmax_output")


        # value for the elements in label before preprocess ranges from 0 to class_num:
        #   0: not care
        #   i (i > 0): the i-th class (1-based)
        # this is because each element is saved as one byte (unsigned 8-bit int) in the label file, and its range is from 0 to 255
        # to match the output layer, the label should be processed
        label = tf.cast(label, tf.int32)
        label = label - 1
        # value for the elements in label after preprocess:
        #   -1: not care
        #   i (i >= 0): the i-th class (0-based)
        # this is to match that the labels parameter for
        # tf.nn.sparse_softmax_cross_entropy_with_logits ranges from [0, num_classes]
        # https://www.tensorflow.org/versions/r0.12/api_docs/python/nn.html#sparse_softmax_cross_entropy_with_logits
        label_indicator = tf.greater(label, -1)
        effective_label = tf.boolean_mask(tensor=label,
                                          mask=label_indicator)

        output = tf.reshape(output, [BATCH_SIZE, -1, cfg.class_num])
        effective_output = tf.boolean_mask(tensor=output,
                                           mask=label_indicator)

        cost = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=effective_output,
            labels=effective_label)
        cost = tf.reduce_mean(cost, name='cross_entropy_loss')

        wrong = prediction_incorrect(effective_output, effective_label)
        # monitor training error
        add_moving_summary(tf.reduce_mean(wrong, name='train_error'))

        # weight decay on all W
        if cfg.weight_decay > 0:
            wd_cost = tf.multiply(cfg.weight_decay, regularize_cost('.*/W', tf.nn.l2_loss), name='wd_cost')
            add_moving_summary(cost, wd_cost)

            self.cost = tf.add_n([cost, wd_cost], name='cost')
        else:
            add_moving_summary(cost)
            self.cost = tf.identity(cost, name='cost')

    def _get_optimizer(self):
        lr = get_scalar_var('learning_rate', 0.1, summary=True)
        return tf.train.MomentumOptimizer(lr, 0.9, use_nesterov=True)

def get_data(train_or_test):
    isTrain = train_or_test == 'train'
    ds = Data(train_or_test, affine_trans=isTrain, hflip=cfg.hflip, warp=isTrain)
    if isTrain:
        augmentors = [
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
        ]
    else:
        augmentors = []
    ds = AugmentImageComponent(ds, augmentors)
    ds = BatchData(ds, BATCH_SIZE, remainder=not isTrain)
    if isTrain:
        ds = PrefetchData(ds, 3, 2)
    return ds

def get_config(ds_train, ds_test):

    return TrainConfig(
        dataflow=ds_train,
        callbacks=[
            ModelSaver(),
            InferenceRunner(ds_test,
                [ScalarStats('cost')]),
            ScheduledHyperParamSetter('learning_rate',
                                      [(1, 1e-1),
                                       (50, 3e-2),
                                       (100, 1e-2)]),
            HumanHyperParamSetter('learning_rate'),
        ],
        model=Model(),
        max_epoch=150,
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', help='comma separated list of GPU(s) to use.', default=0)
    parser.add_argument('--load', help='load model')
    args = parser.parse_args()
    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    logger.auto_set_dir()
    ds_train = get_data("train")
    ds_test = get_data("test")

    config = get_config(ds_train, ds_test)
    if args.load:
        config.session_init = SaverRestore(args.load)
    QueueInputTrainer(config).train()

