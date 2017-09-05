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
        return [InputDesc(tf.uint8, [None, self.img_size, self.img_size, 3], 'input'),
                InputDesc(tf.int32, [None], 'label'),
                InputDesc(tf.float32, [None, 4], 'bbox')]
    def _build_graph(self, inputs):
        image, label, bbox = inputs
        # pdb.set_trace()

   
        image = tf.cast(image, tf.float32) #* (1.0 / 255)
        # Wrong mean/std are used for compatibility with pre-trained models.
        # Should actually add a RGB-BGR conversion here.
        # image_mean = tf.constant([0.485, 0.456, 0.406], dtype=tf.float32)
        # image_std = tf.constant([0.229, 0.224, 0.225], dtype=tf.float32)
        # image = (image - image_mean) / image_std
      
        if self.net_format == 'P-Net':
            # classification
            # pdb.set_trace()
            classification_indicator = tf.not_equal(label, -1)
            classification_label = tf.boolean_mask(tensor = label, mask = classification_indicator)



            classification_indicator = tf.reshape(classification_indicator,(-1, 1))
            classification_indicator = tf.tile(classification_indicator, [1, 2])
            classification_indicator = tf.reshape(classification_indicator, (-1, 2))
            # classification_image = tf.boolean_mask(tensor = image, mask = classification_indicator)
            # classification_image = tf.reshape(classification_image, [-1, 12, 12,3])


            p_net_result = self._p_net_conv(image, cfg.channels_12, cfg.kernel_size_12)
            conv = Conv2D('conv_class',p_net_result,2,(1,1),'VALID')
            # pdb.set_trace()
            conv = tf.reshape(conv, (-1,1*1*2))
            result_label = tf.nn.softmax(conv)
            result_label = tf.identity(result_label, name="labels")

            classification_image = tf.reshape(result_label, [-1,2])
            classification_image = tf.boolean_mask(tensor = classification_image, mask = classification_indicator)
            classification_image = tf.reshape(classification_image, [-1,2])
            classification_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=classification_image, labels=classification_label)
            classification_loss = tf.reduce_sum(classification_loss)


            # # regression
           
            regression_indicator = tf.not_equal(label, 0)
            regression_label = tf.boolean_mask(tensor = bbox, mask = regression_indicator)


            regression_indicator = tf.reshape(regression_indicator, (-1, 1))
            regression_indicator = tf.tile(regression_indicator, [1, 4])
            regression_indicator = tf.reshape(regression_indicator, (-1, 4))
            # regression_image = tf.boolean_mask(tensor = image, mask = regression_indicator)


            regression_image = Conv2D('conv_regress',p_net_result,4,(1,1),'VALID')
            result_bbox = tf.reshape(regression_image, [-1,1*1*4])
            # pdb.set_trace()
            result_bbox = tf.identity(result_bbox,name='bboxs')

            # pdb.set_trace()
            result_bbox = tf.reshape(regression_label, (-1,4))
            result_bbox = tf.boolean_mask(tensor = result_bbox, mask = regression_indicator)
            result_bbox = tf.reshape(result_bbox, (-1,4))
            regression_loss = tf.square(tf.subtract(result_bbox, regression_label))*0.5
            # regression_loss = tf.reduce_sum(regression_image)
            regression_loss = tf.reduce_sum(regression_loss)
            # regression_loss = tf.reduce_mean(tf.reduce_sum(regression_loss*0.5,1), name = 'regression_loss')


            # # landmark
            # landmark_indicator = tf.not_equal(label, 0)
            # landmark_image = tf.boolean_mask(tensor = image, mask = landmark_indicator)
            # landmark_label = tf.boolean_mask(tensor = landmark, mask = landmark_indicator)
            # landmark_image = self._p_net_conv(landmark_image, cfg.channels_12, cfg.kernel_size_12, 10, 1)
            # loss = tf.square(tf.subtract(landmark_image, landmark_label))
            # loss = tf.reduce_mean(tf.reduce_sum(loss,1), name = 'loss')
        
            #  target loss
            loss =   classification_loss + regression_loss

        elif self.net_format == 'R-Net':
            # classification
            classification_indicator = tf.not_equal(label, -1)
            classification_image = tf.boolean_mask(tensor = image, mask = classification_indicator)
            classification_label = tf.boolean_mask(tensor = label, mask = classification_indicator)
            classification_image = self._r_net_conv(classification_image, cfg.channels_24, cfg.kernel_size_24)
            classification_image = tf.reshape(classification_image, [-1,2])
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=classification_image, labels=classification_label)
            loss = tf.reduce_mean(loss, name='crocss_entropy_loss')


            # # regression
            # regression_indicator = tf.not_equal(label, 0)
            # regression_image = tf.boolean_mask(tensor = image, mask = regression_indicator)
            # regression_label = tf.boolean_mask(tensor = landmark, mask = regression_indicator)
            # regression_image = self._r_net_conv(regression_image, cfg.channels_12, cfg.kernel_size_12, 4, 1)
            # loss = tf.square(tf.subtract(regression_image, regression_label))
            # loss = tf.reduce_mean(tf.reduce_sum(loss,1), name = 'loss')   


            # # landmark
            # landmark_indicator = tf.not_equal(label, 0)
            # landmark_image = tf.boolean_mask(tensor = image, mask = landmark_indicator)
            # landmark_label = tf.boolean_mask(tensor = landmark, mask = landmark_indicator)
            # landmark_image = self._r_net_conv(landmark_image, cfg.channels_12, cfg.kernel_size_12, 10, 1)
            # loss = tf.square(tf.subtract(landmark_image, landmark_label))
            # loss = tf.reduce_mean(tf.reduce_sum(loss,1), name = 'loss')
        else:### O-Net
            ##classification
            classification_indicator = tf.not_equal(label, -1)
            classification_image = tf.boolean_mask(tensor = image, mask = classification_indicator)
            classification_label = tf.boolean_mask(tensor = label, mask = classification_indicator)
            classification_image = self._o_net_conv(classification_image, cfg.channels_24, cfg.kernel_size_24)
            classification_image = tf.reshape(classification_image, [-1,2])
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=classification_image, labels=classification_label)
            loss = tf.reduce_mean(loss, name='crocss_entropy_loss')


            # # regression
            # regression_indicator = tf.not_equal(label, 0)
            # regression_image = tf.boolean_mask(tensor = image, mask = regression_indicator)
            # regression_label = tf.boolean_mask(tensor = landmark, mask = regression_indicator)
            # regression_image = self._o_net_conv(regression_image, cfg.channels_12, cfg.kernel_size_12, 4, 1)
            # loss = tf.square(tf.subtract(regression_image, regression_label))
            # loss = tf.reduce_mean(tf.reduce_sum(loss,1), name = 'loss')   


            # # landmark
            # landmark_indicator = tf.not_equal(label, 0)
            # landmark_image = tf.boolean_mask(tensor = image, mask = landmark_indicator)
            # landmark_label = tf.boolean_mask(tensor = landmark, mask = landmark_indicator)
            # landmark_image = self._o_net_conv(landmark_image, cfg.channels_12, cfg.kernel_size_12, 10, 1)
            # loss = tf.square(tf.subtract(landmark_image, landmark_label))
            # loss = tf.reduce_mean(tf.reduce_sum(loss,1), name = 'loss')




        if cfg.weight_decay > 0:
            wd_cost = regularize_cost('.*/W', l2_regularizer(cfg.weight_decay), name='l2_regularize_loss')
            add_moving_summary(loss, wd_cost)
            self.cost = tf.add_n([loss, wd_cost], name='cost')
        else:
            add_moving_summary(loss)
            self.cost = tf.identity(loss, name='cost')


    def _get_optimizer(self):
        lr = get_scalar_var('learning_rate', 0.1, summary=True)
        return tf.train.MomentumOptimizer(lr, 0.9, use_nesterov=True)

    def _p_net_conv(self, img, channels, kernel_size):
        with tf.variable_scope("face_classification") as scope:
            for layer_idx, conv in enumerate(channels):
                # layer_input = tf.identity(img)
                # pdb.set_trace()
                img = Conv2D('conv.{}'.format(layer_idx),
                    img,
                    channels[layer_idx],
                    (kernel_size[layer_idx],kernel_size[layer_idx]),
                    'VALID')
                img = tf.nn.relu(img)
                if layer_idx == 0:
                    img = tf.nn.max_pool(img,[1, kernel_size[layer_idx], kernel_size[layer_idx], 1],[1, 2, 2, 1], 'SAME')
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
        # pdb.set_trace()
        img = tf.reshape(img, [-1, 3*3*64])
        w = tf.truncated_normal([3*3*64, 128], stddev = 0.1)
        b = tf.constant(0.1, shape = [128], stddev = 0.1)
        print("h")
        # img = tf.nn.relu()
        # return Conv2D('conv.{}'.format(layer_idx + 1),img,out_channel,(out_kernel_size,out_kernel_size),'VALID')

    def _o_net_conv(self, img, channels, kernel_size):
        with tf.variable_scope("face_classification") as scope:
            for layer_idx, conv in enumerate(channels):
                # layer_input = tf.identity(img)
                # pdb.set_trace()

                img = Conv2D('conv.{}'.format(layer_idx),
                    img,
                    channels[layer_idx],
                    (kernel_size[layer_idx],kernel_size[layer_idx]),
                    'VALID')
                img = tf.nn.relu(img)
                if layer_idx < 2:
                    img = tf.nn.max_pool(img,[1, 3, 3, 1],[1, 2, 2, 1], 'SAME')
                elif layer_idx == 2:
                    img = tf.nn.max_pool(img,[1, 2, 2, 1],[1, 2, 2, 1], 'SAME')
        # pdb.set_trace()
        img = tf.reshape(img, [-1, 3*3*64])
        w = tf.truncated_normal([3*3*64, 128], stddev = 0.1)
        b = tf.constant(0.1, shape = [128], stddev = 0.1)
        print("h")
        # img = tf.nn.relu()
        # return Conv2D('conv.{}'.format(layer_idx + 1),img,out_channel,(out_kernel_size,out_kernel_size),'VALID')

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
            InferenceRunner(dataset_val, [
                # ClassificationError('train-error', 'error'),
                ScalarStats('cost')]),
            ScheduledHyperParamSetter('learning_rate',
                                      #orginal learning_rate
                                      #[(0, 1e-2), (30, 3e-3), (60, 1e-3), (85, 1e-4), (95, 1e-5)]),
                                      #new learning_rate
                                      [(0, 2e-1)]),
            HumanHyperParamSetter('learning_rate'),
        ],
        model=Model(args.net_format),
        max_epoch=5000,
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
