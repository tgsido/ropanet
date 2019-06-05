### models.py ###
from __future__ import absolute_import, division, print_function, unicode_literals

import os
import io
import json
import sys
import logging
import tensorflow as tf

class ModaNetBaseline(tf.keras.Model):
    def __init__(self):
        super(ModaNetBaseline, self).__init__()

        input_shape = (299, 299, 3)
        NUM_CLASSES = 50
        #initializer = tf.initializers.VarianceScaling(scale=2.0)

        self.inception_model = tf.keras.applications.inception_v3.InceptionV3(
            include_top=False,
            weights='imagenet',
            input_shape=input_shape,
            pooling=None)
        self.inception_model.trainable = False
        #print("self.inception_model.summary():",self.inception_model.summary())

        self.global_average_layer = tf.keras.layers.GlobalAveragePooling2D()

        self.fc = tf.keras.layers.Dense(
            NUM_CLASSES,
            activation='softmax',
            use_bias=True
            )


    def call(self, input_tensor, training=False):
        x = input_tensor
        x = self.inception_model(x)
        x = self.global_average_layer(x)
        x = self.fc(x)
        return x

def baseline_model():
    return ModaNetBaseline()

class InceptionFeatureExtractor(tf.keras.Model):
    def __init__(self):
        super(InceptionFeatureExtractor, self).__init__()

        ### CONSTANTS ###
        INCEPTION_INPUT_SHAPE = (299, 299, 3)
        LANDMARK_INPUT_SHAPE = (8,8,2048)
        NUM_CLASSES = 50
        #initializer = tf.initializers.VarianceScaling(scale=2.0)

        ### INCEPTION MODEL ###
        self.inception_model = tf.keras.applications.inception_v3.InceptionV3(
            include_top=False,
            weights='imagenet',
            input_shape=INCEPTION_INPUT_SHAPE,
            pooling=None)
        self.inception_model.trainable = False




    def call(self, input_tensor, training=False):
        x = input_tensor
        inception_feature_map = self.inception_model(x)
        return inception_feature_map
        return category_logits, landmark_scores

def inception_model():
    return InceptionFeatureExtractor()

class LandmarkModel(tf.keras.Model):
    def __init__(self):
        super(LandmarkModel, self).__init__()

        ### CONSTANTS ###
        INCEPTION_INPUT_SHAPE = (299, 299, 3)
        LANDMARK_INPUT_SHAPE = (8,8,2048)
        NUM_CLASSES = 50
        initializer = tf.keras.initializers.he_normal()

        ### CONV2D # 1 ###
        self.conv2d1 = tf.keras.layers.Conv2D(
            input_shape = LANDMARK_INPUT_SHAPE,
            filters = 2048,
            kernel_size = 1,
            strides = 1,
            padding = 'same',
            activation='sigmoid',
            use_bias = True)


        ### CONV2DT # 1 ###
        self.conv2dt1 = tf.keras.layers.Conv2DTranspose(
            filters = 256,
            kernel_size = 2,
            strides = 2,
            padding = 'valid',
            activation='relu',
            use_bias = True)

        ### CONV2DT # 2 ###
        self.conv2dt2 = tf.keras.layers.Conv2DTranspose(
            filters = 128,
            kernel_size = 2,
            strides = 2,
            padding = 'valid',
            activation='relu',
            use_bias = True)

        ### CONV2DT # 3 ###
        self.conv2dt3 = tf.keras.layers.Conv2DTranspose(
            filters = 64,
            kernel_size = 2,
            strides = 2,
            padding = 'valid',
            activation='relu',
            use_bias = True)

        ### CONV2DT # 4 ###
        self.conv2dt4 = tf.keras.layers.Conv2DTranspose(
            filters = 32,
            kernel_size = 12,
            strides = 1,
            padding = 'valid',
            activation='relu',
            use_bias = True)

        ### CONV2DT # 5 ###
        self.conv2dt5 = tf.keras.layers.Conv2DTranspose(
            filters = 16,
            kernel_size = 2,
            strides = 2,
            padding = 'valid',
            activation='relu',
            use_bias = True)

        ### CONV2DT # 4 ###
        self.conv2dt6 = tf.keras.layers.Conv2DTranspose(
            filters = 8,
            kernel_size = 2,
            strides = 2,
            padding = 'valid',
            activation='relu',
            use_bias = True)


    def call(self, input_tensor, training=False):
        x = input_tensor

        landmark_mask = self.conv2d1(x) # (8,8,2048)
        #print("landmark_mask.shape", landmark_mask.shape)
        landmark_out1 = self.conv2dt1(landmark_mask) # (16x, 64)
        #print("landmark_out1.shape", landmark_out1.shape)
        landmark_out2 = self.conv2dt2(landmark_out1) # (32x, 32 )
        #print("landmark_out2.shape", landmark_out2.shape)
        landmark_out3 = self.conv2dt3(landmark_out2) # (64x, 16 )
        #print("landmark_out3.shape", landmark_out3.shape)
        landmark_out4 = self.conv2dt4(landmark_out3) # (75x, 8 )
        #print("landmark_out4.shape", landmark_out4.shape)
        landmark_out5 = self.conv2dt5(landmark_out4) # (150x, 8 )
        #print("landmark_out5.shape", landmark_out5.shape)
        landmark_out6 = self.conv2dt6(landmark_out5) # (300x, 8 )
        #print("landmark_out4.shape", landmark_out6.shape)
        landmark_all_scores_inter1 = tf.transpose(landmark_out6, perm=[0, 3, 1, 2]) # (bs, 8, 300, 300)
        landmark_all_scores_inter2 = tf.reshape(
            landmark_all_scores_inter1,
            [landmark_all_scores_inter1.shape[0] * landmark_all_scores_inter1.shape[1], -1]
            ) # (bs * 8, 300^2)
        #print("landmark_all_scores_inter1.shape",landmark_all_scores_inter1.shape)
        landmark_all_scores_inter3 = tf.nn.softmax(landmark_all_scores_inter2, axis = 1)
        landmark_all_scores_inter4 = tf.reshape(
            landmark_all_scores_inter3,
            [landmark_all_scores_inter1.shape[0], landmark_all_scores_inter1.shape[1], 300, 300]
        )
        landmark_scores = tf.transpose(landmark_all_scores_inter4, perm=[0, 2, 3, 1])
        #print("landmark_scores.shape",landmark_scores.shape)

        return landmark_mask,landmark_scores

def landmark_model():
    return LandmarkModel()

class GlobalModel(tf.keras.Model):
    def __init__(self):
        super(GlobalModel, self).__init__()

        ### CONSTANTS ###
        INCEPTION_INPUT_SHAPE = (299, 299, 3)
        LANDMARK_INPUT_SHAPE = (8,8,2048)
        NUM_CLASSES = 50
        initializer = tf.keras.initializers.he_normal()

        ### CONV2D # 1 ###
        self.conv2d1 = tf.keras.layers.Conv2D(
            input_shape = LANDMARK_INPUT_SHAPE,
            filters = 2048,
            kernel_size = 5,
            strides = 1,
            padding = 'same',
            activation='tanh',
            use_bias = True)

        ### AVGPOOL ###
        self.avg_pool = tf.keras.layers.GlobalAvgPool2D()

        ### FULLY CONNECTED LOCAL 6 ###
        self.fc_global6 = tf.keras.layers.Dense(
            units=6,
            activation='relu',
            use_bias=True,
            bias_initializer=initializer,
            kernel_initializer=initializer)

    def call(self, input_tensor, training=False):
        x = input_tensor
        conv_out = self.conv2d1(x)
        pool_out = self.avg_pool(conv_out)
        global_out = self.fc_global6(pool_out)
        return global_out

def global_model():
    return GlobalModel()

class LocalModel(tf.keras.Model):
    def __init__(self):
        super(LocalModel, self).__init__()

        ### CONSTANTS ###
        INCEPTION_INPUT_SHAPE = (299, 299, 3)
        LANDMARK_INPUT_SHAPE = (8,8,2048)
        NUM_CLASSES = 50
        initializer = tf.keras.initializers.he_normal()

        ### AVGPOOL ### tried max_pool
        self.avg_pool = tf.keras.layers.GlobalAvgPool2D()

        ### FULLY CONNECTED LOCAL 6 ###
        self.fc_local6 = tf.keras.layers.Dense(
            units=6,
            activation='relu',
            use_bias=True,
            bias_initializer=initializer,
            kernel_initializer=initializer)

    def call(self, input_tensor, training=False):
        x = input_tensor
        max_out = self.avg_pool(x)
        local_out = self.fc_local6(max_out)
        return local_out

def local_model():
    return LocalModel()



class RopaNetCat(tf.keras.Model):
    def __init__(self):
        super(RopaNetCat, self).__init__()

        ### CONSTANTS ###
        INCEPTION_INPUT_SHAPE = (299, 299, 3)
        LANDMARK_INPUT_SHAPE = (8,8,2048)
        NUM_CLASSES = 50
        initializer = tf.keras.initializers.he_normal()

        ### CATEGORY FULLY CONNECTED ###
        self.cat_fc = tf.keras.layers.Dense(
            NUM_CLASSES,
            activation='softmax',
            use_bias=True,
            bias_initializer=initializer,
            kernel_initializer=initializer)


    def call(self, fusion_out, training=False):
        category_logits = self.cat_fc(fusion_out)
        return category_logits

def ropanetcat_model():
    return RopaNetCat()
