### loss_fns.py ###
from __future__ import absolute_import, division, print_function, unicode_literals

import os
import io
import json
import sys
import logging

import tensorflow as tf
import numpy as np

@tf.function
def categorical_batch_loss(y_true,y_pred):
    true_index = np.argmax(y_true, axis=-1)
    loss = 0
    for batch_index in range(y_pred.shape[0]):
        loss += -np.log(y_pred[batch_index][true_index[batch_index]])
    print("loss",loss)
    return loss

@tf.function
def categorical_loss(y_true,y_pred):
    true_index = np.argmax(y_true, axis=-1)
    loss = -tf.math.log(y_pred[true_index])
    print("loss",loss)
    return loss

def ropa_loss(y_labels_one_hot, category_logits, landmark_visibility, landmark_indices, landmark_scores):

    cat_loss = tf.keras.losses.CategoricalCrossentropy(
        y_labels_one_hot,
        category_logits)

    loss_lst = [cat_loss]
    landmark_one_hot = np.zeros_like(landmark_scores) # (bs, 300, 300, 8)
    #print("landmark_one_hot.shape",landmark_one_hot.shape)
    #print("landmark_indices:",landmark_indices)
    #print("landmark_indices.shape",landmark_indices.shape) # (bs, 8, 2)
    #print("landmark_visibility.shape",landmark_visibility.shape) # (bs, 8, 1)
    for batch_index in range(landmark_one_hot.shape[0]):
        landmark_indices_batch = landmark_indices[batch_index] # (8, 2)
        for i in range(landmark_indices_batch.shape[0]):
            x, y = landmark_indices_batch[i][0], landmark_indices_batch[i][1]
            landmark_one_hot[batch_index][x][y][i] = 1

    landmark_one_hot = tf.convert_to_tensor(landmark_one_hot)

    #landmark_loss = tf.Tensor(0)
    for batch_index in range(landmark_indices.shape[0]):
        for i in range(landmark_visibility.shape[0]):
            if landmark_visibility[i] == 0:
                continue
            landmark_one_hot_i = landmark_one_hot[batch_index][:,:,i] # (300, 300, 8)
            landmark_scores_i = landmark_scores[batch_index][:,:,i] # (300, 300, 8)
            flattened_landmark_scores_i = np.ndarray.flatten(landmark_scores_i.numpy())
            flattened_landmark_one_hot_i = np.ndarray.flatten(landmark_one_hot_i.numpy())

            loss_lst.append(tf.keras.losses.CategoricalCrossentropy(
                flattened_landmark_one_hot_i,
                flattened_landmark_scores_i))

    return cat_loss
