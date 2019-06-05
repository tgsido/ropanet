### baseline_fashion_data.py ###
from __future__ import absolute_import, division, print_function, unicode_literals

import os
import io
import json
import sys
import logging
import tensorflow as tf
import pathlib
import random

random.seed(30)

def process_image(image):
  image = tf.image.decode_jpeg(image, channels=3)
  image = tf.image.resize(image, [299, 299])
  image /= 255.0
  return image

def read_and_process_image(path):
  image = tf.io.read_file(path)
  return process_image(image)

def process_path_label_tuple(path, label):
    return read_and_process_image(path), tf.cast(label, tf.int64)

def load_dataset(DEFAULT_DATA_DIR):
    ## pathname ##
    all_categories_names_data_path = os.path.join(DEFAULT_DATA_DIR, "anno_pre","list_category_cloth.txt")
    all_categories_names_data_path = pathlib.Path(all_categories_names_data_path)
    print(all_categories_names_data_path)

    ALL_CATEGORIES = []
    with open(all_categories_names_data_path,'r') as f:
        line_count = 0
        for line in f:
            line_count += 1
            if line_count < 3:
                continue

            ## parse out info ##
            line_lst = line.split()
            category_name = line_lst[0]
            ALL_CATEGORIES.append(category_name)



    ## pathname ##
    category_data_path = os.path.join(DEFAULT_DATA_DIR, "anno_pre","list_category_img.txt")
    category_data_path = pathlib.Path(category_data_path)
    print(category_data_path)
    all_img_paths = []
    all_category_labels = []
    with open(category_data_path,'r') as f:
        line_count = 0
        for line in f:
            line_count += 1
            if line_count < 3:
                continue

            ## parse out info ##
            line_lst = line.split()
            img_path = line_lst[0]
            category_label = line_lst[1]

            ## add to lists ##
            all_img_paths.append(os.path.join(DEFAULT_DATA_DIR,img_path))
            all_category_labels.append(category_label)

    print("len(all_img_paths):",len(all_img_paths))
    print("len(all_category_labels):",len(all_category_labels))

    ## shuffle data ##
    indices = list(range(len(all_img_paths)))
    random.shuffle(indices)
    print("indices[:4]",indices[:4])

    all_img_paths = [all_img_paths[i] for i in indices]
    all_category_labels = [int(all_category_labels[i]) for i in indices]

    ## start and end indices for data split ##

    train_start,train_end = 0,245000
    val_start,val_end = 245000,285000
    test_start,test_end = 285000,len(all_img_paths)
    """
    train_start,train_end = 0,20
    val_start,val_end = 20,25
    test_start,test_end = 25,30
    """

    train_paths = all_img_paths[train_start:train_end]
    train_categories = all_category_labels[train_start:train_end]

    val_paths = all_img_paths[val_start:val_end]
    val_categories = all_category_labels[val_start:val_end]

    test_paths = all_img_paths[test_start:test_end]
    test_categories = all_category_labels[test_start:test_end]


    train_ds = tf.data.Dataset.from_tensor_slices((train_paths, train_categories)).map(process_path_label_tuple)
    val_ds = tf.data.Dataset.from_tensor_slices((val_paths, val_categories)).map(process_path_label_tuple)
    test_ds = tf.data.Dataset.from_tensor_slices((test_paths, test_categories)).map(process_path_label_tuple)

    print("train_ds:",train_ds)
    print("ALL_CATEGORIES:",ALL_CATEGORIES)
    return train_ds, val_ds, test_ds, ALL_CATEGORIES
