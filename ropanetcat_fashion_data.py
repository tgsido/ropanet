### fashion_data.py ###
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

def process_example(path, label, vis_list, indices_lst):
    mapped_tuple = (read_and_process_image(path),
           (tf.cast(label, tf.int64),
           tf.convert_to_tensor(vis_list, tf.int32),
           tf.convert_to_tensor(indices_lst, tf.int32))
           )
    return mapped_tuple

def load_dataset(DEFAULT_DATA_DIR):
    all_categories_names_data_path = os.path.join(DEFAULT_DATA_DIR, "anno_pre","list_category_cloth.txt")
    all_categories_names_data_path = pathlib.Path(all_categories_names_data_path)
    print(all_categories_names_data_path)

    ALL_CATEGORIES = []
    f_obj = open(all_categories_names_data_path,'r')
    line_count = 0
    for line in f:
        line_count += 1
        if line_count < 3:
            continue

        ## parse out info ##
        line_lst = line.split()
        category_name = line_lst[0]
        ALL_CATEGORIES.append(category_name)
    f_obj.close()

    ## pathnames ##
    category_data_path = os.path.join(DEFAULT_DATA_DIR, "anno_pre","list_category_img.txt")
    category_data_path = pathlib.Path(category_data_path)
    #print("category_data_path:",category_data_path)

    landmark_data_path = os.path.join(DEFAULT_DATA_DIR, "anno_pre","list_landmarks.txt")
    landmark_data_path = pathlib.Path(landmark_data_path)
    #print("landmark_data_path:",landmark_data_path)

    data_dict = {}

    with open(category_data_path,'r') as f:
        line_count = 0
        for line in f:
            line_count += 1
            if line_count < 3:
                continue

            ## parse out info ##
            line_lst = line.split()
            img_path = line_lst[0]
            category_label = int(line_lst[1])

            ## add to lists ##
            full_img_path = os.path.join(DEFAULT_DATA_DIR,img_path)
            if full_img_path not in data_dict:
                data_dict[full_img_path] = {}

            data_dict[full_img_path]['category_label'] = category_label

    #print("1st pass: len(data_dict):",len(data_dict))

    with open(landmark_data_path,'r') as f:
        line_count = 0
        for line in f:
            line_count += 1
            if line_count < 3:
                continue

            ## parse out info ##
            line = line.strip()
            line_lst = line.split()
            img_path = line_lst[0]
            landmark_lst = line_lst[2:]

            landmark_visibilities = []
            landmark_indices = []
            #print("line:",line)
            #print("landmark_lst:",landmark_lst)
            for i in range(0, len(landmark_lst)-2, 3):
                #print("i:",i)
                vis = int(landmark_lst[i])
                x_i = int(landmark_lst[i+1])
                y_i = int(landmark_lst[i+2])
                coord = [x_i, y_i]
                landmark_visibilities.append(vis)
                landmark_indices.append(coord)

            MAX_NUM_LANDMARKS = 8
            for i in range(MAX_NUM_LANDMARKS - len(landmark_visibilities)):
                landmark_visibilities.append(0)
                landmark_indices.append([0,0])

            ## add to data_dict ##
            full_img_path = os.path.join(DEFAULT_DATA_DIR,img_path)
            if full_img_path not in data_dict:
                print("not in dict issue:", full_img_path)
                continue
            data_dict[full_img_path]['landmark_visibilities'] = landmark_visibilities
            data_dict[full_img_path]['landmark_indices'] = landmark_indices


    #print("2nd pass len(data_dict):",len(data_dict))



    all_img_paths = []
    all_category_labels = []
    all_landmark_visibilities = []
    all_landmark_indices = []

    for img_path in data_dict:
        ## grab data from data dict ##
        category_label = data_dict[img_path]['category_label']
        landmark_visibilities = data_dict[img_path]['landmark_visibilities']
        landmark_indices = data_dict[img_path]['landmark_indices']

        ## add to lists ##
        all_img_paths.append(img_path)
        all_category_labels.append(category_label)
        assert len(landmark_visibilities) == 8 and len(landmark_indices) == 8
        all_landmark_visibilities.append(landmark_visibilities)
        all_landmark_indices.append(landmark_indices)



    ## shuffle data ##
    indices = list(range(len(all_img_paths)))
    random.shuffle(indices)
    #print("indices[:4]",indices[:4])

    all_img_paths = [all_img_paths[i] for i in indices]
    all_category_labels = [all_category_labels[i] for i in indices]
    all_landmark_visibilities = [all_landmark_visibilities[i] for i in indices]
    all_landmark_indices = [all_landmark_indices[i] for i in indices]

    ## start and end indices for data split ##
    train_start,train_end = 0,245000
    val_start,val_end = 245000,285000
    test_start,test_end = 285000,len(all_img_paths)

    """
    train_start,train_end = 0,5
    val_start,val_end = 200,205
    test_start,test_end = 248,260
    """
    train_paths = all_img_paths[train_start:train_end]
    train_categories = all_category_labels[train_start:train_end]
    train_landmark_visibilities = all_landmark_visibilities[train_start:train_end]
    train_landmark_indices = all_landmark_indices[train_start:train_end]

    val_paths = all_img_paths[val_start:val_end]
    val_categories = all_category_labels[val_start:val_end]
    val_landmark_visibilities = all_landmark_visibilities[val_start:val_end]
    val_landmark_indices = all_landmark_indices[val_start:val_end]

    test_paths = all_img_paths[test_start:test_end]
    test_categories = all_category_labels[test_start:test_end]
    test_landmark_visibilities = all_landmark_visibilities[test_start:test_end]
    test_landmark_indices = all_landmark_indices[test_start:test_end]

    train_ds = tf.data.Dataset.from_tensor_slices((
        train_paths,
        train_categories,
        train_landmark_visibilities,
        train_landmark_indices
        )).map(process_example)

    val_ds = tf.data.Dataset.from_tensor_slices((
        val_paths,
        val_categories,
        val_landmark_visibilities,
        val_landmark_indices
        )).map(process_example)

    test_ds = tf.data.Dataset.from_tensor_slices((
        test_paths,
        test_categories,
        test_landmark_visibilities,
        test_landmark_indices
        )).map(process_example)

    print("Finished loading data from", category_data_path, "and", landmark_data_path, "....")
    print("train_ds:",train_ds)
    print("ALL_CATEGORIES:",ALL_CATEGORIES)
    return train_ds, val_ds, test_ds, ALL_CATEGORIES
