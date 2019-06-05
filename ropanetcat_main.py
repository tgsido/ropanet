### ropanetcat_main.py ###
from __future__ import absolute_import, division, print_function, unicode_literals

import os
import io
import json
import sys
import logging

import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import logging
tf.get_logger().setLevel(logging.ERROR)
import time
import numpy as np

import datetime
from args import get_main_args
from ropanetcat_fashion_data import load_dataset
from models import baseline_model, ropanetcat_model, landmark_model, inception_model, local_model, global_model
from loss_fns import ropa_loss, categorical_loss
from visuals import get_confusion_matrix, generate_img_from_plot, \
update_confusion_matrix, get_error_analysis_fig

from keras.utils import plot_model



logging.basicConfig(level=logging.INFO)

MAIN_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA_DIR = os.path.join(MAIN_DIR, "data")
AUTOTUNE = tf.data.experimental.AUTOTUNE
NUM_CLASSES = 50

def main(args):
    USE_GPU = False
    if USE_GPU:
        device = '/device:GPU:0'
    else:
        device = '/cpu:0'

    if args.output_dir is None:
        print("ERROR: ",args.output_dir, "doesn't exits")
        return

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        print(args.output_dir , "created")

    TENSORBOARD_DIR = os.path.join(args.output_dir,"tb")
    SAVED_MODEL_DIR = os.path.join(args.output_dir,"saved_model")
    VISUALIZATIONS_DIR = os.path.join(args.output_dir,"visulaizations")
    dirs = [TENSORBOARD_DIR, SAVED_MODEL_DIR, VISUALIZATIONS_DIR]
    for dir in dirs:
        if not os.path.exists(dir):
            os.makedirs(dir)

    with tf.device(device):
        print("args",str(args))
        print("MAIN_DIR:",MAIN_DIR)
        print("DEFAULT_DATA_DIR:",DEFAULT_DATA_DIR)

        ### INIT TB LOGGING ###
        #summary_writer = tf.contrib.summary.create_file_writer(os.path.join(TENSORBOARD_DIR,str(datetime.datetime.now())))

        ### Model Hyperparameters ###
        train_batch_size = args.train_batch_size
        predict_batch_size = args.predict_batch_size
        num_train_epochs = args.num_train_epochs
        learning_rate = args.learning_rate

        ### Get Data ###
        train_ds, val_ds, test_ds, ALL_CATEGORIES = load_dataset(DEFAULT_DATA_DIR)

        print("test_ds:",test_ds)
        train_ds = train_ds.batch(train_batch_size)
        train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)

        val_ds = val_ds.batch(predict_batch_size)
        val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)

        test_ds = test_ds.batch(predict_batch_size)
        test_ds = test_ds.prefetch(buffer_size=AUTOTUNE)

        if args.load_frozen_model:
            if not os.path.exists(args.frozen_model_dir):
                print("ERROR: ",args.frozen_model_dir , "doesn't exits")
                return
            print('loading frozen model from:',args.frozen_model_dir)
            # Recreate the model
            model = ropanetcat_model()
            # Load the state of the old model
            model.load_weights(args.frozen_model_dir)
        else:
            model = ropanetcat_model()
        landmarkModel = landmark_model()
        inceptionModel = inception_model()
        globalModel = global_model()
        localModel = local_model()

        #model._set_inputs((None,299, 299, 3))

        if args.do_train:
            print_every = 10
            landmark_train_factor = 2

            # Compute the loss like we did in Part II
            loss_fn = tf.keras.losses.CategoricalCrossentropy()
            #loss_fn = ropa_loss()

            #print("model.summary():",model.summary())
            optimizer = tf.keras.optimizers.Adam(lr=learning_rate)

            ## METRICS ##
            train_loss = tf.keras.metrics.Mean(name='train_loss')
            train_accuracy = tf.keras.metrics.CategoricalAccuracy(name='train_accuracy')
            train_precision = tf.keras.metrics.Precision(name='train_precision')
            train_recall = tf.keras.metrics.Recall(name='train_recall')

            val_loss = tf.keras.metrics.Mean(name='val_loss')
            val_accuracy = tf.keras.metrics.CategoricalAccuracy(name='val_accuracy')
            val_precision = tf.keras.metrics.Precision(name='val_precision')
            val_recall = tf.keras.metrics.Recall(name='val_recall')

            test_loss = tf.keras.metrics.Mean(name='test_loss')
            test_accuracy = tf.keras.metrics.CategoricalAccuracy(name='test_accuracy')
            test_precision = tf.keras.metrics.Precision(name='test_precision')
            test_recall = tf.keras.metrics.Recall(name='test_recall')

            ## AUXILIARY METRICS ##
            # global #
            global_train_loss = tf.keras.metrics.Mean(name='global_train_loss')
            global_val_loss = tf.keras.metrics.Mean(name='global_val_loss')
            global_test_loss = tf.keras.metrics.Mean(name='global_test_loss')
            # local #
            local_train_loss = tf.keras.metrics.Mean(name='local_train_loss')
            local_val_loss = tf.keras.metrics.Mean(name='local_val_loss')
            local_test_loss = tf.keras.metrics.Mean(name='local_test_loss')
            # landmark #
            landmark_train_loss = tf.keras.metrics.Mean(name='landmark_train_loss')
            landmark_val_loss = tf.keras.metrics.Mean(name='landmark_val_loss')
            landmark_test_loss = tf.keras.metrics.Mean(name='landmark_test_loss')

            t = 0
            for epoch in range(num_train_epochs):

                # Reset main metrics # - https://www.tensorflow.org/alpha/guide/migration_guide#new-style_metrics
                train_loss.reset_states()
                train_accuracy.reset_states()
                train_precision.reset_states()
                train_recall.reset_states()

                ## reset auxiliary metrics #
                global_train_loss.reset_states()
                local_train_loss.reset_states()
                landmark_train_loss.reset_states()

                train_confusion_matrix = np.zeros((len(ALL_CATEGORIES),len(ALL_CATEGORIES)))

                train_batch_counter = 0
                t1 = datetime.datetime.now()
                for x,y in train_ds:
                    train_batch_counter += 1
                    y_labels, landmark_visibility, landmark_indices = y
                    y_labels_one_hot = tf.one_hot(y_labels - 1,NUM_CLASSES)


                    #category_logits = model(inception_feature_map, landmark_mask, training=True)


                    with tf.GradientTape() as model_tape:
                        with tf.GradientTape() as global_tape:
                            with tf.GradientTape() as local_tape:
                                with tf.GradientTape() as landmark_tape_pt1:
                                    with tf.GradientTape() as landmark_tape_pt2:
                                        # Use the model function to build the forward pass.
                                        inception_feature_map = inceptionModel(x)
                                        landmark_mask,landmark_scores = landmarkModel(inception_feature_map, training=True)

                                        ## CONDUCT FORWARD PASS ##
                                        global_out = globalModel(inception_feature_map, training=True)
                                        local_out = localModel(inception_feature_map * landmark_mask, training=True)
                                        fusion_out = tf.concat([global_out, local_out], axis=1)
                                        category_logits = model(fusion_out,training=True)
                                        loss = loss_fn(y_labels_one_hot, category_logits)

                                        ### LANDMARK MODEL UPDATES ###
                                        landmark_one_hot = np.zeros_like(landmark_scores) # (bs, 300, 300, 8)
                                        for batch_index in range(landmark_one_hot.shape[0]):
                                            landmark_indices_batch = landmark_indices[batch_index] # (8, 2)
                                            for i in range(landmark_indices_batch.shape[0]):
                                                x, y = landmark_indices_batch[i][0], landmark_indices_batch[i][1]
                                                landmark_one_hot[batch_index][x][y][i] = 1

                                        landmark_loss = 0
                                        landmark_one_hot = tf.convert_to_tensor(landmark_one_hot)
                                        for batch_index in range(landmark_indices.shape[0]):
                                            for i in range(landmark_visibility.shape[0]):
                                                if landmark_visibility[i] == 0:
                                                    continue
                                                landmark_one_hot_i = landmark_one_hot[batch_index][:,:,i] # (300, 300, 8)
                                                landmark_scores_i = landmark_scores[batch_index][:,:,i] # (300, 300, 8)
                                                flattened_landmark_scores_i = tf.reshape(landmark_scores_i, [-1])
                                                flattened_landmark_one_hot_i = tf.reshape(landmark_one_hot_i, [-1])
                                                #flattened_landmark_scores_i = tf.keras.layers.Flatten(landmark_scores_i)
                                                #flattened_landmark_one_hot_i =  tf.keras.layers.Flatten(landmark_one_hot_i)

                                                landmark_loss += categorical_loss(flattened_landmark_one_hot_i,flattened_landmark_scores_i)
                                        landmark_loss = tf.convert_to_tensor(landmark_loss)
                                        # UPDATE LANDMARK MODEL #
                                        # FROM LANDMARK LOSS #
                                        gradients = landmark_tape_pt1.gradient(landmark_loss, landmarkModel.trainable_variables)
                                        optimizer.apply_gradients(zip(gradients, landmarkModel.trainable_variables))
                                        # FROM CATEGORY LOSS #
                                        gradients = landmark_tape_pt2.gradient(loss, landmarkModel.trainable_variables)
                                        optimizer.apply_gradients(zip(gradients, landmarkModel.trainable_variables))

                                        ## CONDUCT OTHER GRADIENT UPDATES AT landmark_train_factor ##
                                        if train_batch_counter % landmark_train_factor == 0:
                                            ### MODEL UPDATES ###
                                            loss = loss_fn(y_labels_one_hot, category_logits)
                                            gradients = model_tape.gradient(loss, model.trainable_variables)
                                            optimizer.apply_gradients(zip(gradients, model.trainable_variables))

                                            ### GLOBAL MODEL UPDATES ###
                                            global_loss = loss_fn(y_labels_one_hot, category_logits)
                                            gradients = global_tape.gradient(global_loss, globalModel.trainable_variables)
                                            optimizer.apply_gradients(zip(gradients, globalModel.trainable_variables))

                                            ### LOCAL MODEL UPDATES ###
                                            local_loss = loss_fn(y_labels_one_hot, category_logits)
                                            gradients = local_tape.gradient(local_loss, localModel.trainable_variables)
                                            optimizer.apply_gradients(zip(gradients, localModel.trainable_variables))



                    t2 = datetime.datetime.now()
                    delta = t2 - t1
                    print("train_batch_counter: " + str(train_batch_counter) + " -> train batch done in", delta.total_seconds(),'seconds')
                    # Update landmark metrics #
                    train_loss.update_state(loss)
                    train_accuracy.update_state(y_labels_one_hot, category_logits)
                    train_accuracy_top3 = tf.keras.metrics.top_k_categorical_accuracy(y_true=y_labels_one_hot,y_pred=category_logits,k=3)
                    train_accuracy_top5 = tf.keras.metrics.top_k_categorical_accuracy(y_true=y_labels_one_hot,y_pred=category_logits,k=5)
                    landmark_train_loss.update_state(landmark_loss)
                    train_precision.update_state(y_labels_one_hot, category_logits)
                    train_recall.update_state(y_labels_one_hot, category_logits)

                    y_true = y_labels.numpy().tolist()
                    y_pred = np.argmax(category_logits, axis = 1) + 1
                    y_pred = y_pred.tolist()
                    update_confusion_matrix(train_confusion_matrix, y_true, y_pred)


                    if t % print_every == 0:
                        val_loss.reset_states()
                        val_accuracy.reset_states()
                        val_precision.reset_states()
                        val_recall.reset_states()
                        val_confusion_matrix = np.zeros((len(ALL_CATEGORIES),len(ALL_CATEGORIES)))

                        ## reset auxiliary metrics #
                        global_val_loss.reset_states()
                        local_val_loss.reset_states()
                        landmark_val_loss.reset_states()

                        val_batch_counter = 0
                        t1 = datetime.datetime.now()
                        for x,y in val_ds:
                            y_labels, landmark_visibility, landmark_indices = y
                            y_labels_one_hot = tf.one_hot(y_labels - 1,NUM_CLASSES)
                            val_batch_counter += 1

                            # During validation at end of epoch, training set to False
                            # Use the model function to build the forward pass.
                            inception_feature_map = inceptionModel(x)
                            landmark_mask,landmark_scores = landmarkModel(inception_feature_map, training=False)
                            global_out = globalModel(inception_feature_map, training=False)
                            local_out = localModel(inception_feature_map * landmark_mask, training=False)
                            fusion_out = tf.concat([global_out, local_out], axis=1)
                            category_logits = model(fusion_out,training=False)
                            loss = loss_fn(y_labels_one_hot, category_logits)

                            ### LANDMARK MODEL LOSS ###
                            landmark_one_hot = np.zeros_like(landmark_scores) # (bs, 300, 300, 8)
                            for batch_index in range(landmark_one_hot.shape[0]):
                                landmark_indices_batch = landmark_indices[batch_index] # (8, 2)
                                for i in range(landmark_indices_batch.shape[0]):
                                    x, y = landmark_indices_batch[i][0], landmark_indices_batch[i][1]
                                    landmark_one_hot[batch_index][x][y][i] = 1

                            landmark_loss = 0
                            landmark_one_hot = tf.convert_to_tensor(landmark_one_hot)
                            for batch_index in range(landmark_indices.shape[0]):
                                for i in range(landmark_visibility.shape[0]):
                                    if landmark_visibility[i] == 0:
                                        continue
                                    landmark_one_hot_i = landmark_one_hot[batch_index][:,:,i] # (300, 300, 8)
                                    landmark_scores_i = landmark_scores[batch_index][:,:,i] # (300, 300, 8)
                                    flattened_landmark_scores_i = tf.reshape(landmark_scores_i, [-1])
                                    flattened_landmark_one_hot_i = tf.reshape(landmark_one_hot_i, [-1])
                                    #flattened_landmark_scores_i = tf.keras.layers.Flatten(landmark_scores_i)
                                    #flattened_landmark_one_hot_i =  tf.keras.layers.Flatten(landmark_one_hot_i)

                                    landmark_loss += categorical_loss(flattened_landmark_one_hot_i,flattened_landmark_scores_i)
                            landmark_loss = tf.convert_to_tensor(landmark_loss)


                            val_loss.update_state(loss)
                            landmark_val_loss.update_state(landmark_loss)
                            val_accuracy.update_state(y_labels_one_hot, category_logits)
                            val_accuracy_top3 = tf.keras.metrics.top_k_categorical_accuracy(y_true=y_labels_one_hot,y_pred=category_logits,k=3)
                            val_accuracy_top5 = tf.keras.metrics.top_k_categorical_accuracy(y_true=y_labels_one_hot,y_pred=category_logits,k=5)
                            val_precision.update_state(y_labels_one_hot, category_logits)
                            val_recall.update_state(y_labels_one_hot, category_logits)

                            y_true = y_labels.numpy().tolist()
                            y_pred = np.argmax(category_logits, axis = 1) + 1
                            y_pred = y_pred.tolist()
                            update_confusion_matrix(val_confusion_matrix, y_true, y_pred)

                            t2 = datetime.datetime.now()
                            delta = t2 - t1
                            print("val batch done in", delta.total_seconds(),'seconds')

                        template = 'Iteration {}, Epoch {}, Train Loss: {}, Train Accuracy: {}, Train Landmark Loss: {}, Val Loss: {}, Val Accuracy: {}, \
                        Val Landmark Loss: {}, top_3_train_acc: {}, top_5_train_acc: {}, top_3_val_acc: {}, top_5_val_acc: {} , Train Precision: {}, \
                        Train Recall: {}, Val Precsion: {}, Val Recall: {}'
                        print (template.format(
                                             t,
                                             epoch+1,
                                             train_loss.result(),
                                             train_accuracy.result()*100,
                                             landmark_train_loss.result(),
                                             val_loss.result(),
                                             val_accuracy.result()*100,
                                             landmark_val_loss.result(),
                                             train_accuracy_top3*100,
                                             train_accuracy_top5*100,
                                             val_accuracy_top3*100,
                                             val_accuracy_top5*100,
                                             train_precision.result(),
                                             train_recall.result(),
                                             val_precision.result(),
                                             val_recall.result()
                                             )
                                )

                        # Generate pics for confusion matrices #.
                        fig = get_confusion_matrix(train_confusion_matrix, categories=ALL_CATEGORIES)
                        train_matrix_image = generate_img_from_plot(fig)

                        fig = get_confusion_matrix(val_confusion_matrix, categories=ALL_CATEGORIES)
                        val_matrix_image = generate_img_from_plot(fig)
                        """
                        with summary_writer.as_default():
                            tf.summary.scalar('train_loss', train_loss.result(), step=t)
                            tf.summary.scalar('train_accuracy', train_accuracy.result() * 100, step=t)
                            tf.summary.scalar('val_loss', val_loss.result(), step=t)
                            tf.summary.scalar('val_accuracy', val_accuracy.result() * 100, step=t)
                            tf.summary.scalar('train_accuracy_top3', train_accuracy_top3 * 100, step=t)
                            tf.summary.scalar('train_accuracy_top5', train_accuracy_top5 * 100, step=t)
                            tf.summary.scalar('val_accuracy_top3', val_accuracy_top3 * 100, step=t)
                            tf.summary.scalar('val_accuracy_top5', val_accuracy_top5 * 100, step=t)
                            tf.summary.scalar('landmark_train_loss', landmark_train_loss.result(), step=t)
                            tf.summary.scalar('landmark_val_loss', landmark_val_loss.result(), step=t)
                            tf.summary.scalar('train_precision', train_precision.result(), step=t)
                            tf.summary.scalar('train_recall', train_recall.result(), step=t)
                            tf.summary.scalar('val_precision', val_precision.result(), step=t)
                            tf.summary.scalar('val_recall', val_recall.result(), step=t)
                            tf.summary.image("Train Confusion Matrix", train_matrix_image, step=t)
                            tf.summary.image("Val Confusion Matrix", val_matrix_image, step=t)
                        """

                    t += 1
                    #print("t:",t, "train_batch_counter:",train_batch_counter)

        ### TEST SET PREDS ###
        test_img_counter = 0
        for x_test,y_test in test_ds:
            y_labels, landmark_visibility, landmark_indices = y_test
            y_labels_one_hot = tf.one_hot(y_labels - 1,NUM_CLASSES)

            # During validation at end of epoch, training set to False
            # Use the model function to build the forward pass.
            inception_feature_map = inceptionModel(x_test)
            landmark_mask,landmark_scores = landmarkModel(inception_feature_map, training=False)
            global_out = globalModel(inception_feature_map, training=False)
            local_out = localModel(inception_feature_map * landmark_mask, training=False)
            fusion_out = tf.concat([global_out, local_out], axis=1)
            category_logits = model(fusion_out,training=False)
            loss = loss_fn(y_labels_one_hot, category_logits)

            ### LANDMARK LOSS ###
            landmark_one_hot = np.zeros_like(landmark_scores) # (bs, 300, 300, 8)
            for batch_index in range(landmark_one_hot.shape[0]):
                landmark_indices_batch = landmark_indices[batch_index] # (8, 2)
                for i in range(landmark_indices_batch.shape[0]):
                    x, y = landmark_indices_batch[i][0], landmark_indices_batch[i][1]
                    landmark_one_hot[batch_index][x][y][i] = 1

            landmark_loss = 0
            landmark_one_hot = tf.convert_to_tensor(landmark_one_hot)
            for batch_index in range(landmark_indices.shape[0]):
                for i in range(landmark_visibility.shape[0]):
                    if landmark_visibility[i] == 0:
                        continue
                    landmark_one_hot_i = landmark_one_hot[batch_index][:,:,i] # (300, 300, 8)
                    landmark_scores_i = landmark_scores[batch_index][:,:,i] # (300, 300, 8)
                    flattened_landmark_scores_i = tf.reshape(landmark_scores_i, [-1])
                    flattened_landmark_one_hot_i = tf.reshape(landmark_one_hot_i, [-1])
                    #flattened_landmark_scores_i = tf.keras.layers.Flatten(landmark_scores_i)
                    #flattened_landmark_one_hot_i =  tf.keras.layers.Flatten(landmark_one_hot_i)

                    landmark_loss += categorical_loss(flattened_landmark_one_hot_i,flattened_landmark_scores_i)
            landmark_loss = tf.convert_to_tensor(landmark_loss)

            test_loss.update_state(loss)
            landmark_test_loss.update_state(landmark_loss)
            test_accuracy.update_state(y_labels_one_hot, category_logits)
            test_accuracy_top3 = tf.keras.metrics.top_k_categorical_accuracy(y_true=y_labels_one_hot,y_pred=category_logits,k=3)
            test_accuracy_top5 = tf.keras.metrics.top_k_categorical_accuracy(y_true=y_labels_one_hot,y_pred=category_logits,k=5)
            test_precision.update_state(y_labels_one_hot, category_logits)
            test_recall.update_state(y_labels_one_hot, category_logits)

            y_true = y_labels.numpy().tolist()
            y_pred = np.argmax(category_logits, axis = 1) + 1
            y_pred = y_pred.tolist()

            for i in range(x_test.shape[0]):
                test_img_counter += 1
                img_matrix = x_test[i]
                pred_category = ALL_CATEGORIES[y_pred[i] - 1]
                true_category = ALL_CATEGORIES[y_true[i] - 1]
                fig = get_error_analysis_fig(img_matrix, pred_category, true_category)
                test_image = generate_img_from_plot(fig)
                """
                with summary_writer.as_default():
                    tf.summary.image("Test Image #{}".format(test_img_counter), test_image, step=test_img_counter)
                """



        template = 'Test Loss: {}, Test Accuracy: {}, top_3_test_acc: {}, top_5_test_acc: {}, Landmark Test Loss: {}, \
        Test Precision: {}, Test Recall: {}'
        print (template.format(
                             test_loss.result(),
                             test_accuracy.result()*100,
                             test_accuracy_top3*100,
                             test_accuracy_top5*100,
                             landmark_test_loss.result(),
                             test_precision.result(),
                             test_recall.result()
                             )
                )

        ### VISUALIZATION CODE ###

        ### SAVE MODEL TO OUTPUT_DIR ###
        curr_date_str = str(datetime.datetime.now())
        model.save_weights(os.path.join(SAVED_MODEL_DIR,curr_date_str), save_format='tf')


    #if args.do_predict:




if __name__ == '__main__':
    main(get_main_args())
