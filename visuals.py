### visuals.py ###
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
import io
import tensorflow as tf

def update_confusion_matrix(matrix, y_true, y_pred):
    for index in range(len(y_true)):
        i = y_true[index] - 1
        j = y_pred[index] - 1
        matrix[i][j] += 1

def get_error_analysis_fig(img_matrix, pred_category, true_category):
    figure = plt.figure(figsize=(35, 35))
    plt.imshow(img_matrix)
    plt.title("predicted category: {} -- true category: {}".format(pred_category, true_category))
    return figure

"""
Referenced code from https://www.tensorflow.org/tensorboard/r2/image_summaries
for plotting functions
"""
def generate_img_from_plot(fig):
  buf = io.BytesIO()
  plt.savefig(buf, format='png')
  buf.seek(0)
  image = tf.expand_dims(tf.image.decode_png(buf.getvalue(), channels=4), axis = 0)
  return image

def get_confusion_matrix(matrix, categories):
  """
  Returns a matplotlib figure containing the plotted confusion matrix.

  Args:
    cm (array, shape = [n, n]): a confusion matrix of integer classes
    class_names (array, shape = [n]): String names of the integer classes
  """
  figure = plt.figure(figsize=(35, 35))
  plt.imshow(matrix, interpolation='nearest', cmap=plt.cm.Blues)
  plt.yticks(np.arange(len(categories)), categories)
  plt.xticks(np.arange(len(categories)), categories, rotation=60)
  plt.title("Confusion matrix")
  plt.colorbar()

  # Normalize the confusion matrix.
  #cm = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)

  for i in range(matrix.shape[0]):
      for j in range(matrix.shape[1]):
          if matrix[i][j] > matrix.max()/2:
              color = 'pink'
          else:
              color = 'black'
          plt.text(j, i, matrix[i][j], color=color, horizontalalignment="center")

  plt.tight_layout()
  plt.ylabel('True Category')
  plt.xlabel('Predicted Category')
  return figure
