import itertools
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import sklearn.metrics
import tfplot


def plot_confusion_matrix(cm, class_names):
    """
    Returns a matplotlib figure containing the plotted confusion matrix.

    Args:
    cm (array, shape = [n, n]): a confusion matrix of integer classes
    class_names (array, shape = [n]): String names of the integer classes
    """
    # Normalize the confusion matrix.
    cm_norm = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=4)

    figure = plt.figure(figsize=(5, 5))
    plt.imshow(cm_norm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    # cm = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)
    
    # Use white text if squares are dark; otherwise black.
    threshold = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        color = "white" if cm[i, j] > threshold else "black"
        plt.text(j, i, '{}({:.2f}%)'.format(cm[i, j], 100 * cm_norm[i, j]), horizontalalignment="center", color=color)

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    return figure


# y_true = [2, 0, 2, 2, 0, 1]
# y_pred = [0, 0, 2, 2, 0, 2]
# cm = sklearn.metrics.confusion_matrix(y_true, y_pred)
# cm = np.array([[2, 0, 0], [0, 0, 1], [1, 0, 2]])
# print(cm, type(cm))
# figure = plot_confusion_matrix(cm, np.array(['0', '1', '2']))
# # plt.show()


# img_d_summary_dir = "./tblog/summaries/img"
# print(img_d_summary_dir)

# sess = tf.InteractiveSession()


# img_d_summary_writer = [
#     tf.summary.FileWriter("./tblog/summaries/img/" + 'train', sess.graph),
#     tf.summary.FileWriter("./tblog/summaries/img/" + 'test', sess.graph),
# ]
# summary = [
#     tfplot.figure.to_summary(figure, tag='train'), 
#     tfplot.figure.to_summary(figure, tag='test')
# ]

# for epoch in range(10):
#     for iw, w in enumerate(img_d_summary_writer):
#         w.add_summary(summary[iw], epoch)
#         w.flush()