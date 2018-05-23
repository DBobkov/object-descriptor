#/usr/bin/python2
import numpy as np
use_x_server = True
import matplotlib as mpl
if use_x_server==False: # this is needed in case you are doing remote connection
    mpl.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf
import math
from time import time


class OptStats:
    best_test_acc = 0
    best_train_acc = 0
    train_acc_arr = np.array([])
    test_acc_arr = np.array([])
    best_avg_test_acc = 0
    best_test_at_epoch = -1


def computeF1Score(precision, recall):
    '''
    Get F1 score given precision and recall
    :param precision:
    :param recall:
    :return:
    '''
    f1 = 2 * precision * recall / (precision + recall)
    f1[np.isnan(f1)] = 0
    return f1

class ClassificationMatrics:
    avg_prec = 0
    avg_rec = 0
    total_prec = 0
    f1Score = 0


def computeClassificationMetricFromConfMat(cm_mat):
    metric = ClassificationMatrics()

    true_pos = np.diag(cm_mat)

    #precision
    prec_sum_ax0 = np.sum(cm_mat, axis=0)
    prec_sum_ax0[prec_sum_ax0 == 0] = 1
    precision = true_pos / prec_sum_ax0
    metric.avg_prec = np.mean(precision)
    metric.total_prec = np.sum(true_pos) / np.sum(cm_mat.flatten())

    #recall
    rec_sum_ax1 = np.sum(cm_mat, axis=1)
    rec_sum_ax1[rec_sum_ax1 == 0] = 1
    recall = true_pos / rec_sum_ax1
    metric.avg_rec = np.mean(recall)

    # f1score
    f1 = computeF1Score(precision, recall)
    metric.f1Score = np.mean(f1)

    return metric


def sgd_training_one_epoch(concatenate_training_set,
                           batch_size,
                           feat_size,
                           keep_probability,
                           sess,
                           train_step,
                           x,
                           y_,
                           keep_prob,
                           merged,
                           train_writer):
    '''
    Perform one epoch training using SGD over our data with the given parameters
    :param concatenate_training_set:
    :param batch_size:
    :param feat_size:
    :param keep_probability:
    :param sess:
    :param train_step:
    :param x:
    :param y_:
    :param keep_prob:
    :param merged:
    :param train_writer:
    :return:
    '''

    numb_instances = concatenate_training_set.shape[0]
    batch_num = int(math.ceil(float(numb_instances) / batch_size))

    step_b = int(math.ceil(batch_num/3.0))
    for batch_ind in range(batch_num):
        if batch_ind % step_b == 0:
            print("Train Batch %d/%d" % ((batch_ind + 1), batch_num))

        start_idx = batch_ind * batch_size
        end_idx = (batch_ind + 1) * batch_size
        end_idx = min(end_idx, numb_instances)
        used_ids = np.arange(start_idx, end_idx)

        mini_batch_feat = concatenate_training_set[used_ids, 0:feat_size]
        mini_batch_label = concatenate_training_set[used_ids, feat_size:]

        summary, step = sess.run([merged,
                                  train_step],
                                 feed_dict={x: mini_batch_feat,
                                            y_: mini_batch_label,
                                            keep_prob: keep_probability})

        train_writer.add_summary(summary, step)


def evaluate_accuracy(feat, label, batch_size, keep_prob, accuracy, x, y_, sess):
    '''
    Compute the accuracy on the given dataset
    :param feat:
    :param label:
    :param batch_size:
    :param keep_prob:
    :param accuracy:
    :param x:
    :param y_:
    :param sess:
    :return:
    '''

    numb_instances, desc_size = feat.shape
    keep_probability_testing = 1.0 # perform no drop out

    ## iterate over batches
    batch_num = int(math.ceil(float(numb_instances) / batch_size))
    acc_value = 0
    print('Evaluating accuracy for %d batches and %d objects' % (batch_num, numb_instances))
    for batch_ind in range(batch_num):
        print("Batch %d/%d" % (batch_ind, batch_num))

        start_idx = batch_ind * batch_size
        end_idx = (batch_ind + 1) * batch_size
        end_idx = min(end_idx, numb_instances)
        print("Start %d end %d" % (start_idx, end_idx))

        used_ids = np.arange(start_idx, end_idx)
        acc_value_one_batch = accuracy.eval(  session=sess,
                                              feed_dict={x: np.array(feat[used_ids, :]),
                                                         y_: np.array(label[used_ids, :]),
                                                         keep_prob: keep_probability_testing})
        normalization_acc = float(used_ids.size) / numb_instances
        acc_value_one_batch = acc_value_one_batch * normalization_acc
        acc_value = acc_value + acc_value_one_batch
    return acc_value

def getConfusionMatrix(sess,
                       x,
                       y_,
                       keep_prob,
                       confusion_matrix_tf,
                       feature,
                       label,
                       normalized ):
    '''
    Calculate confusion matrix
    :param sess:
    :param x:
    :param y_:
    :param keep_prob:
    :param confusion_matrix_tf:
    :param feature:
    :param label:
    :param normalized: whether you want normalized confusion matrix
    :return:
    '''

    cm = confusion_matrix_tf.eval(session=sess,
                                  feed_dict={x: feature,
                                             y_: label,
                                             keep_prob: 1.0})

    cm = cm.astype(float)
    if normalized:
        row_sums = cm.sum(axis=1)
        cm = cm / row_sums[:, np.newaxis]

    return cm



def writeConfusionMatrix(sess,
                         x,
                         y_,
                         keep_prob,
                         confusion_matrix_tf,
                         feat,
                         label,
                         label_strings,
                         out_dir ):
    '''
    Write confusion matrix to NUMPY array
    :param sess:
    :param x:
    :param y_:
    :param keep_prob:
    :param confusion_matrix_tf:
    :param feat:
    :param label:
    :param label_strings:
    :param out_dir:
    :return:
    '''
    normalized = False
    cm = getConfusionMatrix(sess,
                            x,
                            y_,
                            keep_prob,
                            confusion_matrix_tf,
                            feat,
                            label,
                            normalized)
    np.save(out_dir + "/conf_mat.npy", cm) # write 

    normalized=True
    cm_norm = getConfusionMatrix(sess,
                                 x,
                                 y_,
                                 keep_prob,
                                 confusion_matrix_tf,
                                 feat,
                                 label,
                                 normalized)

    print cm_norm.shape

    file_out = out_dir + "/conf_mat.png"
    assert (len(label_strings) == cm_norm.shape[0])
    plot_confusion_matrix_norm(cm_norm, file_out, label_strings, close_afterwards=True)



def plot_confusion_matrix_norm(conf_mat, file_out, label_strings, close_afterwards=False):
    '''
    Plot confusion matrix into graph
    :param conf_mat:
    :param file_out:
    :param label_strings:
    :param close_afterwards:
    :return:
    '''
    fig = plt.figure()
    plt.clf()
    ax = fig.add_subplot(111)
    ax.set_aspect(1)
    res = ax.imshow(conf_mat, cmap=plt.cm.jet,
                    interpolation='nearest')

    width, height = conf_mat.shape

    for x in xrange(width):
        for y in xrange(height):
            value = conf_mat[x][y]
            string_float = "{:.2f}".format(value)
            #str()
            ax.annotate(string_float, xy=(y, x),
                        horizontalalignment='center',
                        size=3,
                        verticalalignment='center')

    cb = fig.colorbar(res)

    #label_strings
    plt.xticks(range(len(label_strings)), label_strings, size='small', rotation='vertical')
    plt.yticks(range(len(label_strings)), label_strings, size='small')
    plt.savefig(file_out, format='png', dpi = 700)
    if close_afterwards:
        plt.close(fig)

def printIntoFigAll(test_accuracy, train_accuracy, fil_filename):
    range_ticks = range(test_accuracy.size)
    dpi_v = 500
    if use_x_server:
        plt.plot(range_ticks, test_accuracy, 'b')
        plt.plot(range_ticks, train_accuracy, 'r')
        plt.legend(['Test accuracy', 'Train accuracy'])
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.grid()
        plt.savefig(fil_filename, dpi = dpi_v)
        plt.savefig(fil_filename + ".svg") # svg is vector graphic, should be better. But unfortunately becomes too large with large number of epochs
    else:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(range_ticks, test_accuracy, 'b')
        ax.plot(range_ticks, train_accuracy, 'r')
        ax.legend(['Test accuracy', 'Train accuracy'])
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Accuracy')
        ax.grid()
        fig.savefig(fil_filename, dpi = dpi_v)
        fig.savefig(fil_filename + ".svg") # svg is vector graphic, should be better. But unfortunately becomes too large with large number of epochs

def get_number_parameters():
    total_parameters = 0
    for variable in tf.trainable_variables():
        shape = variable.get_shape()
        variable_parametes = 1
        for dim in shape:
            variable_parametes *= dim.value
        total_parameters += variable_parametes
    return total_parameters

def get_folder_out_m40():
    t = time()
    dataset = "m40"
    out_folder = "out/" + dataset + "_" + str(t) + "/"
    return out_folder

def get_folder_out_stanford():
    t = time()
    dataset = "stanford"
    out_folder = "out/" + dataset + "_" + str(t) + "/"
    return out_folder



def weight_variable(shape, stddev):
    initial = tf.truncated_normal(shape, stddev=stddev,seed=100)#here we can set seed for random initialization
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def conv3d(x, W):
    return tf.nn.conv3d(x, W, strides=[1, 1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')

def max_pool_3d(x):
    return tf.nn.max_pool3d(x,ksize=[1,2,2,2,1],strides=[1,1,1,1,1],padding='SAME')



def find_best_square_reshape_from_array(descriptor_size):
    best_square_size=int(round(math.sqrt(descriptor_size)))
    for idx in range(best_square_size):
        if round(descriptor_size/(float(best_square_size)-idx))==descriptor_size/(float(best_square_size)-idx):
            rows_num=(best_square_size-idx)
            cols_num=descriptor_size/(best_square_size-idx)
            break
        else:
            continue
    return rows_num,cols_num



#-------------------------------------------------------------------
# conv4d equivalent with dilation (taken from https://github.com/mhuen/TFScripts/blob/master/py/tfScripts.py#L5)
#-------------------------------------------------------------------
def conv4d_stacked(input,
                   filter,
                   strides=[1,1,1,1,1,1],
                   padding='SAME',
                   dilation_rate=None,
                   stack_axis=None,
                   stack_nested=False,
                   ):
    '''
      Computes a convolution over 4 dimensions.
      Python generalization of tensorflow's conv3d with dilation.
      conv4d_stacked uses tensorflows conv3d and stacks results along
      stack_axis.
  Parameters
  ----------
  input : A Tensor.
          Shape [batch, x_dim, y_dim, z_dim, t_dim, in_channels]
  filter: A Tensor. Must have the same type as input.
          Shape [x_dim, y_dim, z_dim, t_dim, in_channels, out_channels].
          in_channels must match between input and filter
  strides: A list of ints that has length 6. 1-D tensor of length 6.
           The stride of the sliding window for each dimension of input.
           Must have strides[0] = strides[5] = 1.
  padding: A string from: "SAME", "VALID". The type of padding algorithm to use.
  dilation_rate: Optional. Sequence of 4 ints >= 1.
                 Specifies the filter upsampling/input downsampling rate.
                 Equivalent to dilation_rate in tensorflows tf.nn.convolution
  stack_axis: Int
            Axis along which the convolutions will be stacked.
            By default the axis with the lowest output dimensionality will be
            chosen. This is only an educated guess of the best choice!
  stack_nested: Bool
            If set to True, this will stack in a for loop seperately and afterwards
            combine the results. In most cases slower, but maybe less memory needed.

  Returns
  -------
          A Tensor. Has the same type as input.
    '''
    # heuristically choose stack_axis
    if stack_axis == None:
        if dilation_rate == None:
            dil_array = np.ones(4)
        else:
            dil_array = np.asarray(dilation_rate)
        outputsizes = np.asarray(input.get_shape().as_list()[1:5])/np.asarray(strides[1:5])
        outputsizes -= dil_array*( np.asarray(filter.get_shape().as_list()[:4])-1)
        stack_axis = np.argmin(outputsizes)+1

    if dilation_rate != None:
        dilation_along_stack_axis = dilation_rate[stack_axis-1]
    else:
        dilation_along_stack_axis = 1

    tensors_t = tf.unstack(input,axis=stack_axis)
    kernel_t = tf.unstack(filter,axis=stack_axis-1)

    len_ts = filter.get_shape().as_list()[stack_axis-1]
    size_of_t_dim = input.get_shape().as_list()[stack_axis]

    if len_ts % 2 ==1:
        # uneven filter size: same size to left and right
        filter_l = int(len_ts/2)
        filter_r = int(len_ts/2)
    else:
        # even filter size: one more to right
        filter_l = int(len_ts/2) -1
        filter_r = int(len_ts/2)

    # The start index is important for strides and dilation
    # The strides start with the first element
    # that works and is VALID:
    start_index = 0
    if padding == 'VALID':
        for i in range(size_of_t_dim):
            if len( range(  max(i - dilation_along_stack_axis*filter_l,0),
                            min(i + dilation_along_stack_axis*filter_r+1,
                                size_of_t_dim),dilation_along_stack_axis)
                    ) == len_ts:
                # we found the first index that doesn't need padding
                break
        start_index = i
        print 'start_index',start_index

    # loop over all t_j in t
    result_t = []
    for i in range(start_index,size_of_t_dim,strides[stack_axis]):

        kernel_patch = []
        input_patch = []
        tensors_t_convoluted = []

        if padding == 'VALID':

            # Get indices t_s
            indices_t_s = range(  max(i - dilation_along_stack_axis*filter_l,0),
                                  min(i + dilation_along_stack_axis*filter_r+1,size_of_t_dim),
                                  dilation_along_stack_axis)

            # check if Padding = 'VALID'
            if len(indices_t_s) == len_ts:

                # sum over all remaining index_t_i in indices_t_s
                for j,index_t_i in enumerate(indices_t_s):
                    if not stack_nested:
                        kernel_patch.append(kernel_t[j])
                        input_patch.append(tensors_t[index_t_i])
                    else:
                        if dilation_rate != None:
                            tensors_t_convoluted.append( tf.nn.convolution(input=tensors_t[index_t_i],
                                                                           filter=kernel_t[j],
                                                                           strides=strides[1:stack_axis+1]+strides[stack_axis:5],
                                                                           padding=padding,
                                                                           dilation_rate=dilation_rate[:stack_axis-1]+dilation_rate[stack_axis:])
                                                         )
                        else:
                            tensors_t_convoluted.append( tf.nn.conv3d(input=tensors_t[index_t_i],
                                                                      filter=kernel_t[j],
                                                                      strides=strides[:stack_axis]+strides[stack_axis+1:],
                                                                      padding=padding)
                                                         )
                if stack_nested:
                    sum_tensors_t_s = tf.add_n(tensors_t_convoluted)
                    # put together
                    result_t.append(sum_tensors_t_s)

        elif padding == 'SAME':

            # Get indices t_s
            indices_t_s = range(i - dilation_along_stack_axis*filter_l,
                                (i + 1) + dilation_along_stack_axis*filter_r,
                                dilation_along_stack_axis)

            for kernel_j,j in enumerate(indices_t_s):
                # we can just leave out the invalid t coordinates
                # since they will be padded with 0's and therfore
                # don't contribute to the sum

                if 0 <= j < size_of_t_dim:
                    if not stack_nested:
                        kernel_patch.append(kernel_t[kernel_j])
                        input_patch.append(tensors_t[j])
                    else:
                        if dilation_rate != None:
                            tensors_t_convoluted.append( tf.nn.convolution(input=tensors_t[j],
                                                                           filter=kernel_t[kernel_j],
                                                                           strides=strides[1:stack_axis+1]+strides[stack_axis:5],
                                                                           padding=padding,
                                                                           dilation_rate=dilation_rate[:stack_axis-1]+dilation_rate[stack_axis:])
                                                         )
                        else:
                            tensors_t_convoluted.append( tf.nn.conv3d(input=tensors_t[j],
                                                                      filter=kernel_t[kernel_j],
                                                                      strides=strides[:stack_axis]+strides[stack_axis+1:],
                                                                      padding=padding)
                                                         )
            if stack_nested:
                sum_tensors_t_s = tf.add_n(tensors_t_convoluted)
                # put together
                result_t.append(sum_tensors_t_s)

        if not stack_nested:
            if kernel_patch:
                kernel_patch = tf.concat(kernel_patch,axis=3)
                input_patch = tf.concat(input_patch,axis=4)
                if dilation_rate != None:
                    result_patch = tf.nn.convolution(input=input_patch,
                                                     filter=kernel_patch,
                                                     strides=strides[1:stack_axis]+strides[stack_axis+1:5],
                                                     padding=padding,
                                                     dilation_rate=dilation_rate[:stack_axis-1]+dilation_rate[stack_axis:])
                else:
                    result_patch = tf.nn.conv3d(input=input_patch,
                                                filter=kernel_patch,
                                                strides=strides[:stack_axis]+strides[stack_axis+1:],
                                                padding=padding)
                result_t.append(result_patch)

    # stack together
    return tf.stack(result_t,axis=stack_axis)
