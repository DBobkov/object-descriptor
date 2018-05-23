#/usr/bin/python2

import tensorflow as tf
import common_utils_nn


############ Defining the network
def get_2dcnn_our_model(feature_set_training, label_set_training, param):
    input_feature = feature_set_training

    descriptor_size = input_feature.shape[1]
    N_classes = label_set_training.shape[1]

    assert(N_classes>1) # TOP HAT NOTATION REQUIRED!

    # input layer
    x = tf.placeholder(tf.float32, shape=[None, descriptor_size])
    y_ = tf.placeholder(tf.float32, shape=[None, N_classes])
    x_image = tf.reshape(x, [-1, 20, 4, 5, 3, 1])  # be careful of shape consistency

    do_reshaping = True
    if do_reshaping:
        ###### Reshaping
        rows_num, cols_num = common_utils_nn.find_best_square_reshape_from_array(descriptor_size)
        print('Using desc size %d, row %d, col %d' % (descriptor_size, rows_num, cols_num))
        h_pool1_reshape = tf.reshape(x_image, [-1, rows_num, cols_num, 1])

        ###### conv layer 1
        W_conv1 = common_utils_nn.weight_variable([5, 5, 1, param.layer1_feature_maps_num], param.init_stddev)
        b_conv1 = common_utils_nn.bias_variable([param.layer1_feature_maps_num])
        h_conv1 = tf.nn.relu(common_utils_nn.conv2d(h_pool1_reshape, W_conv1) + b_conv1)
    else:
        ###### conv layer 1
        W_conv1 = common_utils_nn.weight_variable([5, 5, 1, param.layer1_feature_maps_num], param.init_stddev)
        b_conv1 = common_utils_nn.bias_variable([param.layer1_feature_maps_num])
        h_conv1 = tf.nn.relu(common_utils_nn.conv2d(x_image, W_conv1) + b_conv1)

    ###### conv layer 2
    W_conv2 = common_utils_nn.weight_variable([5, 5, param.layer1_feature_maps_num, param.layer2_feature_maps_num], param.init_stddev)
    b_conv2 = common_utils_nn.bias_variable([param.layer2_feature_maps_num])
    h_conv2 = tf.nn.relu(common_utils_nn.conv2d(h_conv1, W_conv2) + b_conv2)

    ###### conv layer 3
    W_conv3 = common_utils_nn.weight_variable([5, 5, param.layer2_feature_maps_num, param.layer3_feature_maps_num], param.init_stddev)
    b_conv3 = common_utils_nn.bias_variable([param.layer3_feature_maps_num])
    h_conv3 = tf.nn.relu(common_utils_nn.conv2d(h_conv2, W_conv3) + b_conv3)
    # pooling
    h_pool3 = common_utils_nn.max_pool_2x2(h_conv3)
    # pooling again
    h_pool_final_flat = tf.reshape(h_pool3, [-1, h_pool3.shape[1].value * h_pool3.shape[2].value * param.layer3_feature_maps_num])  # be careful of shape consistency

    ###### fully connected layer 1
    W_fc1 = common_utils_nn.weight_variable([h_pool3.shape[1].value * h_pool3.shape[2].value * param.layer3_feature_maps_num,
                                             param.layer4_fc_num], param.init_stddev)  # be careful of shape consistency
    b_fc1 = common_utils_nn.bias_variable([param.layer4_fc_num])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool_final_flat, W_fc1) + b_fc1)

    ###### dropout layer
    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    W_fc2 = common_utils_nn.weight_variable([param.layer4_fc_num, N_classes], param.init_stddev)
    b_fc2 = common_utils_nn.bias_variable([N_classes])
    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

    layers_var = {'h_conv1' : h_conv1,
                  'h_conv2' : h_conv2,
                  'h_conv3' : h_conv3}

    return x, y_conv, y_, keep_prob, layers_var

import numpy as np

############ Defining the network
def get_3dcnn_our_model(feature_set_training, label_set_training, param):
    input_feature = feature_set_training

    desc_dims = np.array(([20,4,5,3]))

    descriptor_size = input_feature.shape[1]
    N_classes = label_set_training.shape[1]

    assert (N_classes > 1)  # TOP HAT NOTATION REQUIRED!

    # input layer
    x = tf.placeholder(tf.float32, shape=[None, descriptor_size])
    y_ = tf.placeholder(tf.float32, shape=[None, N_classes])
    x_image = tf.reshape(x, [-1, 20, 4, 5, 3, 1])  # be careful of shape consistency

    ###### Reshape into 3d
    h_pool1_reshape = tf.reshape(x_image, [-1, 20, 4, 15, 1])

    ###### conv3d layer 1
    W_conv1 = common_utils_nn.weight_variable([5, 5, 1, 1, param.layer1_feature_maps_num], param.init_stddev)
    b_conv1 = common_utils_nn.bias_variable([param.layer1_feature_maps_num])
    h_conv1 = tf.nn.relu(common_utils_nn.conv3d(h_pool1_reshape, W_conv1) + b_conv1)


    ###### conv3d layer 2
    W_conv2 = common_utils_nn.weight_variable([5, 5, 1, param.layer1_feature_maps_num, param.layer2_feature_maps_num],
                                              param.init_stddev)
    b_conv2 = common_utils_nn.bias_variable([param.layer2_feature_maps_num])
    h_conv2 = tf.nn.relu(common_utils_nn.conv3d(h_conv1, W_conv2) + b_conv2)

    ###### conv3d layer 3
    W_conv3 = common_utils_nn.weight_variable([5, 5, 1, param.layer2_feature_maps_num, param.layer3_feature_maps_num], param.init_stddev)
    b_conv3 = common_utils_nn.bias_variable([param.layer3_feature_maps_num])
    h_conv3 = tf.nn.relu(common_utils_nn.conv3d(h_conv2, W_conv3) + b_conv3)

    ###### Reshape into 3d
    rows_num, cols_num = common_utils_nn.find_best_square_reshape_from_array(
        descriptor_size)
    h_conv3_reshape = tf.reshape(h_conv3, [-1, rows_num, cols_num, 1])

    # pooling
    h_pool3 = common_utils_nn.max_pool_2x2(h_conv3_reshape)
    # pooling again
    h_pool_final_flat = tf.reshape(h_pool3, [-1, h_pool3.shape[1].value * h_pool3.shape[2].value * param.layer3_feature_maps_num])  # be careful of shape consistency

    ###### fully connected layer 1
    W_fc1 = common_utils_nn.weight_variable(
        [h_pool3.shape[1].value * h_pool3.shape[2].value * param.layer3_feature_maps_num,
         param.layer4_fc_num], param.init_stddev)  # be careful of shape consistency
    b_fc1 = common_utils_nn.bias_variable([param.layer4_fc_num])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool_final_flat, W_fc1) + b_fc1)

    ###### dropout layer
    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    W_fc2 = common_utils_nn.weight_variable([param.layer4_fc_num, N_classes], param.init_stddev)
    b_fc2 = common_utils_nn.bias_variable([N_classes])
    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

    layers_var = {'h_conv1': h_conv1,
                  'h_conv2': h_conv2,
                  'h_conv3': h_conv3}

    return x, y_conv, y_, keep_prob, layers_var


############ Defining the network
def get_4dcnn_our_model(feature_set_training, label_set_training, param):
    input_feature = feature_set_training

    descriptor_size = input_feature.shape[1]
    N_classes = label_set_training.shape[1]
    assert(N_classes>1) # TOP HAT NOTATION REQUIRED!

    # input layer
    x = tf.placeholder(tf.float32, shape=[None, descriptor_size])
    y_ = tf.placeholder(tf.float32, shape=[None, N_classes])
    x_image = tf.reshape(x, [-1, 20, 4, 5, 3, 1])  # be careful of shape consistency

    ###### conv layer 1 4d
    W_conv1 = common_utils_nn.weight_variable([5, 2, 2, 1, 1, param.layer1_feature_maps_num], param.init_stddev)
    b_conv1 = common_utils_nn.bias_variable([param.layer1_feature_maps_num])
    h_conv1 = tf.nn.relu(common_utils_nn.conv4d_stacked(x_image, W_conv1, stack_axis = 1) + b_conv1)

    ###### conv layer 2 4d
    W_conv2 = common_utils_nn.weight_variable([5, 2, 2, 1,  param.layer1_feature_maps_num, param.layer2_feature_maps_num], param.init_stddev)
    b_conv2 = common_utils_nn.bias_variable([param.layer2_feature_maps_num])
    h_conv2 = tf.nn.relu(common_utils_nn.conv4d_stacked(h_conv1, W_conv2, stack_axis = 1) + b_conv2)

    ###### Reshaping
    rows_num,cols_num = common_utils_nn.find_best_square_reshape_from_array(descriptor_size) #is this correct? # was 1200
    h_pool1_reshape = tf.reshape(h_conv2, [-1, rows_num, cols_num , param.layer2_feature_maps_num])

    ###### conv layer 3python wrapper_stanford_train.py --batch_size 88 --log_dir stanford10_wahl/ --dataset stanford10OurSplitNew --gpu 0 --max_epoch 1500
    W_conv3 = common_utils_nn.weight_variable([5, 5, param.layer2_feature_maps_num, param.layer3_feature_maps_num], param.init_stddev)
    b_conv3 = common_utils_nn.bias_variable([param.layer3_feature_maps_num])
    h_conv3 = tf.nn.relu(common_utils_nn.conv2d(h_pool1_reshape, W_conv3) + b_conv3)

    # pooling
    h_pool3 = common_utils_nn.max_pool_2x2(h_conv3)
    # pooling again
    h_pool_final_flat = tf.reshape(h_pool3, [-1, h_pool3.shape[1].value * h_pool3.shape[2].value * param.layer3_feature_maps_num])  # be careful of shape consistency

    ###### fully connected layer 1
    W_fc1 = common_utils_nn.weight_variable([h_pool3.shape[1].value * h_pool3.shape[2].value * param.layer3_feature_maps_num,
                                             param.layer4_fc_num], param.init_stddev)  # be careful of shape consistency
    b_fc1 = common_utils_nn.bias_variable([param.layer4_fc_num])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool_final_flat, W_fc1) + b_fc1)

    ###### dropout layer
    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    W_fc2 = common_utils_nn.weight_variable([param.layer4_fc_num, N_classes], param.init_stddev)
    b_fc2 = common_utils_nn.bias_variable([N_classes])
    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

    layers_var = {'h_conv1' : h_conv1,
                  'h_conv2' : h_conv2,
                  'h_conv3' : h_conv3}

    return x, y_conv, y_, keep_prob, layers_var



############ Defining the network
def get_3dcnn_wahl_model(feature_set_training,
                         label_set_training, param):
    '''
    Get 3D convolution based neural network model
    :param feature_set_training:
    :param label_set_training:
    :param param:
    :return:
    '''
    input_feature = feature_set_training

    desc_dims = np.array(([6,6,6,6]))

    descriptor_size = input_feature.shape[1]
    N_classes = label_set_training.shape[1]

    assert (N_classes > 1)  # TOP HAT NOTATION REQUIRED!

    # input layer
    x = tf.placeholder(tf.float32, shape=[None, descriptor_size])
    y_ = tf.placeholder(tf.float32, shape=[None, N_classes])
    x_image = tf.reshape(x, [-1, desc_dims[0], desc_dims[1], desc_dims[2], desc_dims[3], 1])  # be careful of shape consistency

    ###### Reshape into 3d
    h_pool1_reshape = tf.reshape(x_image, [-1, 9, 9, 16, 1])

    ###### conv3d layer 1
    W_conv1 = common_utils_nn.weight_variable([5, 5, 1, 1, param.layer1_feature_maps_num], param.init_stddev)
    b_conv1 = common_utils_nn.bias_variable([param.layer1_feature_maps_num])
    h_conv1 = tf.nn.relu(common_utils_nn.conv3d(h_pool1_reshape, W_conv1) + b_conv1)


    ###### conv3d layer 2
    W_conv2 = common_utils_nn.weight_variable([5, 5, 1, param.layer1_feature_maps_num, param.layer2_feature_maps_num],
                                              param.init_stddev)
    b_conv2 = common_utils_nn.bias_variable([param.layer2_feature_maps_num])
    h_conv2 = tf.nn.relu(common_utils_nn.conv3d(h_conv1, W_conv2) + b_conv2)

    ###### conv3d layer 3
    W_conv3 = common_utils_nn.weight_variable([5, 5, 1, param.layer2_feature_maps_num, param.layer3_feature_maps_num], param.init_stddev)
    b_conv3 = common_utils_nn.bias_variable([param.layer3_feature_maps_num])
    h_conv3 = tf.nn.relu(common_utils_nn.conv3d(h_conv2, W_conv3) + b_conv3)

    ###### Reshape into 3d
    rows_num, cols_num = common_utils_nn.find_best_square_reshape_from_array(descriptor_size)  # is this correct? # was 1200
    h_conv3_reshape = tf.reshape(h_conv3, [-1, rows_num, cols_num, 1])

    # pooling
    h_pool3 = common_utils_nn.max_pool_2x2(h_conv3_reshape)
    # pooling again
    h_pool_final_flat = tf.reshape(h_pool3, [-1, h_pool3.shape[1].value * h_pool3.shape[2].value * param.layer3_feature_maps_num])  # be careful of shape consistency

    ###### fully connected layer 1
    W_fc1 = common_utils_nn.weight_variable(
        [h_pool3.shape[1].value * h_pool3.shape[2].value * param.layer3_feature_maps_num,
         param.layer4_fc_num], param.init_stddev)  # be careful of shape consistency
    b_fc1 = common_utils_nn.bias_variable([param.layer4_fc_num])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool_final_flat, W_fc1) + b_fc1)

    ###### dropout layer
    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    W_fc2 = common_utils_nn.weight_variable([param.layer4_fc_num, N_classes], param.init_stddev)
    b_fc2 = common_utils_nn.bias_variable([N_classes])
    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

    layers_var = {'h_conv1': h_conv1,
                  'h_conv2': h_conv2,
                  'h_conv3': h_conv3}

    return x, y_conv, y_, keep_prob, layers_var


def get_2dcnn_wahl_model(feature_set_training, label_set_training, param):
    '''
    Get 2D convolution based neural network based model
    :param feature_set_training:
    :param label_set_training:
    :param param:
    :return:
    '''
    input_feature = feature_set_training

    descriptor_size = input_feature.shape[1]
    N_classes = label_set_training.shape[1]
    assert(N_classes>1) # TOP HAT NOTATION REQUIRED!

    # input layer
    x = tf.placeholder(tf.float32, shape=[None, descriptor_size])
    y_ = tf.placeholder(tf.float32, shape=[None, N_classes])
    x_image = tf.reshape(x, [-1, 6, 6, 6, 6, 1])  # be careful of shape consistency

    ###### Reshaping
    rows_num, cols_num = common_utils_nn.find_best_square_reshape_from_array(descriptor_size)
    print('Using desc size %d, reshaping into rows %d x cols %d' % (descriptor_size, rows_num, cols_num))
    h_pool1_reshape = tf.reshape(x_image, [-1, rows_num, cols_num, 1])

    ###### conv layer 1
    W_conv1 = common_utils_nn.weight_variable([5, 5, 1, param.layer1_feature_maps_num], param.init_stddev)
    b_conv1 = common_utils_nn.bias_variable([param.layer1_feature_maps_num])
    h_conv1 = tf.nn.relu(common_utils_nn.conv2d(h_pool1_reshape, W_conv1) + b_conv1)


    ###### conv layer 2
    W_conv2 = common_utils_nn.weight_variable([5, 5, param.layer1_feature_maps_num, param.layer2_feature_maps_num], param.init_stddev)
    b_conv2 = common_utils_nn.bias_variable([param.layer2_feature_maps_num])
    h_conv2 = tf.nn.relu(common_utils_nn.conv2d(h_conv1, W_conv2) + b_conv2)

    ###### conv layer 3
    W_conv3 = common_utils_nn.weight_variable([5, 5, param.layer2_feature_maps_num, param.layer3_feature_maps_num], param.init_stddev)
    b_conv3 = common_utils_nn.bias_variable([param.layer3_feature_maps_num])
    h_conv3 = tf.nn.relu(common_utils_nn.conv2d(h_conv2, W_conv3) + b_conv3)

    # pooling
    h_pool3 = common_utils_nn.max_pool_2x2(h_conv3)
    # pooling again
    h_pool_final_flat = tf.reshape(h_pool3, [-1, h_pool3.shape[1].value * h_pool3.shape[2].value * param.layer3_feature_maps_num])  # be careful of shape consistency

    ###### fully connected layer 1
    W_fc1 = common_utils_nn.weight_variable([h_pool3.shape[1].value * h_pool3.shape[2].value * param.layer3_feature_maps_num,
                                             param.layer4_fc_num], param.init_stddev)  # be careful of shape consistency
    b_fc1 = common_utils_nn.bias_variable([param.layer4_fc_num])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool_final_flat, W_fc1) + b_fc1)

    ###### dropout layer
    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    W_fc2 = common_utils_nn.weight_variable([param.layer4_fc_num, N_classes], param.init_stddev)
    b_fc2 = common_utils_nn.bias_variable([N_classes])
    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

    layers_var = {'h_conv1' : h_conv1,
                  'h_conv2' : h_conv2,
                  'h_conv3' : h_conv3}

    return x, y_conv, y_, keep_prob, layers_var


############ Defining the network
def get_4dcnn_wahl_model(feature_set_training, label_set_training, param):
    input_feature = feature_set_training

    descriptor_size = input_feature.shape[1]
    N_classes = label_set_training.shape[1]
    assert(N_classes>1) # TOP HAT NOTATION REQUIRED!

    # input layer
    x = tf.placeholder(tf.float32, shape=[None, descriptor_size])
    y_ = tf.placeholder(tf.float32, shape=[None, N_classes])
    x_image = tf.reshape(x, [-1, 6, 6, 6, 6, 1])  # be careful of shape consistency

    ###### conv layer 1 4d
    W_conv1 = common_utils_nn.weight_variable([3, 3, 3, 1, 1, param.layer1_feature_maps_num], param.init_stddev)
    b_conv1 = common_utils_nn.bias_variable([param.layer1_feature_maps_num])
    h_conv1 = tf.nn.relu(common_utils_nn.conv4d_stacked(x_image, W_conv1, stack_axis = 1) + b_conv1)

    ###### conv layer 2 4d
    W_conv2 = common_utils_nn.weight_variable([3, 3, 3, 1,  param.layer1_feature_maps_num, param.layer2_feature_maps_num], param.init_stddev)
    b_conv2 = common_utils_nn.bias_variable([param.layer2_feature_maps_num])
    h_conv2 = tf.nn.relu(common_utils_nn.conv4d_stacked(h_conv1, W_conv2, stack_axis = 1) + b_conv2)

    ###### Reshaping
    rows_num, cols_num = common_utils_nn.find_best_square_reshape_from_array(descriptor_size)
    h_pool1_reshape = tf.reshape(h_conv2, [-1, rows_num, cols_num , param.layer2_feature_maps_num])

    ###### conv layer 3
    W_conv3 = common_utils_nn.weight_variable([4, 4, param.layer2_feature_maps_num, param.layer3_feature_maps_num], param.init_stddev)
    b_conv3 = common_utils_nn.bias_variable([param.layer3_feature_maps_num])
    h_conv3 = tf.nn.relu(common_utils_nn.conv2d(h_pool1_reshape, W_conv3) + b_conv3)

    # pooling
    h_pool3 = common_utils_nn.max_pool_2x2(h_conv3)

    # pooling again
    h_pool_final_flat = tf.reshape(h_pool3, [-1, h_pool3.shape[1].value * h_pool3.shape[2].value * param.layer3_feature_maps_num])  # be careful of shape consistency

    ###### fully connected layer 1
    W_fc1 = common_utils_nn.weight_variable([h_pool3.shape[1].value * h_pool3.shape[2].value * param.layer3_feature_maps_num,
                                             param.layer4_fc_num], param.init_stddev)  # be careful of shape consistency
    b_fc1 = common_utils_nn.bias_variable([param.layer4_fc_num])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool_final_flat, W_fc1) + b_fc1)

    ###### dropout layer
    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    W_fc2 = common_utils_nn.weight_variable([param.layer4_fc_num, N_classes], param.init_stddev)
    b_fc2 = common_utils_nn.bias_variable([N_classes])
    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

    layers_var = {'h_conv1' : h_conv1,
                  'h_conv2' : h_conv2,
                  'h_conv3' : h_conv3}

    return x, y_conv, y_, keep_prob, layers_var