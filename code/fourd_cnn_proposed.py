#/usr/bin/python2
import numpy as np
import tensorflow as tf
import os
import common_utils_nn
import nn_models
from time import time

epoch_to_write_plots_models = 2 # how often we plot models (how many epochs)


def get_loss(pred, label):
    '''
    Set the classification loss for the given problem
    :param pred:
    :param label:
    :return:
    '''
    loss = tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=label)
    classify_loss = tf.reduce_mean(loss)
    tf.summary.scalar('classify loss', classify_loss)
    return classify_loss


def get_learning_rate(batch, param):
    '''
    Setup the learning rate
    :param batch:
    :param param:
    :return:
    '''
    base_lr = param.base_lr
    batch_size = param.batch_size
    decay_step = param.decay_step
    decay_rate = param.decay_rate

    learning_rate = tf.train.exponential_decay(
                        base_lr,  # Base learning rate.
                        batch * batch_size,  # Current index into the dataset.
                        decay_step,          # Decay step.
                        decay_rate,          # Decay rate.
                        staircase=True)
    learning_rate = tf.maximum(learning_rate, 0.00001) # CLIP THE LEARNING RATE!
    return learning_rate

def get_bn_decay(batch, param):
    '''
    Set exponential decay (optimization with momentum)
    :param batch:
    :param param:
    :return:
    '''
    BN_INIT_DECAY = param.bn_init_decay
    batch_size = param.batch_size
    BN_DECAY_DECAY_STEP = param.bn_decay_decay_step
    BN_DECAY_DECAY_RATE = param.bn_decay_decay_rate
    BN_DECAY_CLIP = param.bn_decay_clip

    bn_momentum = tf.train.exponential_decay(
                      BN_INIT_DECAY,
                      batch*batch_size,
                      BN_DECAY_DECAY_STEP,
                      BN_DECAY_DECAY_RATE,
                      staircase=True)

    bn_decay = tf.minimum(BN_DECAY_CLIP, 1 - bn_momentum)
    return bn_decay


def set_device_name(gpu_index):
    if gpu_index<0:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    else:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
        # doing this, because nothing else works, from https://stackoverflow.com/questions/37893755/tensorflow-set-cuda-visible-devices-within-jupyter


def init_training(param, dataset):
    '''
    Wrapper for training initialization
    :param param:
    :param dataset:
    :return:
    '''

    if param.inputWahlDescriptor==True:
        if param.dim_model == 4:
            x, y_pred, y_true, keep_prob, layers_var = nn_models.get_4dcnn_wahl_model(dataset.train_feat, dataset.train_label, param)
        elif param.dim_model == 3:
            x, y_pred, y_true, keep_prob, layers_var = nn_models.get_3dcnn_wahl_model(dataset.train_feat, dataset.train_label, param)
        elif param.dim_model == 2:
            x, y_pred, y_true, keep_prob, layers_var = nn_models.get_2dcnn_wahl_model(dataset.train_feat, dataset.train_label, param)
        else:
            exit(-1)
    else:
        if param.dim_model==4:
            x, y_pred, y_true, keep_prob, layers_var = nn_models.get_4dcnn_our_model(dataset.train_feat,
                                                                                        dataset.train_label, param)
        elif param.dim_model == 3:
            x, y_pred, y_true, keep_prob, layers_var = nn_models.get_3dcnn_our_model(dataset.train_feat,
                                                                                 dataset.train_label, param)
        elif param.dim_model == 2:
            x, y_pred, y_true, keep_prob, layers_var = nn_models.get_2dcnn_our_model(dataset.train_feat,
                                                                                 dataset.train_label, param)
        else:
            exit(-1)

    # setup metrics
    #######
    batch = tf.Variable(0)
    bn_decay = get_bn_decay(batch, param)
    tf.summary.scalar('bn_decay', bn_decay)

    loss = get_loss(y_pred, y_true)
    tf.summary.scalar('loss', loss)

    correct = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y_true, 1))
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
    tf.summary.scalar('accuracy', accuracy)

    # Get training operator
    learning_rate = get_learning_rate(batch, param)
    tf.summary.scalar('learning_rate', learning_rate)

    ####### Set up the optimizer
    optimizer = tf.train.AdamOptimizer(learning_rate)

    train_op = optimizer.minimize(loss, global_step=batch)

    num_classes = dataset.train_label.shape[1]
    print('Number of classes used %d' % num_classes)

    confusion_matrix_tf = tf.confusion_matrix(tf.argmax(y_true, 1), tf.argmax(y_pred, 1), num_classes)

    assert( dataset.test_label.shape[1] == num_classes )


    config = tf.ConfigProto()
    config.gpu_options.allow_growth = False
    config.allow_soft_placement = True
    config.log_device_placement = False
    sess = tf.Session(config = config)
    sess.run(tf.global_variables_initializer())
    sess.as_default()


    merged = tf.summary.merge_all()
    log_dir = param.log_dir + '/train/'
    train_writer = tf.summary.FileWriter(log_dir, sess.graph)

    # Setup the variables in dict for our network
    ops = {'y_true': y_true,
        'y_pred': y_pred,
        'loss': loss,
        'x': x,
        'keep_prob': keep_prob,
        'train_op': train_op,
        'merged': merged,
        'step': batch,
        'conf_mat': confusion_matrix_tf,
        'sess': sess,
        'accuracy': accuracy}

    ops.update(layers_var)

    return ops, train_writer

def log_string(log_out, out_str):
    ''''
    Log to file and get string
    '''
    log_out.write(out_str+'\n')
    log_out.flush()
    print(out_str)

def train(param, dataset):
    '''
    Wrapper for training using the given parameters and dataset
    :param param:
    :param dataset:
    :return:
    '''

    set_device_name(param.gpu_index)
    ops, train_writer = init_training(param, dataset)

    keep_prob = 1.0 - param.dropout_prob

    log_stats = common_utils_nn.OptStats()
    if not os.path.exists(param.log_dir):
        os.makedirs(param.log_dir)

    log_out = open(os.path.join(param.log_dir, 'log_train.txt'), 'w')

    model_saver = tf.train.Saver()
    curr_model_folder = param.log_dir + '/current_model/'
    best_model_folder = param.log_dir + '/best_model/'
    if not os.path.exists(curr_model_folder):
        os.makedirs(curr_model_folder)
    if not os.path.exists(best_model_folder):
        os.makedirs(best_model_folder)

    log_stats.train_acc_arr = np.zeros((param.max_epoch,))
    log_stats.test_acc_arr = np.zeros((param.max_epoch,))

    #############

    total_parameters = common_utils_nn.get_number_parameters()
    log_string(log_out, "Total parameters %d" % total_parameters)

    #############
    # some pre-allocation   
    concatenate_training_set = np.column_stack((dataset.train_feat, dataset.train_label))
    numb_instances, desc_size = dataset.train_feat.shape

    label_strings = dataset.label_strings

    ########## save parameters used for training
    param.save_parameters(param.log_dir)
    ##########
    time_start_all = time()

    for epoch in range(param.max_epoch):
        t1 = time()

        ### TRAINING
        np.random.shuffle(concatenate_training_set)  # randomly shuffle the training dataset, very important for good performance

        common_utils_nn.sgd_training_one_epoch(concatenate_training_set,
                                            param.batch_size,
                                            desc_size,
                                            keep_prob,
                                            ops['sess'],
                                            ops['train_op'],
                                            ops['x'],
                                            ops['y_true'],
                                            ops['keep_prob'],
                                            ops['merged'],
                                            train_writer)

        ## TRAINING accuracy
        train_accuracy = common_utils_nn.evaluate_accuracy(dataset.train_feat,
                                                        dataset.train_label,
                                                        param.batch_size,
                                                        ops['keep_prob'],
                                                        ops['accuracy'],
                                                        ops['x'],
                                                        ops['y_true'],
                                                        ops['sess'])

        ### TESTING
        test_accuracy = common_utils_nn.evaluate_accuracy( dataset.test_feat,
                                                        dataset.test_label,
                                                        param.batch_size,
                                                        ops['keep_prob'],
                                                        ops['accuracy'],
                                                        ops['x'],
                                                        ops['y_true'],
                                                        ops['sess'])

        log_stats.train_acc_arr[epoch] = train_accuracy
        log_stats.test_acc_arr[epoch] = test_accuracy

        ##### conf mat stat
        normalized = False


        cm = common_utils_nn.getConfusionMatrix(ops['sess'],
                                                ops['x'],
                                                ops['y_true'],
                                                ops['keep_prob'],
                                                ops['conf_mat'],
                                                dataset.test_feat,
                                                dataset.test_label,
                                                normalized )
        
        metric = common_utils_nn.computeClassificationMetricFromConfMat(cm)

        #### test mean class accuracy
        class_accuracy = np.diagonal(cm) / cm.sum(axis=1)
        class_accuracy = np.nan_to_num(class_accuracy)
        test_mean_acc = np.mean(class_accuracy)

        ####
        if log_stats.best_test_acc<test_accuracy:
            log_stats.best_test_acc = test_accuracy
            log_stats.best_test_at_epoch = epoch

            log_stats.best_avg_test_acc = test_mean_acc
            model_saver.save(ops['sess'], best_model_folder + '/model.ckpt')

            common_utils_nn.writeConfusionMatrix(ops['sess'],
                                                    ops['x'],
                                                    ops['y_true'],
                                                    ops['keep_prob'],
                                                    ops['conf_mat'],
                                                    dataset.test_feat,
                                                    dataset.test_label,
                                                    label_strings,
                                                    best_model_folder)


        write_data = epoch % epoch_to_write_plots_models == 0
        if write_data:
            common_utils_nn.writeConfusionMatrix(ops['sess'],
                                                    ops['x'],
                                                    ops['y_true'],
                                                    ops['keep_prob'],
                                                    ops['conf_mat'],
                                                    dataset.test_feat,
                                                    dataset.test_label,
                                                    label_strings,
                                                    param.log_dir)

            common_utils_nn.printIntoFigAll(log_stats.test_acc_arr[:epoch + 1],
                                            log_stats.train_acc_arr[:epoch + 1],
                                            param.log_dir + 'current.png')
            # Train the model and save it in the end (based on https://stackoverflow.com/questions/33759623/tensorflow-how-to-save-restore-a-model)
            model_saver.save(ops['sess'], curr_model_folder + "/model.ckpt")

        time_per_epoch = time() - t1
        if epoch>=0:
            diff_train_accuracy = log_stats.train_acc_arr[epoch] - log_stats.train_acc_arr[epoch-1]
            diff_test_accuracy = log_stats.test_acc_arr[epoch] - log_stats.test_acc_arr[epoch - 1]

            log_string(log_out, "epoch %d/%d, train accuracy %g (diff %g); test accuracy %g (diff %g, avg %f, f1-score %f). Best test acc so far %g (avg %f). Time per epoch %f s" %
                (epoch, param.max_epoch,
                train_accuracy, diff_train_accuracy,
                test_accuracy, diff_test_accuracy,
                test_mean_acc, metric.f1Score,
                log_stats.best_test_acc, log_stats.best_avg_test_acc,
                time_per_epoch))

        np.save(param.log_dir + "/history.npy", log_stats.test_acc_arr[:epoch])

        time_elapsed_total = time() - time_start_all
        log_string(log_out, 'Finished the entire training into %s. It took %f seconds' % (param.log_dir, time_elapsed_total))