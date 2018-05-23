#/usr/bin/python2
import fourd_cnn_proposed
import dataset_utils
import argparse
import LearningParameters
import dataset_constants

if __name__ == "__main__":
    #############
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
    parser.add_argument('--log_dir', default='log', help='Log dir [default: log]')
    parser.add_argument('--max_epoch', type=int, default=250, help='Epoch to run [default: 250]')
    parser.add_argument('--batch_size', type=int, default=88, help='Batch Size during training [default: 32]')
    parser.add_argument('--learning_rate', type=float, default=0.0005, help='Initial learning rate [default: 0.0005]')
    parser.add_argument('--init_stddev', type=float, default=0.25, help='Initialization std dev of Gaussian distribution [default: 0.25]')
    parser.add_argument('--dataset', default='scannet_shapenet55', help='stanford13OurSplit or stanford10OurSplit [default: stanford10OurSplit]')
    parser.add_argument('--dropout_prob', type=float, default=0.5, help='Epoch to run [default: 0.5]')
    parser.add_argument('--conv_dim', type=int, default=4, help='Dimensionality of input convolutions [default: 4]')
    parser.add_argument('--desc_type', type=int, default=0, help='0 is our descriptor, 1 is wahl et al. descriptor [default: 0]')
    FLAGS = parser.parse_args()

    #############
    param = LearningParameters.LearningParameters()
    param.gpu_index = FLAGS.gpu
    param.log_dir = FLAGS.log_dir + '/'
    param.max_epoch = FLAGS.max_epoch
    param.batch_size = FLAGS.batch_size
    param.init_stddev = FLAGS.init_stddev
    param.adam_opt_learn_rate = FLAGS.learning_rate
    param.dropout_prob = FLAGS.dropout_prob
    param.inputWahlDescriptor = FLAGS.desc_type!=0
    param.dim_model = FLAGS.conv_dim

    dir = dataset_constants.stanford_desc_dataset_dict.get(FLAGS.dataset, 'none')
    if dir == 'none':
        print('Unsupported dataset type %s, exit!' % FLAGS.dataset)
        exit(1)
    param.input_dir = dir

    dict_labels = dataset_utils.stanford_label_to_int_dict
    if param.inputWahlDescriptor:
        descriptor_to_use = 'wahl_long_desc_run_0_'
    else:
        descriptor_to_use = 'merged_proposed_kl_bugfix4_desc_run_0_'
        
    stanford_dataset = dataset_utils.Dataset(param.input_dir, dict_labels, descriptor_to_use) 

    ## check input labels
    N_classes = len(dict_labels)
    classes_train = stanford_dataset.train_label.shape[1]
    classes_test = stanford_dataset.test_label.shape[1]
    assert(N_classes == classes_train)
    assert (N_classes == classes_test)
    print('Used classes %d, train %d test %d' % (N_classes, classes_train, classes_test))
    
    # we check the dimension of this nparray to ensure our data is correct
    N_train = stanford_dataset.train_feat.shape
    N_l_train = stanford_dataset.train_label.shape
    print("Training dataset on stanford, N_train %dx%d, N_label_train %dx%d" % (N_train[0], N_train[1], N_l_train[0], N_l_train[1]))

    N_test = stanford_dataset.test_feat.shape
    N_l_test = stanford_dataset.test_label.shape
    print("Test on stanford, N_test %dx%d, N_label_test %dx%d" % (N_test[0], N_test[1], N_l_test[0], N_l_test[1]))

    fourd_cnn_proposed.train(param, stanford_dataset)