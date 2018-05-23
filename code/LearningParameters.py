#/usr/bin/python2
import json

class LearningParameters:
    batch_size = 1000  # 0.045 # 0.06#0.06#0.17#06# #BEST 0.145
    max_epoch = 3000
    dropout_prob = 0.5  # 0.5
    adam_opt_learn_rate = 5 * 1e-4  # e-2 too large, e-3 fine, e-4 too slow

    # Reduced network size (comparable to (Qi et al., pointnet 2017) now! 7,413,906 parameters)
    layer1_feature_maps_num = 32
    layer2_feature_maps_num = 64
    layer3_feature_maps_num = 48
    layer4_fc_num = 1024

    init_stddev = 0.25
    gpu_index = -1
    log_dir = "a"
    input_dir = "a"
    def save_parameters(self, out_folder):
        a = json.dumps(self.__dict__)
        filename = out_folder + "param.txt"
        text_file = open(filename, "w")
        text_file.write(a)
        text_file.close()

    base_lr = 0.001
    decay_step = 200000
    decay_rate = 0.7
    bn_init_decay = 0.5
    bn_decay_decay_step = 200000
    bn_decay_decay_rate =  0.5
    bn_decay_clip = 0.99
    momentum = 0.9