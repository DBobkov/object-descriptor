import numpy as np
import glob
import os

use_arch_elements_stanford = False # whether we use architectural elements in the Stanford dataset (walls, beams, floor, ceiling)

def get_list_strings_from_dict(dict_labels):
    '''
    Get list strings from dictionary
    :param dict_labels:
    :return:
    '''
    label_strings = list(dict_labels.keys())
    label_strings.sort()
    for i in range(len(label_strings) - 1):
        i1 = dict_labels.get(label_strings[i])
        i2 = dict_labels.get(label_strings[i + 1])
        assert (i2 > i1)

    return label_strings

class Dataset:
    def __init__(self, input_dir, dict_labels, prefix = None):
        '''
        Dataset
        :param input_dir:
        :param dict_labels:
        :param prefix:
        '''

        if prefix is not None:
            train_feat_fn = input_dir + prefix + "train_feat.npy"
            train_labels_fn = input_dir + prefix + "train_labels.npy"
            test_feat_fn = input_dir + prefix + "test_feat.npy"
            test_labels_fn = input_dir + prefix + "test_labels.npy"
        else:
            train_feat_fn = input_dir + "train_feat.npy"
            train_labels_fn = input_dir + "train_labels.npy"
            test_feat_fn = input_dir + "test_feat.npy"
            test_labels_fn = input_dir + "test_labels.npy"

        self.train_feat = np.load(train_feat_fn)
        self.train_label = np.load(train_labels_fn)
        self.test_feat = np.load(test_feat_fn)
        self.test_label = np.load(test_labels_fn)
        self.list_files_fn = ''

        # init list of string with labels
        self.label_strings = get_list_strings_from_dict(dict_labels)

def get_list_files_folder(folder, extension):
    '''
    Walk through the folder and get list of files
    :param folder:
    :param extension:
    :return:
    '''
    ### Caution, this goes inside the sub-folders!!!
    list_files = []
    for root, dirs, files in os.walk(folder):
        for file in files:
            if file.endswith(extension):
                name = os.path.join(root, file)
                list_files.append(name)

    N_files = len(list_files)
    print('Found %d files in %s' % (N_files, folder))
    assert (N_files > 0)
    return list_files


############ Definition of labels for different datasets
m40_label_to_int_dict = {
    'airplane': 1,
    'bathtub' : 2,
    'bed':      3,
    'bench':     4,
    'bookshelf': 5,
    'bottle': 6,
    'bowl': 7,
    'car': 8,
    'chair': 9,
    'cone': 10,
    'cup': 11,
    'curtain': 12,
    'desk': 13,
    'door': 14,
    'dresser': 15,
    'flower': 16,
    'glass': 17,
    'guitar': 18,
    'keyboard': 19,
    'lamp': 20,
    'laptop': 21,
    'mantel': 22,
    'monitor': 23,
    'night': 24,
    'person': 25,
    'piano': 26,
    'plant': 27,
    'radio': 28,
    'range': 29,
    'sink': 30,
    'sofa': 31,
    'stairs': 32,
    'stool': 33,
    'table': 34,
    'tent': 35,
    'toilet': 36,
    'tv': 37,
    'vase': 38,
    'wardrobe': 39,
    'xbox': 40,
}

if use_arch_elements_stanford:
    stanford_label_to_int_dict = {
        'beam': 1,
        'board' : 2,
        'bookcase': 3,
        'ceiling': 4,
        'chair': 5,
        'column': 6,
        'door': 7,
        'floor': 8,
        'sofa': 9,
        'stairs': 10,
        'table': 11,
        'wall': 12,
        'window': 13,
    }
else:
    stanford_label_to_int_dict = {
        'beam': 1,
        'board' : 2,
        'bookcase': 3,
        'chair': 4,
        'column': 5,
        'door': 6,
        'sofa': 7,
        'stairs': 8,
        'table': 9,
        'window': 10,
}

scannet_label_to_int = {
    'basket': 1,
    'bathtub': 2,
    'bed': 3,
    'cabinet': 4,
    'chair': 5,
    'keyboard': 6,
    'lamp': 7,
    'microwave': 8,
    'pillow': 9,
    'printer': 10,
    'shelf': 11,
    'stove': 12,
    'table': 13,
    'tv': 14,
}


def m40_label_to_int(label):
    return m40_label_to_int_dict[label]

def m40_int_to_label(l_id):
    key = next(key for key, value in m40_label_to_int_dict.items() if value == l_id)
    return key



def getDictFromInt(dataset_int):
    '''
    Depending on dataset_int, we get dataset dictionary for mapping labeling to integer indeces
    :param dataset_int:
    :return:
    '''
    if dataset_int == 0:
        return stanford_label_to_int_dict
    elif dataset_int == 1:
        return m40_label_to_int_dict
    elif dataset_int == 2:
        return scannet_label_to_int
    else:
        print('Error!')

def get_labels_from_folder(folder, dataset_int, tophat):
    '''
    Get labels from folder based on file namings
    :param folder:
    :param dataset_int:
    :param tophat:
    :return:
    '''

    NUM_CLASSES = len(getDictFromInt(dataset_int))

    list_files = get_list_files_folder(folder, ".pcd")
    N_files = len(list_files)

    if tophat:
        label_set = np.zeros((N_files, NUM_CLASSES), dtype='uint8')
    else:
        label_set = np.zeros((N_files,1), dtype='uint8')

    dict_dataset = getDictFromInt(dataset_int)

    for file_id in range(N_files):
        label = get_label_from_filename(list_files[file_id], dataset_int)

        l_id = int(dict_dataset[label]) - 1

        if tophat:
            label_set[file_id, l_id] = 1
        else:
            label_set[file_id] = l_id
    return label_set


def stanford_label_to_int(label):
    return int(stanford_label_to_int_dict[label])

def stanford_int_to_label(l_id):
    key = next(key for key, value in stanford_label_to_int_dict.items() if value == l_id)
    return key

def get_label_from_filename(fn, dataset_int):
    '''
    Get label from filename based on certain convention
    :param fn:
    :param dataset_int:
    :return:
    '''
    base = os.path.basename(fn)
    if dataset_int!=2:
        label = base.rsplit('_')[0]
    else:
        label = base.split('_')[0]
    return label

def preallocate( folders ):
    '''
    For efficient pre-allocation, we can already pre-compute the number of objects we have in our dataset
    :param folders:
    :return:
    '''

    tmp = folders[0]
    tmp_file = glob.glob(tmp)
    feat_desc = np.load(tmp_file[0])
    feat_size = feat_desc.size

    num_objs = 0
    for folder in folders:
        files = glob.glob(folder)
        assert (len(files) > 0)


        num_objs_this = len(files)
        num_objs += num_objs_this

    return feat_size, num_objs


def read_npy_from_directory_and_sort_by_labels(path, input_areas, dataset_int):
    '''
    Read files and sort by labels
    :param path:
    :param input_areas:
    :param dataset_int:
    :return:
    '''
    area_num = len(input_areas)

    for idx in range(area_num):
        temp_str = [path + input_areas[idx] + '/*.npy']
        if idx==0:
            folders = temp_str
        else:
            folders.extend(temp_str)

    dict_dataset = getDictFromInt(dataset_int)

    list_files = glob.glob(folders[0])
    print('List files size %d, path %s, %s' % (len(list_files), path, folders[0]))

    ### pre-allocate, find out feat size
    feat_size, num_objs = preallocate(folders)
    feature_set = np.zeros((num_objs, feat_size), dtype=float)
    label_set = np.zeros((num_objs,1), dtype=int)

    file_counter = 0
    for folder in folders:
        files = glob.glob(folder)
        assert(len(files)>0)
        print("Processing folder %s, there are %d files" % (folder, len(files)))

        for file in files:
            if file_counter % 100  == 0:
                print('Progress %d/%d' % (file_counter, len(files)))

            feat_desc = np.load(file)
            feat_desc = feat_desc.transpose()
            label = get_label_from_filename(file, dataset_int)

            l_id = int(dict_dataset.get(label, 0))

            label_this = l_id - 1

            feature_set[file_counter, :] = feat_desc
            label_set[file_counter] = label_this

            file_counter += 1

    return feature_set, label_set, list_files

class Fold:
    train_areas = []
    test_areas = []