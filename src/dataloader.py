
import os
import gzip
import copy
import pickle
import torch
# import cv2
import glob

import PIL.Image
import pandas as pd
import numpy as np
import concurrent.futures

import mlbase.utilities.mlutilities as ml


def _convert_to_tensor(X, T):
    if not isinstance(X, torch.Tensor):
        X, T = list(map(lambda x: torch.from_numpy(
            x).float(), [X, T]))
    T = T.flatten().type(torch.LongTensor)
    return X, T


def _intel(data_dir, seed):
    class_names = ['mountain', 'street',
                   'glacier', 'buildings', 'sea', 'forest']

    def _load_data(dir, class_names):
        # TODO: import cv2
        # W, H = 128, 128
        # X, T = [], []
        # for t, class_name in enumerate(class_names):
        #     class_dir = os.path.join(dir, class_name)
        #     for file in os.listdir(class_dir):
        #         img = cv2.imread(os.path.join(class_dir, file))
        #         img = cv2.resize(cv2.cvtColor(
        #             img, cv2.COLOR_BGR2RGB), (W, H)) / 255.
        #         X.append(img)
        #         T.append(t)
        # return np.stack(X).transpose(0, 3, 1, 2), np.array(T).reshape(-1, 1)
        return None, None

    train_dir = os.path.join(data_dir, 'seg_train')
    test_dir = os.path.join(data_dir, 'seg_test')

    Xtrain, Ttrain = _load_data(train_dir, class_names)
    Xtest, Ttest = _load_data(test_dir, class_names)

    Xtrain, Ttrain, Xval, Tval = ml.partition(Xtrain, Ttrain, 0.80, shuffle=True,
                                              classification=True, seed=seed)

    print(np.unique(Ttrain, return_counts=True),
          np.unique(Tval, return_counts=True),
          np.unique(Ttest, return_counts=True), sep='\n')

    print(Xtrain.shape, Xval.shape, Xtest.shape, sep='\n')

    return Xtrain, Ttrain, Xval, Tval, Xtest, Ttest


def _tc_partition_data_by_storm(train_df, test_df, train_fraction, seed):
    storms = train_df.storm_id.unique()
    n_storms = len(storms)
    train_frac = int(n_storms * train_fraction)
    inds = np.arange(n_storms)

    np.random.seed(seed)
    np.random.shuffle(inds)

    train_storm_ids = storms[inds[:train_frac]]
    val_storm_ids = storms[inds[train_frac:]]

    train = train_df.loc[train_df.storm_id.isin(
        train_storm_ids)].reset_index(drop=True)
    val = train_df.loc[train_df.storm_id.isin(
        val_storm_ids)].reset_index(drop=True)

    return train, val, test_df


def _tc_read_images(files, threads=12):
    """Read images by filenames to a numpy array

    N = number of images
    W = width of images
    H = height of images
    C = number of channels 

    :param files: list of files to read
    :param threads: number of threads to use for reading
    :return: images: (N,W,H,C) numpy array of images
    """
    W, H = 128, 128  # global width and height for image resolution

    def _reading_thread(files, n_images, section):
        """Helper function for reading in different threads"""
        for i, f in enumerate(files[section:section+n_images]):
            with PIL.Image.open(f).convert('L') as im:
                # convert from 366 x 366 to ...
                images[i +
                       section] = np.expand_dims(im.resize((W, H)), axis=0) / 255.
    images = np.zeros((len(files), 1, W, H))
    n_images = len(files) // threads

    with concurrent.futures.ThreadPoolExecutor(max_workers=threads) as executor:
        for section in [n_images * i for i in range(threads + 1)]:
            executor.submit(_reading_thread, files, n_images, section)

    return images


def _tc_load_data(train, val, test):
    """Convert DataFrame to numpy array for training, 
    validation, and test datasets

    F = number of target output features

    :param train: _training_ df with file_name and wind_speed variables 
    :param val: _validation_ df with file_name and wind_speed variables 
    :param test: _test_ df with file_name and wind_speed variables 
    :return: X_: (N,W,H,C) numpy array of _ images
    :return: T_: (N,F) numpy array of _ targets
    """
    Xtrain = _tc_read_images(train.file_name.values)
    Ttrain = train.wind_speed.values.reshape(-1, 1)

    Xval = _tc_read_images(val.file_name.values)
    Tval = val.wind_speed.values.reshape(-1, 1)

    Xtest = _tc_read_images(test.file_name.values)
    Ttest = test.wind_speed.values.reshape(-1, 1)

    return Xtrain, Ttrain, Xval, Tval, Xtest, Ttest


def _tc_categorize(targets):
    """
    Saffir-Simpson Hurricane Wind Scale
    1: 64-82 kt
    2: 83-95 kt
    3: 96-112 kt
    4: 113-136 kt
    5: 137 kt or higher
    """
    T = copy.deepcopy(targets)
    # T[T < 50] = -1
    # T[(T >= 50) & (T <= 63)] = 0
    # T[(T >= 64) & (T <= 82)] = 1
    # T[(T >= 83) & (T <= 95)] = 2
    # T[(T >= 96) & (T <= 112)] = 3
    # T[(T >= 113) & (T <= 136)] = 4
    # T[T >= 137] = 5

    # three classes
    T[T < 50] = -1
    T[(T >= 50) & (T <= 63)] = 0
    T[(T >= 64) & (T <= 95)] = 1
    T[T > 95] = 2

    # binary
    # T[T < 45] = -1
    # T[(T >= 45) & (T <= 64)] = 0
    # T[T > 64] = 1
    return T


def _tc(data_dir, seed):
    train_df = pd.read_csv(os.path.join(data_dir, 'train_metadata.csv'))
    test_df = pd.read_csv(os.path.join(data_dir, 'test_metadata.csv'))
    train, val, test = _tc_partition_data_by_storm(
        train_df, test_df, train_fraction=0.80, seed=seed)
    Xtrain, Ttrain, Xval, Tval, Xtest, Ttest = _tc_load_data(train, val, test)

    # convert regression to classification
    Ttrain = _tc_categorize(Ttrain)
    inds = np.where(Ttrain == -1)
    Xtrain = np.delete(Xtrain, inds, 0)
    Ttrain = np.delete(Ttrain, inds, 0).astype(int)

    Tval = _tc_categorize(Tval)
    inds = np.where(Tval == -1)
    Xval = np.delete(Xval, inds, 0)
    Tval = np.delete(Tval, inds, 0).astype(int)

    Ttest = _tc_categorize(Ttest)
    inds = np.where(Ttest == -1)
    Xtest = np.delete(Xtest, inds, 0)
    Ttest = np.delete(Ttest, inds, 0).astype(int)

    print(np.unique(Ttrain, return_counts=True))
    print(np.unique(Tval, return_counts=True))
    print(np.unique(Ttest, return_counts=True))

    return Xtrain, Ttrain, Xval, Tval, Xtest, Ttest


def _cifar(data_dir, seed):
    def load(file_name):
        data = []

        for file in glob.glob(os.path.join(data_dir, file_name)):
            with open(file, 'rb') as fo:
                data.append(pickle.load(fo, encoding='bytes'))

        images = np.array(data[0][b'data'])
        labels = np.array(data[0][b'labels'])

        if len(data) > 1:
            for i in range(1, len(data)):
                images = np.concatenate((images, data[i][b'data']), axis=0)
                labels = np.concatenate((labels, data[i][b'labels']), axis=0)

        return (images.reshape(-1, 3, 32, 32)/255.0).astype(np.float32), labels.reshape((-1, 1))

    Xtrain, Ttrain = load('data_batch_*')
    Xtest, Ttest, = load('test_batch')

    Xtrain, Ttrain, Xval, Tval = ml.partition(Xtrain, Ttrain, 0.80, shuffle=True,
                                              classification=True, seed=seed)

    print(np.unique(Ttrain, return_counts=True),
          np.unique(Tval, return_counts=True),
          np.unique(Ttest, return_counts=True), sep='\n')

    print(Xtrain.shape, Xval.shape, Xtest.shape, sep='\n')

    with open(os.path.join(data_dir, 'batches.meta'), 'rb') as f:
        class_names = pickle.load(f, encoding='bytes')
    class_names = class_names[b'label_names']

    return Xtrain, Ttrain, Xval, Tval, Xtest, Ttest


def _cluttered(data_dir):
    dim = 60
    mnist_cluttered = 'mnist_cluttered_60x60_6distortions.npz'

    data = np.load(os.path.join(data_dir, mnist_cluttered))
    Xtrain, Ttrain = data['x_train'].reshape(
        (-1, 1, dim, dim)), np.argmax(data['y_train'], axis=-1).reshape(-1, 1)
    Xval, Tval = data['x_valid'].reshape(
        (-1, 1, dim, dim)), np.argmax(data['y_valid'], axis=-1).reshape(-1, 1)
    Xtest, Ttest = data['x_test'].reshape(
        (-1, 1, dim, dim)), np.argmax(data['y_test'], axis=-1).reshape(-1, 1)

    return Xtrain, Ttrain, Xval, Tval, Xtest, Ttest


def _mnist(data_dir):
    with gzip.open(os.path.join(data_dir, 'mnist.pkl.gz'), 'rb') as f:
        train_set, val_set, test_set = pickle.load(f, encoding='latin1')

    Xtrain = train_set[0].reshape((-1, 1, 28, 28))
    Ttrain = train_set[1].reshape((-1, 1))

    Xval = val_set[0].reshape((-1, 1, 28, 28))
    Tval = val_set[1].reshape((-1, 1))

    Xtest = test_set[0].reshape((-1, 1, 28, 28))
    Ttest = test_set[1].reshape((-1, 1))

    return Xtrain, Ttrain, Xval, Tval, Xtest, Ttest


def get_dataset(config, test=False, all=False):
    data_name = config['data_name']
    data_dir = config['data_dir']

    if data_name == 'mnist':
        Xtrain, Ttrain, Xval, Tval, Xtest, Ttest = _mnist(data_dir)
    elif data_name == 'cluttered':
        Xtrain, Ttrain, Xval, Tval, Xtest, Ttest = _cluttered(data_dir)
    elif data_name == 'tc':
        Xtrain, Ttrain, Xval, Tval, Xtest, Ttest = _tc(
            data_dir, config['seed'])
    elif data_name == 'intel':
        Xtrain, Ttrain, Xval, Tval, Xtest, Ttest = _intel(
            data_dir, config['seed'])
    elif data_name == 'cifar':
        Xtrain, Ttrain, Xval, Tval, Xtest, Ttest = _cifar(
            data_dir, config['seed'])

    if all:
        return Xtrain, Ttrain, Xval, Tval, Xtest, Ttest

    if not test:
        Xtrain, Ttrain = _convert_to_tensor(Xtrain, Ttrain)
        Xval, Tval = _convert_to_tensor(Xval, Tval)
        return Xtrain, Ttrain, Xval, Tval
    else:
        Xtest, Ttest = _convert_to_tensor(Xtest, Ttest)
        return Xtest, Ttest


if __name__ == "__main__":
    Xtrain, Ttrain, Xval, Tval, Xtest, Ttest = _mnist(
        '/s/chopin/l/grad/stock/nvme/data/cs/')
    print(Xtrain.shape, Ttrain.shape, Xval.shape,
          Tval.shape, Xtest.shape, Ttest.shape)
    print(Xtrain.max(), Xtrain.min(), np.unique(Ttrain, return_counts=True))

    Xtrain, Ttrain, Xval, Tval, Xtest, Ttest = _tc(
        '/s/chopin/l/grad/stock/nvme/data/ai2es/mlhub/nasa_tc', seed=1234)
    print(Xtrain.shape, Ttrain.shape, Xval.shape,
          Tval.shape, Xtest.shape, Ttest.shape)
    print(Xtrain.max(), Xtrain.min(), np.unique(Ttrain, return_counts=True))
